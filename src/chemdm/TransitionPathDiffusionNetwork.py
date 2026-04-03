import torch as pt
import torch.nn as nn

from chemdm.AtomicOnlyInformation import AtomicInformation
from chemdm.MLP import MultiLayerPerceptron
from chemdm.MolecularEmbeddingNetwork import MolecularEmbeddingGNN
from chemdm.embedding import ArcLengthEmbedding
from chemdm.MoleculeGraph import Molecule, findAllNeighborsReactantProduct, recenterMolecule
from chemdm.DistanceRBFEmbedding import DistanceRBFEmbedding


class TransitionPathDiffusionGNN( nn.Module ):
    """
    Denoising graph neural network for transition path diffusion.

    Given noisy intermediate positions x_t, reactant and product graphs
    (xA, xB), normalized arclength s, and diffusion timestep t, predicts
    the clean positions x_0.

    The architecture mirrors TransitionPathGNN but with two key changes:
      1. Positions are initialized from the noisy input x_t instead of
         the linear interpolation (1-s)*xA + s*xB.
      2. A diffusion-timestep embedding is concatenated to the node state
         alongside the arclength embedding.
    """

    def __init__( self,
                  xA_embedding_network : MolecularEmbeddingGNN,
                  xB_embedding_network : MolecularEmbeddingGNN,
                  message_size : int,
                  n_layers : int,
                  d_cutoff : float,
                  n_s_freq : int = 8,
                  n_t_freq : int = 8,
                ) -> None:
        super().__init__()

        self.message_size = message_size
        self.n_layers = n_layers
        self.d_cutoff = d_cutoff

        # --- Atomic information embedding ---
        self.atom_information = AtomicInformation()
        self.atomic_information_outputs = 64
        info_neurons = [ self.atom_information.numberOfOutputs(), 64, self.atomic_information_outputs ]
        self.atom_info_embedding = MultiLayerPerceptron( info_neurons, nn.GELU, "embedding_atomic_info" )

        # --- Arclength embedding ---
        self.arclength_embedding = ArcLengthEmbedding( n_s_freq )

        # --- Diffusion timestep embedding ---
        self.timestep_embedding = ArcLengthEmbedding( n_t_freq )

        # --- Molecular embedding networks ---
        self.xA_embedding_network = xA_embedding_network
        self.xB_embedding_network = xB_embedding_network

        # Total node state size
        self.state_size = (
              self.xA_embedding_network.state_size
            + self.xB_embedding_network.state_size
            + self.atomic_information_outputs
            + self.arclength_embedding.getNumberOfFeatures()
            + self.timestep_embedding.getNumberOfFeatures()
        )

        # --- Edge features ---
        self.rbf = DistanceRBFEmbedding( 0.0, d_cutoff, n_rbf=10 )
        self.n_edge_features = 3 * self.rbf.out_dim + 7

        # --- Message-passing networks ---
        hidden_neurons = max( 64, self.message_size )
        message_neurons = [ 2 * self.state_size + self.n_edge_features,
                            hidden_neurons, hidden_neurons, self.message_size ]
        message_networks = []
        for l in range( self.n_layers ):
            message_networks.append(
                MultiLayerPerceptron( message_neurons, nn.GELU, f"message_layer_{l}" )
            )
        self.message_networks = nn.ModuleList( message_networks )

        # --- State update networks ---
        state_neurons = [ self.state_size + self.message_size,
                          hidden_neurons, hidden_neurons, self.state_size ]
        state_networks = []
        for l in range( self.n_layers ):
            state_networks.append(
                MultiLayerPerceptron( state_neurons, nn.GELU, f"state_layer_{l}", init_zero=True )
            )
        self.state_networks = nn.ModuleList( state_networks )

        # --- Equivariant position-update networks ---
        alpha_neurons = [ 2 * self.state_size + self.n_edge_features,
                          hidden_neurons, hidden_neurons, 1 ]
        beta_neurons  = [ self.state_size, hidden_neurons, hidden_neurons, 1 ]
        gamma_neurons = [ self.state_size, hidden_neurons, hidden_neurons, 1 ]

        alpha_networks, beta_networks, gamma_networks = [], [], []
        for l in range( self.n_layers ):
            alpha_networks.append(
                MultiLayerPerceptron( alpha_neurons, nn.GELU, f"alpha_layer_{l}", init_zero=True )
            )
            beta_networks.append(
                MultiLayerPerceptron( beta_neurons, nn.GELU, f"beta_layer_{l}", init_zero=True )
            )
            gamma_networks.append(
                MultiLayerPerceptron( gamma_neurons, nn.GELU, f"gamma_layer_{l}", init_zero=True )
            )
        self.alpha_networks = nn.ModuleList( alpha_networks )
        self.beta_networks  = nn.ModuleList( beta_networks )
        self.gamma_networks = nn.ModuleList( gamma_networks )

    def forward( self,
                 x_t : pt.Tensor,
                 xA  : Molecule,
                 xB  : Molecule,
                 s   : pt.Tensor,
                 t   : pt.Tensor,
               ) -> Molecule:
        """
        Predict clean positions x_0 from noisy positions x_t.

        Arguments
        ---------
        x_t : (N, 3)  noisy atomic positions at diffusion step t.
        xA  : Molecule reactant graph.
        xB  : Molecule product graph.
        s   : (N,)    normalized arclength, one value per atom (atoms in the
              same molecule share the same s).
        t   : (N,)    normalized diffusion timestep in [0, 1], one value per
              atom (atoms in the same molecule share the same value).

        Returns
        -------
        x_0_pred : Molecule with predicted clean positions.
        """
        assert pt.all( xA.Z == xB.Z ), f"`xA` and `xB` must have the same atoms in the same ordering."
        s = s.flatten()
        t = t.flatten()
        N = len( xA.Z )
        assert s.numel() == N, f"`s` must have the same number of elements as the number of atoms in xA and xB."
        assert t.numel() == N, f"`t` must have the same number of elements as the number of atoms in xA and xB."

        # Calculate the atomic embedding
        Z_info = self.atom_information( xA )
        atom_embedding = self.atom_info_embedding( Z_info )   # (N, 64)

        # Calculate the molecular embeddings
        hA = self.xA_embedding_network( xA )  # (N, c_A)
        hB = self.xB_embedding_network( xB )  # (N, c_B)

        # Arclength and diffusion-time embedding and combine
        s_embed = self.arclength_embedding( s ) # (N, 2*n_s_freq+2)
        t_embed = self.timestep_embedding( t ) # (N, 2*n_t_freq)
        if s_embed.ndim == 1:
            s_embed = s_embed[None, :]
        if t_embed.ndim == 1:
            t_embed = t_embed[None, :]
        h = pt.cat( (atom_embedding, hA, hB, s_embed, t_embed), dim=1 )


        # --- Initialize positions from the noisy input ---
        x = x_t
        for l in range( self.n_layers ):
            all_edges, is_bond_A, is_bond_B = findAllNeighborsReactantProduct(
                xA, xB, x, self.d_cutoff
            )
            src = all_edges[:, 0]
            dst = all_edges[:, 1]

            # Edge features (invariant scalars)
            dx   = x[src] - x[dst]
            dist = pt.sqrt( (dx * dx).sum(dim=1, keepdim=True) )
            rbf  = self.rbf( dist )

            dxA     = xA.x[src] - xA.x[dst]
            dist_xA = pt.sqrt( (dxA * dxA).sum(dim=1, keepdim=True) )
            rbf_A   = self.rbf( dist_xA )

            dxB     = xB.x[src] - xB.x[dst]
            dist_xB = pt.sqrt( (dxB * dxB).sum(dim=1, keepdim=True) )
            rbf_B   = self.rbf( dist_xB )

            edge_features = pt.cat( (
                is_bond_A[:, None], is_bond_B[:, None],
                dist, dist**2, dist_xA, dist_xB, dist_xA - dist_xB,
                rbf, rbf_A, rbf_B,
            ), dim=1 )

            # Compute the edge messages
            message_inputs = pt.cat( (h[src], h[dst], edge_features), dim=1 )
            messages = self.message_networks[l]( message_inputs )
            node_messages = pt.zeros( N, self.message_size, device=h.device, dtype=h.dtype )
            node_messages.index_add_( 0, dst, messages )

            # State update
            update_inputs = pt.cat((h, node_messages), dim=1)
            h = h + self.state_networks[l]( update_inputs )

            # Equivariant position update
            edge_inputs = pt.cat( (h[src], h[dst], edge_features), dim=1 )
            alpha = self.alpha_networks[l]( edge_inputs )
            beta  = self.beta_networks[l]( h )
            gamma = self.gamma_networks[l]( h )

            neighbor_update = pt.zeros_like( x )
            neighbor_update.index_add_( 0, dst, alpha * dx )

            x = x + neighbor_update \
                  + beta  * (1.0 - s[:, None]) * (xA.x - x) \
                  + gamma * s[:, None]         * (xB.x - x)

        # Wrap in a Molecule and enforce zero center of mass
        x_molecule = xA.copyWithNewPositions( x )
        x_molecule = recenterMolecule( x_molecule )
        return x_molecule
