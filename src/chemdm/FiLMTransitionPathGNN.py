import torch as pt
import torch.nn as nn

from chemdm.AtomicOnlyInformation import AtomicInformation
from chemdm.MLP import MultiLayerPerceptron
from chemdm.MolecularEmbeddingNetwork import MolecularEmbeddingGNN
from chemdm.embedding import ArcLengthEmbedding
from chemdm.MoleculeGraph import Molecule, findAllNeighborsReactantProduct, recenterMolecule
from chemdm.DistanceRBFEmbedding import DistanceRBFEmbedding


class FiLMTransitionPathGNN( nn.Module ):
    """
    Transition-path GNN with FiLM conditioning on reactant/product embeddings.

    Unlike TransitionPathGNN, the endpoint embeddings (hA, hB) are NOT
    concatenated into the node state.  Instead they enter exclusively
    through Feature-wise Linear Modulation (FiLM) at every message-passing
    layer.  The node state h is built from atomic information and arclength
    only; endpoint awareness is injected via per-atom (gamma, beta) produced
    by small MLPs that read (hA, hB, s_embed).

    Deterministic — no diffusion.
    """

    def __init__( self,
                  xA_embedding_network : MolecularEmbeddingGNN,
                  xB_embedding_network : MolecularEmbeddingGNN,
                  message_size : int,
                  n_layers : int,
                  d_cutoff : float,
                  n_freq : int = 8,
                ) -> None:
        super().__init__()

        self.message_size = message_size
        self.n_layers = n_layers
        self.d_cutoff = d_cutoff

        # Atomic information embedding
        self.atom_information = AtomicInformation()
        self.atomic_information_outputs = 64
        info_neurons = [ self.atom_information.numberOfOutputs(), 64, self.atomic_information_outputs ]
        self.atom_info_embedding = MultiLayerPerceptron( info_neurons, nn.GELU, "embedding_atomic_info" )

        # Arclength embedding
        self.arclength_embedding = ArcLengthEmbedding( n_freq )

        # Molecular embedding networks (used for FiLM conditioning only)
        self.xA_embedding_network = xA_embedding_network
        self.xB_embedding_network = xB_embedding_network

        # Node state: atomic info + arclength (no hA, hB)
        self.state_size = self.atomic_information_outputs + self.arclength_embedding.getNumberOfFeatures()

        # FiLM conditioning input: hA + hB + s_embed
        film_input_size = ( self.xA_embedding_network.state_size
                          + self.xB_embedding_network.state_size
                          + self.arclength_embedding.getNumberOfFeatures() )
        self.film_act = nn.GELU()

        # Edge features
        self.rbf = DistanceRBFEmbedding( 0.0, d_cutoff, n_rbf=10 )
        self.n_edge_features = 3 * self.rbf.out_dim + 7

        # Per-layer networks
        hidden_neurons = max( 64, self.message_size )

        message_networks = []
        state_networks = []
        film_networks = []
        alpha_networks = []
        beta_networks = []
        gamma_networks = []

        message_neurons = [ 2 * self.state_size + self.n_edge_features,
                            hidden_neurons, hidden_neurons, self.message_size ]
        state_neurons = [ self.state_size + self.message_size,
                          hidden_neurons, hidden_neurons, self.state_size ]
        film_neurons = [ film_input_size, hidden_neurons, 2 * self.state_size ]
        alpha_neurons = [ 2 * self.state_size + self.n_edge_features,
                          hidden_neurons, hidden_neurons, 1 ]
        beta_neurons = [ self.state_size, hidden_neurons, hidden_neurons, 1 ]
        gamma_neurons = [ self.state_size, hidden_neurons, hidden_neurons, 1 ]

        for l in range( self.n_layers ):
            message_networks.append( MultiLayerPerceptron( message_neurons, nn.GELU, f"message_layer_{l}" ) )
            state_networks.append( MultiLayerPerceptron( state_neurons, nn.GELU, f"state_layer_{l}", init_zero=True ) )
            film_networks.append( MultiLayerPerceptron( film_neurons, nn.GELU, f"film_layer_{l}", init_zero=True ) )
            alpha_networks.append( MultiLayerPerceptron( alpha_neurons, nn.GELU, f"alpha_layer_{l}", init_zero=True ) )
            beta_networks.append( MultiLayerPerceptron( beta_neurons, nn.GELU, f"beta_layer_{l}", init_zero=True ) )
            gamma_networks.append( MultiLayerPerceptron( gamma_neurons, nn.GELU, f"gamma_layer_{l}", init_zero=True ) )

        self.message_networks = nn.ModuleList( message_networks )
        self.state_networks = nn.ModuleList( state_networks )
        self.film_networks = nn.ModuleList( film_networks )
        self.alpha_networks = nn.ModuleList( alpha_networks )
        self.beta_networks = nn.ModuleList( beta_networks )
        self.gamma_networks = nn.ModuleList( gamma_networks )

    def forward( self,
                 xA : Molecule,
                 xB : Molecule,
                 s  : pt.Tensor,
               ) -> Molecule:
        assert pt.all( xA.Z == xB.Z ), \
            "`xA` and `xB` must have the same atoms in the same ordering."
        s = s.flatten()
        N = len( xA.Z )
        assert s.numel() == N

        # Compute embeddings (once)
        Z_info = self.atom_information( xA )
        atom_embedding = self.atom_info_embedding( Z_info )   # (N, 64)

        hA = self.xA_embedding_network( xA )  # (N, c_A)
        hB = self.xB_embedding_network( xB )  # (N, c_B)

        s_embed = self.arclength_embedding( s )
        if s_embed.ndim == 1:
            s_embed = s_embed[None, :]

        # Node state: atomic info + arclength only
        h = pt.cat( (atom_embedding, s_embed), dim=1 )

        # FiLM conditioning input (constant across layers)
        film_input = pt.cat( (hA, hB, s_embed), dim=1 )

        # Initial positions: linear interpolation
        x = (1.0 - s[:, None]) * xA.x + s[:, None] * xB.x

        # Message-passing layers with FiLM
        for l in range( self.n_layers ):
            all_edges, is_bond_A, is_bond_B = findAllNeighborsReactantProduct( xA, xB, x, self.d_cutoff )
            src = all_edges[:, 0]
            dst = all_edges[:, 1]

            # Edge features
            dx = x[src] - x[dst]
            dist = pt.sqrt( (dx * dx).sum(dim=1, keepdim=True) )
            rbf = self.rbf( dist )

            dxA = xA.x[src] - xA.x[dst]
            dist_xA = pt.sqrt( (dxA * dxA).sum(dim=1, keepdim=True) )
            rbf_A = self.rbf( dist_xA )

            dxB = xB.x[src] - xB.x[dst]
            dist_xB = pt.sqrt( (dxB * dxB).sum(dim=1, keepdim=True) )
            rbf_B = self.rbf( dist_xB )

            edge_features = pt.cat( (
                is_bond_A[:, None], is_bond_B[:, None],
                dist, dist**2, dist_xA, dist_xB, dist_xA - dist_xB,
                rbf, rbf_A, rbf_B,
            ), dim=1 )

            # Messages
            message_inputs = pt.cat( (h[src], h[dst], edge_features), dim=1 )
            messages = self.message_networks[l]( message_inputs )
            node_messages = pt.zeros( N, self.message_size, device=h.device, dtype=h.dtype )
            node_messages.index_add_( 0, dst, messages )

            # State update (residual)
            update_inputs = pt.cat( (h, node_messages), dim=1 )
            h = h + self.state_networks[l]( update_inputs )

            # FiLM modulation
            film_out = self.film_networks[l]( film_input )        # (N, 2 * state_size)
            film_gamma = film_out[:, :self.state_size]             # (N, state_size)
            film_beta  = film_out[:, self.state_size:]             # (N, state_size)
            h = self.film_act( (1.0 + film_gamma) * h + film_beta )  # FiLM nonlinearity to increase expressiveness

            # Equivariant position update
            edge_inputs = pt.cat( (h[src], h[dst], edge_features), dim=1 )
            alpha = self.alpha_networks[l]( edge_inputs )
            beta = self.beta_networks[l]( h )
            gamma = self.gamma_networks[l]( h )

            edge_updates = alpha * dx
            neighbor_update = pt.zeros_like( x )
            neighbor_update.index_add_( 0, dst, edge_updates )

            x = x + neighbor_update \
                  + beta  * (1.0 - s[:, None]) * (xA.x - x) \
                  + gamma * s[:, None]         * (xB.x - x)

        # Dirichlet boundary conditions
        base = (1.0 - s[:, None]) * xA.x + s[:, None] * xB.x
        correction = x - base
        x_final = base + s[:, None] * (1.0 - s[:, None]) * correction

        # Enforce zero center of mass
        x_molecule = xA.copyWithNewPositions( x_final )
        x_molecule = recenterMolecule( x_molecule )
        return x_molecule
