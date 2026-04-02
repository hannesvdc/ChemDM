import torch as pt
import torch.nn as nn

from chemdm.AtomicOnlyInformation import AtomicInformation
from chemdm.MLP import MultiLayerPerceptron
from chemdm.MolecularEmbeddingNetwork import MolecularEmbeddingGNN
from chemdm.embedding import ArcLengthEmbedding
from chemdm.MoleculeGraph import Molecule, findAllNeighborsReactantProduct, recenterMolecule
from chemdm.DistanceRBFEmbedding import DistanceRBFEmbedding

class TransitionPathGNN( nn.Module ):
    """
    Main graph neural network to predict transition paths, combining chemical information network,
    initial and final state embedding networks, arclength embedding network, and main graph neural network.
    """
    def __init__( self,
                  xA_embedding_network : MolecularEmbeddingGNN,
                  xB_embedding_network : MolecularEmbeddingGNN,
                  message_size : int,
                  n_layers : int,
                  d_cutoff : float,
                  n_freq : int = 8
                  ) -> None:
        super().__init__()

        self.message_size = message_size
        self.n_layers = n_layers
        self.d_cutoff = d_cutoff
        self.n_freq = n_freq

        # Embed information about the atoms.
        self.atom_information = AtomicInformation()
        self.atomic_information_outputs = 64
        info_neurons_per_layer = [self.atom_information.numberOfOutputs(), 64, self.atomic_information_outputs]
        self.atom_info_embedding = MultiLayerPerceptron( info_neurons_per_layer, nn.GELU, "embedding_atomic_info" )

        # Arclength embedding
        self.arclength_embedding = ArcLengthEmbedding( self.n_freq )

        # Relevant Chemical Information
        self.xA_embedding_network = xA_embedding_network
        self.xB_embedding_network = xB_embedding_network
        self.state_size = self.xA_embedding_network.state_size \
                        + self.xB_embedding_network.state_size \
                        + self.atomic_information_outputs \
                        + self.arclength_embedding.getNumberOfFeatures()

        # Edge features
        self.rbf = DistanceRBFEmbedding( 0.0, d_cutoff, n_rbf=10 )
        self.n_edge_features = 3 * self.rbf.out_dim + 7

        # Nonlinear Message and Nodal Update layers
        hidden_neurons = max(64, self.message_size )
        message_neurons_per_layer = [ 2*self.state_size + self.n_edge_features, hidden_neurons, hidden_neurons, self.message_size ]
        message_networks = []
        for l in range( self.n_layers ):
            message_network = MultiLayerPerceptron( message_neurons_per_layer, nn.GELU, f"message_layer_{l}")
            message_networks.append( message_network )
        self.message_networks = nn.ModuleList( message_networks )

        # Feature update networks
        state_networks = []
        state_neurons_per_layer = [ self.state_size + self.message_size, hidden_neurons, hidden_neurons, self.state_size ]
        for l in range( self.n_layers ):
            state_network = MultiLayerPerceptron( state_neurons_per_layer, nn.GELU, f"state_layer_{l}", init_zero=True )
            state_networks.append( state_network )
        self.state_networks = nn.ModuleList( state_networks )

        # Position update networks
        alpha_networks = []
        beta_networks = []
        gamma_networks = []
        alpha_neurons_per_layer = [ 2*self.state_size + self.n_edge_features, hidden_neurons, hidden_neurons, 1 ]
        beta_neurons_per_layer = [ self.state_size, hidden_neurons, hidden_neurons, 1]
        gamma_neurons_per_layer = [ self.state_size, hidden_neurons, hidden_neurons, 1]
        for l in range( self.n_layers ):
            alpha_network = MultiLayerPerceptron( alpha_neurons_per_layer, nn.GELU, f"alpha_layer_{l}", init_zero=True)
            alpha_networks.append( alpha_network )
            beta_network = MultiLayerPerceptron( beta_neurons_per_layer, nn.GELU, f"beta_layer_{l}", init_zero=True )
            beta_networks.append( beta_network )
            gamma_network = MultiLayerPerceptron( gamma_neurons_per_layer, nn.GELU, f"gamma_layer_{l}", init_zero=True )
            gamma_networks.append( gamma_network )
        self.alpha_networks = nn.ModuleList( alpha_networks )
        self.beta_networks = nn.ModuleList( beta_networks )
        self.gamma_networks = nn.ModuleList( gamma_networks )

    def forward( self,
                 xA : Molecule,
                 xB : Molecule,
                 s : pt.Tensor,
               ) -> Molecule:
        assert pt.all( xA.Z == xB.Z ), f"`xA` and `xB` must have the same atoms in the same ordering."
        s = s.flatten()
        assert s.numel() == len(xA.Z), f"`s` must have the same number of elements as the number of atoms in xA and xB."
        N = len( xA.Z )

        # Calculate the atomic embedding
        Z_info = self.atom_information( xA ) # (N, info)
        atom_embedding = self.atom_info_embedding( Z_info ) # shape (N, c)

        # Calculate the molecular embeddings
        hA = self.xA_embedding_network( xA )
        hB = self.xB_embedding_network( xB )

        # Arclength embedding and combine
        s_embed = self.arclength_embedding(s)
        if s_embed.ndim == 1:
            s_embed = s_embed[None,:]
        h = pt.cat( (atom_embedding, hA, hB, s_embed), dim=1)
        base = (1.0 - s[:,None]) * xA.x + s[:,None] * xB.x
        x = base.clone()

        # Iterate over the layers and update the states
        for l in range( self.n_layers ):
            all_edges, is_bond_A, is_bond_B = findAllNeighborsReactantProduct( xA, xB, x, self.d_cutoff)
            src = all_edges[:,0]
            dst = all_edges[:,1]

            ## Our method uses a staggered updating scheme to pass more information.
            # (h, x) -> h_new
            # (h_new, x) -> x_new

            # Calculate the edge features
            dx = x[src] - x[dst]
            dist = pt.sqrt( (dx * dx).sum(dim=1, keepdim=True) )
            rbf_embedding = self.rbf( dist )
            dxA = xA.x[src] - xA.x[dst]
            dist_xA = pt.sqrt( (dxA * dxA).sum(dim=1, keepdim=True) )
            rbf_A = self.rbf( dist_xA )
            dxB = xB.x[src] - xB.x[dst]
            dist_xB = pt.sqrt( (dxB * dxB).sum(dim=1, keepdim=True) )
            rbf_B = self.rbf( dist_xB )
            edge_features = pt.cat( (is_bond_A[:,None], 
                                     is_bond_B[:,None], 
                                     dist, dist**2, dist_xA, dist_xB, dist_xA - dist_xB, 
                                     rbf_embedding, rbf_A, rbf_B), dim=1 )

            # Compute the edge messages
            message_inputs = pt.cat( (h[src,:], h[dst,:], edge_features), dim=1 )
            messages = self.message_networks[l]( message_inputs ) # (N_neighbors, self.message_size)
            node_messages = pt.zeros(
                N, self.message_size,
                device=h.device, dtype=h.dtype
            )
            node_messages.index_add_(0, dst, messages)
            
            # Update the nodal state 
            update_inputs = pt.cat((h, node_messages), dim=1)
            h = h + self.state_networks[l]( update_inputs )

            # Update the positions equivariantly
            edge_inputs = pt.cat( (h[src,:], h[dst,:], edge_features), dim=1 )
            alpha = self.alpha_networks[l]( edge_inputs )
            beta = self.beta_networks[l]( h )
            gamma = self.gamma_networks[l]( h )
            edge_updates = alpha * dx
            neighbor_update = pt.zeros_like(x)                # (N, 3)
            neighbor_update.index_add_(0, dst, edge_updates)  # sum over neighbors j for each i = dst
            x = x + neighbor_update + beta  * (1.0 - s[:, None]) * (xA.x - x) \
                                    + gamma * s[:, None]         * (xB.x - x) # (N, 3)

        # Ensure a zero center of mass
        x_molecule = xA.copyWithNewPositions( x )
        x_molecule = recenterMolecule( x_molecule )
        return x_molecule