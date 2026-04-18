import torch as pt
import torch.nn as nn

from chemdm.MoleculeGraph import Molecule, findAllNeighbors
from chemdm.AtomicAndGraphInformation import MoleculeInformation
from chemdm.MLP import MultiLayerPerceptron
from chemdm.DistanceRBFEmbedding import DistanceRBFEmbedding

class MolecularEmbeddingGNN( nn.Module ):
    def __init__( self,
                  state_size : int,
                  message_size : int,
                  n_layers : int,
                  d_cutoff : float,
                ) -> None:
        super().__init__()

        self.state_size = state_size # c
        self.message_size = message_size # m
        self.n_layers = n_layers
        self.d_cutoff = d_cutoff

        # Embedding of molecular information
        self.molecule_information = MoleculeInformation( )
        info_neurons_per_layer = [self.molecule_information.numberOfOutputs(), 64, self.state_size]
        self.molecule_info_embedding = MultiLayerPerceptron( info_neurons_per_layer, nn.GELU, "molecule_embedding" )

        # There must be a more principled way instead of hard-coding this.
        self.rbf = DistanceRBFEmbedding( 0.0, d_cutoff, n_rbf=10 )
        self.n_edge_features = 2 + self.rbf.out_dim

        # Nonlinear Message and Nodal Update layers
        hidden_message_neurons = max(64, self.message_size )
        message_neurons_per_layer = [ 2*self.state_size + self.n_edge_features, hidden_message_neurons, hidden_message_neurons, self.message_size ]
        message_networks = []
        hidden_update_size = max( 64, self.state_size )
        update_neurons_per_layer = [ self.state_size + self.message_size, hidden_update_size, hidden_update_size, self.state_size ]
        state_update_networks = []
        for l in range( n_layers ):
            message_network = MultiLayerPerceptron( message_neurons_per_layer, nn.GELU, f"message_layer_{l}")
            message_networks.append( message_network )
            update_network = MultiLayerPerceptron( update_neurons_per_layer, nn.GELU, f"update_layer_{l}", init_zero=True )
            state_update_networks.append( update_network )
        self.message_networks = nn.ModuleList( message_networks )
        self.state_update_networks = nn.ModuleList( state_update_networks )

    def forward( self, molecule : Molecule ) -> pt.Tensor:

        # Calculate the atomic embedding
        molecule_info = self.molecule_information( molecule ) # (N, info)
        h = self.molecule_info_embedding( molecule_info ) # shape (N, c)

        # Construct all neighbors
        all_edges, is_bond = findAllNeighbors( molecule, self.d_cutoff )
        src = all_edges[:,0]
        dst = all_edges[:,1]

        # Calculate the edge features
        dx = molecule.x[src] - molecule.x[dst] # (E,3)
        dist2 = (dx * dx).sum(dim=1, keepdim=True)
        dist = pt.sqrt( dist2 ) # (E,1)
        rbf_embedding = self.rbf( dist ) # shape ( E, n_rbf)
        edge_features = pt.cat( [is_bond[:,None], dist, rbf_embedding], dim=1)

        # move through all the layers of the GNN
        for l in range( self.n_layers ):

            # Compute the edge messages
            message_inputs = pt.cat( (h[src,:], h[dst,:], edge_features), dim=1 ) # (N_neighbors, 2*self.state_size + self.n_edge_features)
            messages = self.message_networks[l]( message_inputs ) # (N_neighbors, self.message_size)

            # Accumulate into node messages
            node_messages = pt.zeros(
                h.shape[0], self.message_size,
                device=h.device, dtype=h.dtype
            )
            node_messages.index_add_(0, dst, messages)
            
            # Update the nodal state 
            update_inputs = pt.cat((h, node_messages), dim=1)
            h = h + self.state_update_networks[l]( update_inputs )

        # The end result is a new state of shape (N, c)
        return h