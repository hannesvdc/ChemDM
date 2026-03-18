import torch as pt
import torch.nn as nn

from chemdm.AtomicAndGraphInformation import AtomicAndGraphInformation
from chemdm.MLP import MultiLayerPerceptron

from typing import List, Set

class MolecularEmbeddingNetwork( nn.Module ):

    def __init__( self, 
                  state_size : int, # The number of features per node
                  d_cutoff : float,
                  n_layers : int,
                  message_size : int,
                  ) -> None:
        super().__init__()
        self.state_size = state_size # c
        self.message_size = message_size # m
        self.d_cutoff = d_cutoff
        self.n_layers = n_layers

        # Embedding of atomic information
        self.atomic_information = AtomicAndGraphInformation()
        info_neurons_per_layer = [self.atomic_information.numberOfOutputs(), 64, self.state_size]
        self.atomic_embedding = MultiLayerPerceptron( info_neurons_per_layer, nn.GELU, "embedding_atomic_info" )

        # Can we compute this? 1 + n_distance_embeddings?
        self.n_edge_features = 3

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
            update_network = MultiLayerPerceptron( update_neurons_per_layer, nn.GELU, f"update_layer_{l}")
            state_update_networks.append( update_network )
        self.message_networks = nn.ModuleList( message_networks )
        self.state_update_networks = nn.ModuleList( state_update_networks )
        
    def forward( self,
                 Z : pt.Tensor, # (N,)
                 x : pt.Tensor, # (N,3)
                 G : List[Set[int]],
               ) -> pt.Tensor:
        Z = Z.flatten()
        N = len(Z)
        assert x.shape[0] == N, f"The number of atoms in `x` and `Z` must match, but got shapes {N} and {x.shape}."
        assert x.ndim == 2 and x.shape[1] == 3, f"`x` must have shape (N,3) but got {x.shape}."
        assert len(G) == N, f"The number of entries in the bond graph `G` must match the number of atoms, but got {len(G)}"

        # Put the graph in a different datastructure
        edge_list = [[i, j] for i, neighbors in enumerate(G) for j in neighbors]
        if len(edge_list) == 0:
            g_indices = pt.empty( (0,2), dtype=pt.long, device=x.device )
        else:
            g_indices = pt.tensor( edge_list, dtype=pt.long, device=x.device )
        bond_mask = pt.zeros( (N, N), dtype=pt.bool, device=x.device )
        bond_mask[g_indices[:,0], g_indices[:,1]] = True
        bond_mask[g_indices[:,1], g_indices[:,0]] = True # for symmetry
        bond_mask.fill_diagonal_(False) # remove self-bonds

        # Calculate the atomic embedding
        Z_info = self.atomic_information( Z, G ) # (N, info)
        h = self.atomic_embedding( Z_info ) # shape (N, c)

        # Construct the edge features
        distances = pt.norm( x[:,None,:] - x[None,:,:], dim=2 )
        edge_features = pt.stack( (bond_mask, distances, distances**2), dim=2 ) # (N,N,3)

        # Construct the set of all neighbors, they are fixed for the embedding network.
        cutoff_mask = distances <= self.d_cutoff
        cutoff_mask.fill_diagonal_(False)
        neighbor_mask = bond_mask | cutoff_mask

        # move through all the layers of the GNN
        for l in range( self.n_layers ):
            # Concatenate h_i, h_j and e_ij
            h_i = h[:, None, :]   # (N, 1, c)
            h_j = h[None, :, :]   # (1, N, c)
            pair_feat = pt.cat([ h_i.expand(N, N, self.state_size), h_j.expand(N, N, self.state_size), edge_features ], dim=2)

            # Create the edge messages
            messages = self.message_networks[l]( pair_feat ) # (N, N, m)
            neighbor_messages = neighbor_mask[:,:,None] * messages

            # Create nodal messages and update
            m_i = pt.sum( neighbor_messages, dim=1 )
            h = h + self.state_update_networks[l]( pt.cat( (h, m_i), dim=1) )

        # The end result is a new state of shape (N, c)
        return h