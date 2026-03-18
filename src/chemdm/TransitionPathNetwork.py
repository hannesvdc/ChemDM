import torch as pt
import torch.nn as nn

from chemdm.AtomicOnlyInformation import AtomicOnlyInformation
from chemdm.MLP import MultiLayerPerceptron
from chemdm.MolecularEmbeddingNetwork import MolecularEmbeddingNetwork
from chemdm.embedding import ArcLengthEmbedding

from typing import List, Set

class TransitionPathNetwork( nn.Module ):
    """
    Main neural network to predict transition paths, combining chemical information network,
    initial and final state embedding networks, arclength embedding network, and main graph neural network.
    """
    def __init__( self,
                  xA_embedding_network : MolecularEmbeddingNetwork,
                  xB_embedding_network : MolecularEmbeddingNetwork,
                  d_cutoff : float,
                  n_layers : int,
                  message_size : int,
                  n_freq : int = 8
                  ) -> None:
        super().__init__()

        self.d_cutoff = d_cutoff
        self.n_layers = n_layers
        self.n_freq = n_freq
        self.message_size = message_size

        # Embed information about the atoms.
        self.atomic_information = AtomicOnlyInformation()
        self.atomic_information_outputs = 64
        info_neurons_per_layer = [self.atomic_information.numberOfOutputs(), 64, self.atomic_information_outputs]
        self.atomic_info_embedding = MultiLayerPerceptron( info_neurons_per_layer, nn.GELU, "embedding_atomic_info" )

        # Arclength embedding
        self.arclength_embedding = ArcLengthEmbedding( self.n_freq )

        # Relevant Chemical Information
        self.xA_embedding_network = xA_embedding_network
        self.xB_embedding_network = xB_embedding_network
        self.state_size = self.xA_embedding_network.state_size \
                        + self.xB_embedding_network.state_size \
                        + self.atomic_information_outputs \
                        + self.arclength_embedding.getNumberOfFeatures()

        # Nonlinear Message and Nodal Update layers
        self.n_edge_features = 6
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
            state_network = MultiLayerPerceptron( state_neurons_per_layer, nn.GELU, f"state_layer_{l}")
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
            alpha_network = MultiLayerPerceptron( alpha_neurons_per_layer, nn.GELU, f"alpha_layer_{l}")
            alpha_networks.append( alpha_network )
            beta_network = MultiLayerPerceptron( beta_neurons_per_layer, nn.GELU, f"beta_layer_{l}")
            beta_networks.append( beta_network )
            gamma_network = MultiLayerPerceptron( gamma_neurons_per_layer, nn.GELU, f"gamma_layer_{l}")
            gamma_networks.append( gamma_network )
        self.alpha_networks = nn.ModuleList( alpha_networks )
        self.beta_networks = nn.ModuleList( beta_networks )
        self.gamma_networks = nn.ModuleList( gamma_networks )

    def create_bond_mask( self, 
                          G : List[Set[int]],
                          device : pt.device,
                        ) -> pt.Tensor:
        N = len(G)

        edge_list = [[i, j] for i, neighbors in enumerate(G) for j in neighbors]
        if len(edge_list) == 0:
            g_indices = pt.empty( (0,2), dtype=pt.long, device=device )
        else:
            g_indices = pt.tensor( edge_list, dtype=pt.long, device=device )

        bond_mask = pt.zeros( (N, N), dtype=pt.bool, device=device )
        bond_mask[g_indices[:,0], g_indices[:,1]] = True
        bond_mask[g_indices[:,1], g_indices[:,0]] = True # for symmetry
        bond_mask.fill_diagonal_(False) # remove self-bonds

        return bond_mask

    def forward( self,
                 Z : pt.Tensor, # (N,)
                 xA : pt.Tensor,
                 xB : pt.Tensor,
                 GA : List[Set[int]],
                 GB : List[Set[int]],
                 s : pt.Tensor,
               ) -> pt.Tensor:
        Z = Z.flatten()
        N = len(Z)
        assert len(GA) == N, f"The number of entries in the bond graph `GA` must match the number of atoms, but got {len(GA)}"
        assert len(GB) == N, f"The number of entries in the bond graph `GB` must match the number of atoms, but got {len(GB)}"
        assert xA.shape == (N, 3), f"`xA` must have shape (N,3) but got {xA.shape}."
        assert xB.shape == (N, 3), f"`xB` must have shape (N,3) but got {xB.shape}."

        # Put the graphs in a different datastructure
        bond_mask_A = self.create_bond_mask( GA, xA.device )
        bond_mask_B = self.create_bond_mask( GB, xB.device )
        distances_A = pt.norm( xA[:,None,:] - xA[None,:,:], dim=2 )
        distances_B = pt.norm( xB[:,None,:] - xB[None,:,:], dim=2 )
        
        # Calculate the atomic embedding
        Z_info = self.atomic_information( Z ) # (N, info)
        atom_embedding = self.atomic_info_embedding( Z_info ) # shape (N, c)

        # Calculate the molecular embeddings
        hA = self.xA_embedding_network( Z, xA, GA )
        hB = self.xB_embedding_network( Z, xB, GB )

        # Initial Nodal state
        s_embed = self.arclength_embedding(s)
        if s_embed.ndim == 1:
            s_embed = s_embed[None,:]
        s_embed = s_embed.expand(N, -1)
        h = pt.cat( (atom_embedding, hA, hB, s_embed), dim=1)
        x = (1.0 - s) * xA + s * xB

        # Iterate over the layers and update the states
        for l in range( self.n_layers ):
            # Calculate distances and construct the edge features
            direction = x[None,:,:] - x[:,None,:]
            distances = pt.norm( direction, dim=2 )
            edge_features = pt.stack( (bond_mask_A.to(x.dtype), 
                                       bond_mask_B.to(x.dtype), 
                                       distances, 
                                       distances**2, 
                                       distances_A, 
                                       distances_B), dim=2 ) # (N,N,e)

            # Construct the set of all neighbors
            cutoff_mask = (distances <= self.d_cutoff)
            cutoff_mask.fill_diagonal_(False)
            neighbor_mask = bond_mask_A | bond_mask_B | cutoff_mask

            # Create the messages
            h_i = h[:, None, :]   # (N, 1, c)
            h_j = h[None, :, :]   # (1, N, c)
            pair_feat = pt.cat([ h_i.expand(N, N, self.state_size), h_j.expand(N, N, self.state_size), edge_features ], dim=2) # (N,N,2C+e)

            # Update the positions equivariantly
            alpha = self.alpha_networks[l]( pair_feat )
            neighbor_alpha = neighbor_mask[:,:,None] * alpha
            beta = self.beta_networks[l]( h )
            gamma = self.gamma_networks[l]( h )
            x = x + pt.sum(neighbor_alpha * direction, dim=1) + beta * (xA - x) + gamma * (xB - x) # all (N,3)

            # Create the edge messages
            messages = self.message_networks[l]( pair_feat ) # (N, N, m)
            neighbor_messages = neighbor_mask[:,:,None] * messages
            m_i = pt.sum( neighbor_messages, dim=1 )
            h = h + self.state_networks[l]( pt.cat( (h, m_i), dim=1) )

        # Return just the position
        return x