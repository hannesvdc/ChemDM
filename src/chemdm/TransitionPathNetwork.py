import torch as pt
import torch.nn as nn

from chemdm.AtomicOnlyInformation import AtomicOnlyInformation, AtomicInformation
from chemdm.MLP import MultiLayerPerceptron
from chemdm.MolecularEmbeddingNetwork import MolecularEmbeddingNetwork, MolecularEmbeddingGNN
from chemdm.embedding import ArcLengthEmbedding
from chemdm.MoleculeGraph import Molecule, findAllNeighborsReactantProduct

from typing import List, Set

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
            state_network = MultiLayerPerceptron( state_neurons_per_layer, nn.GELU, f"state_layer_{l}", init_zero=True )
            state_networks.append( state_network )
        self.state_networks = nn.ModuleList( state_networks )

        # Position update networks
        alpha_networks = []
        beta_networks = []
        alpha_neurons_per_layer = [ 2*self.state_size + self.n_edge_features, hidden_neurons, hidden_neurons, 1 ]
        beta_neurons_per_layer = [ self.state_size, hidden_neurons, hidden_neurons, 1]
        for l in range( self.n_layers ):
            alpha_network = MultiLayerPerceptron( alpha_neurons_per_layer, nn.GELU, f"alpha_layer_{l}", init_zero=True)
            alpha_networks.append( alpha_network )
            beta_network = MultiLayerPerceptron( beta_neurons_per_layer, nn.GELU, f"beta_layer_{l}", init_zero=True )
            beta_networks.append( beta_network )
        self.alpha_networks = nn.ModuleList( alpha_networks )
        self.beta_networks = nn.ModuleList( beta_networks )

    def forward( self,
                 xA : Molecule,
                 xB : Molecule,
                 s : pt.Tensor,
               ) -> pt.Tensor:
        assert pt.all( xA.Z == xB.Z ), f"`xA` and `xB` must have the same atoms in the same ordering."
        N = len(xA.Z)

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
        s_embed = s_embed.expand(N, -1)
        h = pt.cat( (atom_embedding, hA, hB, s_embed), dim=1)
        base = (1.0 - s) * xA.x + s * xB.x
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
            dist2 = (dx * dx).sum(dim=1, keepdim=True)
            dist = pt.sqrt( dist2 )
            dxA = xA.x[src] - xA.x[dst]
            dist_xA = pt.sqrt( (dxA * dxA).sum(dim=1, keepdim=True) )
            dxB = xB.x[src] - xB.x[dst]
            dist_xB = pt.sqrt( (dxB * dxB).sum(dim=1, keepdim=True) )
            edge_features = pt.cat( (is_bond_A[:,None], is_bond_B[:,None], dist, dist2, dist_xA, dist_xB), dim=1 )

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
            edge_updates = alpha * dx
            neighbor_update = pt.zeros_like(x)                # (N, 3)
            neighbor_update.index_add_(0, dst, edge_updates)  # sum over neighbors j for each i = dst
            x = x + neighbor_update + beta * (base - x)       # (N, 3)


        # Return just the position
        return x

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
            state_network = MultiLayerPerceptron( state_neurons_per_layer, nn.GELU, f"state_layer_{l}", init_zero=True )
            state_networks.append( state_network )
        self.state_networks = nn.ModuleList( state_networks )

        # Position update networks
        alpha_networks = []
        beta_networks = []
        alpha_neurons_per_layer = [ 2*self.state_size + self.n_edge_features, hidden_neurons, hidden_neurons, 1 ]
        beta_neurons_per_layer = [ self.state_size, hidden_neurons, hidden_neurons, 1]
        for l in range( self.n_layers ):
            alpha_network = MultiLayerPerceptron( alpha_neurons_per_layer, nn.GELU, f"alpha_layer_{l}", init_zero=True)
            alpha_networks.append( alpha_network )
            beta_network = MultiLayerPerceptron( beta_neurons_per_layer, nn.GELU, f"beta_layer_{l}", init_zero=True )
            beta_networks.append( beta_network )
        self.alpha_networks = nn.ModuleList( alpha_networks )
        self.beta_networks = nn.ModuleList( beta_networks )

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
        base = (1.0 - s) * xA + s * xB
        x = base

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
            x = x + pt.sum(neighbor_alpha * direction, dim=1) + beta * (base - x) # all (N,3)

            # Create the edge messages
            messages = self.message_networks[l]( pair_feat ) # (N, N, m)
            neighbor_messages = neighbor_mask[:,:,None] * messages
            m_i = pt.sum( neighbor_messages, dim=1 )
            h = h + self.state_networks[l]( pt.cat( (h, m_i), dim=1) )

        # Return just the position
        return x