import torch as pt
import torch.nn as nn

from typing import List

from chemdm.embedding import ArcLengthEmbedding
from Butane import cartesianToInternal

def _all_equal( l : List[int] ) -> bool:
    return len( set(l) ) <= 1

class EquivariantNetwork( nn.Module ):

    def __init__(self, n_freq : int,
                       hidden_layers : List[int], 
                ) -> None:
        super().__init__()
        self.n_atoms = 4
        self.n_features = 9 

        self.arc_embedding = ArcLengthEmbedding( n_freq=n_freq )
        n_embeds = self.arc_embedding.n_embeddings

        self.act = nn.GELU()

        # Hidden layers
        input_dim = 4 + 4 + 2 * n_embeds
        output_dim = self.n_atoms * self.n_features
        layers = []
        for n in range( len(hidden_layers) ):
            in_neurons = input_dim if n == 0 else hidden_layers[n-1]
            out_neurons = hidden_layers[n]
            layers.append( nn.Linear(in_neurons, out_neurons, bias=True) )
        self.layers = nn.ModuleList( layers )

        # Output layer
        self.output_layer = nn.Linear( hidden_layers[-1], output_dim, bias=True )
    
    def compute_equivariant_basis(self, 
                                  xA: pt.Tensor, # (B, 4, 3)
                                  xB: pt.Tensor
                                ) -> pt.Tensor:
        """
        xA, xB: (B, 4, 3), assumed centered
        returns basis: (B, 4, 9, 3)
        """
        B, n_atoms, _ = xA.shape

        # atomwise features
        f1 = xA
        f2 = xB
        f3 = xB - xA

        # bond vectors
        b1A = xA[:, 1, :] - xA[:, 0, :]
        b2A = xA[:, 2, :] - xA[:, 1, :]
        b3A = xA[:, 3, :] - xA[:, 2, :]

        b1B = xB[:, 1, :] - xB[:, 0, :]
        b2B = xB[:, 2, :] - xB[:, 1, :]
        b3B = xB[:, 3, :] - xB[:, 2, :]

        # repeat shared global vectors over atoms
        globals_ = [b1A, b2A, b3A, b1B, b2B, b3B]
        globals_ = [g[:, None, :].expand(B, n_atoms, 3) for g in globals_]

        basis = pt.stack([f1, f2, f3] + globals_, dim=2)   # (B, 4, 9, 3)
        return basis

    def forward(self, xA : pt.Tensor, # (B, 4, 3)
                      xB : pt.Tensor, # (B, 4, 3)
                      s : pt.Tensor, # (B,)
                    ) -> pt.Tensor:
        if s.ndim > 1:
            s = s.flatten()
        assert xA.ndim == 3 and xA.shape[1:] == (4, 3), f"`xA` must have shape (B,4,3) but got {xA.shape}."
        assert xB.ndim == 3 and xB.shape[1:] == (4, 3), f"`xB` must have shape (B,4,3) but got {xB.shape}."
        assert _all_equal([ xA.shape[0], xB.shape[0], s.shape[0]]), \
            f"`xA`, `xB` and `s` must have the same leading (batch) dimension."
        B = xA.shape[0]
        
        # Center xA and xB just to be sure.
        xA = xA - pt.mean( xA, dim=1, keepdim=True )
        xB = xB - pt.mean( xB, dim=1, keepdim=True )

        # Arclength embedding
        s_embed = self.arc_embedding( s )

        # Internal features of xA and xB
        xA_int = cartesianToInternal( xA )
        xB_int = cartesianToInternal( xB )
        
        # Apply the network layers
        x = pt.cat( (xA_int, xB_int, s_embed), dim=1 ) # shape (B, 4 + 4 + 2*n_freq)
        for layer_idx in range( len(self.layers) ):
            x = self.layers[layer_idx](x)
            x = self.act( x )

        # Apply the final output layer and extract the invariant coefficients
        x = self.output_layer( x )
        alpha = x.reshape( (B, self.n_atoms, self.n_features) ) # (B, 4, n_features)

        # Create the equivariant features 
        f = self.compute_equivariant_basis( xA, xB ) # shape (B, 4, n_features, 3)

        # Recenter the molecule
        xs = pt.sum( alpha[:,:,:,None] * f, dim=2 ) # (B, 4, 3)
        xs = xs - pt.mean( xs, dim=1, keepdim=True )

        return xs