import math
import torch as pt
import torch.nn as nn

from collections import OrderedDict
from typing import List

from chemdm.embedding import ArcLengthEmbedding

def _all_equal( l : List[int] ) -> bool:
    return len( set(l) ) <= 1

class RegressionNetwork( nn.Module ):

    def __init__(self, n_freq : int,
                       hidden_layers : List[int], 
                ) -> None:
        super().__init__()

        self.arc_embedding = ArcLengthEmbedding( n_freq=n_freq )
        n_embeds = self.arc_embedding.n_embeddings

        self.act = nn.GELU()

        # Hidden layers
        layers = []
        for n in range( len(hidden_layers) ):
            in_neurons = 4 + n_embeds if n == 0 else hidden_layers[n-1]
            out_neurons = hidden_layers[n]
            layers.append( nn.Linear(in_neurons, out_neurons, bias=True) )
        self.layers = nn.ModuleList( layers )

        # Output layer
        self.output_layer = nn.Linear( hidden_layers[-1], 2, bias=True )

    def forward(self, xA : pt.Tensor, # (B, 2)
                      xB : pt.Tensor, #(B,2)
                      s : pt.Tensor, # (B,)
                    ) -> pt.Tensor:
        if s.ndim > 1:
            s = s.flatten()
        assert xA.ndim == 2 and xA.shape[1] == 2, f"`xA` must have shape (B,2) but got {xA.shape}."
        assert xB.ndim == 2 and xB.shape[1] == 2, f"`xB` must have shape (B,2) but got {xB.shape}."
        assert _all_equal([ xA.shape[0], xB.shape[0], s.shape[0]]), \
            f"`u`, `xA`, `xB` and `s` must have the same leading (batch) dimension."

        # Time embedding
        s_embed = self.arc_embedding( s )
        
        # Apply the network layer, film and activations
        x = pt.cat( (xA, xB, s_embed), dim=1 ) # shape (B, 4 + 4*n_freq)
        for layer_idx in range( len(self.layers) ):
            x = self.layers[layer_idx](x)
            x = self.act( x )

        # Apply the final output layer
        x = self.output_layer( x )
        return x