import math
import torch as pt
import torch.nn as nn

from collections import OrderedDict
from typing import List

def _all_equal( l : List[int] ) -> bool:
    return len( set(l) ) <= 1

class SinusoidalEmbedding(nn.Module):
    """
    Standard sinusoidal embedding for scalar diffusion time t in [0,1].
    Output shape: (B, 2*n_freq)
    """
    def __init__(self, n_freq: int = 16):
        super().__init__()
        self.n_freq = n_freq

    def forward(self, t: pt.Tensor) -> pt.Tensor:
        # t: (B,)
        # frequencies: (n_freq,)
        freqs = (2.0 * math.pi) * (2.0 ** pt.arange(self.n_freq, device=t.device, dtype=t.dtype))
        t = t[:, None]  # (B,1)
        emb = pt.cat([pt.sin(freqs[None, :] * t), pt.cos(freqs[None, :] * t)], dim=1)
        return emb  # (B, 2*n_freq)

class RegressionNetwork( nn.Module ):

    def __init__(self, n_freq : int,
                       hidden_layers : List[int], 
                ) -> None:
        super().__init__()

        self.time_embedding = SinusoidalEmbedding( n_freq=n_freq )

        self.act = nn.GELU()

        # Hidden layers
        layers = []
        for n in range( len(hidden_layers) ):
            in_neurons = 4+2*n_freq if n == 0 else hidden_layers[n-1]
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
        s_embed = self.time_embedding( s )
        
        # Apply the network layer, film and activations
        x = pt.cat( (xA, xB, s_embed), dim=1 ) # shape (B, 4 + 4*n_freq)
        for layer_idx in range( len(self.layers) ):
            x = self.layers[layer_idx](x)
            x = self.act( x )

        # Apply the final output layer
        x = self.output_layer( x )
        return x