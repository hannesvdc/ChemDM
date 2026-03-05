import math
import torch as pt
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from typing import List, Tuple

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

class FiLMNetwork( nn.Module ):

    def __init__( self, input_dim : int,
                        hidden_layers : List[int],
                        target_hidden_layers : List[int],
                        act : nn.Module = nn.GELU() ):
        super().__init__()

        layers = []
        for n in range( len(hidden_layers) ):
            in_neurons = input_dim if n == 0 else hidden_layers[n-1]
            out_neurons = hidden_layers[n]
            layers.append( ( f"Linear_{n+1}", nn.Linear(in_neurons, out_neurons, bias=True) ) )
            layers.append( ( f"Act_{n+1}", act.__class__() ) )
        self.backbone = nn.Sequential( OrderedDict( layers ) )
        
        # Heads: one (gamma,beta) head per target layer
        self.target_hidden_layers = target_hidden_layers
        last = hidden_layers[-1]

        self.heads = nn.ModuleList([ nn.Linear(last, 2 * h, bias=True) for h in target_hidden_layers ])

    def forward(self, c : pt.Tensor, # (B,ndim) 
                    ) -> Tuple[List[pt.Tensor], List[pt.Tensor]]:

        # Apply the hidden layers
        h = self.backbone( c )

        gammas: List[pt.Tensor] = []
        betas: List[pt.Tensor] = []
        for head, width in zip(self.heads, self.target_hidden_layers):
            out = head(h)                        # (B, 2*width)
            gamma = out[:, :width]
            beta  = out[:, width:]
            gammas.append(gamma)
            betas.append(beta)

        return gammas, betas

class ScoreNetwork( nn.Module ):

    def __init__(self, n_freq : int,
                       hidden_layers : List[int],
                       film_hidden_layers : List[int] ) -> None:
        super().__init__()

        self.time_embedding = SinusoidalEmbedding( n_freq=n_freq )

        self.act = nn.GELU()
        self.film = FiLMNetwork( 4 + 2*n_freq + 2*n_freq, film_hidden_layers, hidden_layers, self.act )

        # Hidden layers
        layers = []
        for n in range( len(hidden_layers) ):
            in_neurons = 2 if n == 0 else hidden_layers[n-1]
            out_neurons = hidden_layers[n]
            layers.append( nn.Linear(in_neurons, out_neurons, bias=True) )
        self.layers = nn.ModuleList( layers )

        # Output layer
        self.output_layer = nn.Linear( hidden_layers[-1], 2, bias=True )

    def forward(self, u : pt.Tensor, # (B,2)
                      t : pt.Tensor, # (B,)
                      xA : pt.Tensor, # (B, 2)
                      xB : pt.Tensor, #(B,2)
                      s : pt.Tensor, # (B,)
                    ) -> pt.Tensor:
        if t.ndim > 1:
            t = t.flatten()
        if s.ndim > 1:
            s = s.flatten()
        assert u.ndim == 2 and u.shape[1] == 2, f"`u` must have shape (B,2) but got {u.shape}."
        assert xA.ndim == 2 and xA.shape[1] == 2, f"`xA` must have shape (B,2) but got {xA.shape}."
        assert xB.ndim == 2 and xB.shape[1] == 2, f"`xB` must have shape (B,2) but got {xB.shape}."
        assert _all_equal([ u.shape[0], xA.shape[0], xB.shape[0], s.shape[0], t.shape[0]]), \
            f"`u`, `t`, `xA`, `xB` and `s` must have the same leading (batch) dimension."

        # Time embedding
        t_embed = self.time_embedding( t )
        s_embed = self.time_embedding( s )
        
        # Apply the network layer, film and activations
        film_input = pt.cat( (xA, xB, s_embed, t_embed), dim=1 ) # shape (B, 4 + 4*n_freq)
        gammas, betas = self.film( film_input )
        for layer_idx in range( len(self.layers) ):
            gam = gammas[layer_idx]
            bet = betas[layer_idx]

            u = self.layers[layer_idx](u)
            u = (1.0 + gam) * u + bet
            u = F.gelu( u )

        # Apply the final output layer
        u = self.output_layer( u )
        return u