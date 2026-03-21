import math
import torch as pt
import torch.nn as nn

class SinusoidalEmbedding(nn.Module):
    """
    Standard sinusoidal embedding for scalar diffusion time t in [0,1].
    Output shape: (B, 2*n_freq)
    """
    def __init__(self, n_freq: int = 16):
        super().__init__()
        self.n_freq = n_freq
        self.n_embeddings = 2 * self.n_freq

    def forward(self, t: pt.Tensor) -> pt.Tensor:
        # t: (B,)
        # frequencies: (n_freq,)
        freqs = (2.0 * math.pi) * (2.0 ** pt.arange(self.n_freq, device=t.device, dtype=t.dtype))
        t = t[:, None]  # (B,1)
        emb = pt.cat([ pt.sin(freqs[None, :] * t), pt.cos(freqs[None, :] * t)], dim=1 )
        return emb  # (B, 1+2*n_freq)
    
class ArcLengthEmbedding(nn.Module):
    """
    Standard sinusoidal embedding for the normalized arclength s in [0,1],
    but with `s` and `1-s` concatenated so the network can figure out end points.
    Output shape: (B, 2*n_freq+2)
    """
    def __init__(self, n_freq: int = 16):
        super().__init__()
        self.n_freq = n_freq
        self.n_embeddings = 2 * self.n_freq + 2 # last two for `s` and `1-s`

    def getNumberOfFeatures( self ) -> int:
        return self.n_embeddings

    def forward(self, s: pt.Tensor) -> pt.Tensor:
        # s: (B,)
        # frequencies: (n_freq,)
        freqs = 2.0 * math.pi * pt.arange(1, self.n_freq + 1, device=s.device, dtype=s.dtype)
        s = pt.atleast_1d( s ) # to handle scalar ege cases
        s = s[:, None]  # (B,1)
        emb = pt.cat([s, 1.0-s, pt.sin(freqs[None, :] * s), pt.cos(freqs[None, :] * s)], dim=1)
        return emb  # (B, 2*n_freq)
