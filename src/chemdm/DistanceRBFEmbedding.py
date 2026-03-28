import torch as pt
import torch.nn as nn

class DistanceRBFEmbedding(nn.Module):
    def __init__(
        self,
        d_min: float = 0.0,
        d_max: float = 5.0,
        n_rbf: int = 10,
        gamma: float | None = None,
    ):
        """
        Radial basis function embedding for scalar distances.

        Parameters
        ----------
        d_min : float
            Smallest RBF center.
        d_max : float
            Largest RBF center.
        n_rbf : int
            Number of Gaussian RBFs.
        gamma : float | None
            Width parameter in exp(-gamma * (d - mu)^2).
            If None, it is chosen automatically from the center spacing.
        """
        super().__init__()

        assert n_rbf >= 2, "`n_rbf` must be at least 2."
        assert d_max > d_min, "`d_max` must be strictly larger than `d_min`."

        mu = pt.linspace(d_min, d_max, n_rbf)   # inclusive endpoints
        delta = float(mu[1] - mu[0])

        if gamma is None:
            gamma = 1.0 / (delta * delta)

        self.register_buffer("mu", mu)
        self.gamma = gamma

    @property
    def out_dim(self) -> int:
        return pt.numel( self.mu )

    def forward(self, d: pt.Tensor) -> pt.Tensor:
        """
        Expects distances with trailing dimension 1, e.g.
            (E, 1) or (B, E, 1)

        Returns
            (E, n_rbf) or (B, E, n_rbf)
        """
        assert d.shape[-1] == 1, f"Expected trailing dimension 1 for distances, got shape {tuple(d.shape)}"

        d = d.squeeze(-1)
        return pt.exp(-self.gamma * (d[..., None] - self.mu) ** 2)