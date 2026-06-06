"""
TorsionalScoreNetwork — the full score model for torsional diffusion.

Composes:
    1. Atom embedding: Z -> 0e scalar features.
    2. Sinusoidal embedding of log(σ(t)) -> 0e scalar features added per atom.
    3. Lift 0e scalars to the trunk irreps (higher-l blocks initialise to 0;
       they will be populated by the attention layers via TPs with Y(r̂_ij)).
    4. K layers of EquivariantAttentionLayer (SE(3)-Transformer Q/K/V attention).
    5. PseudotorqueHead: per-rotatable-bond pseudoscalar (0o) readout.

The output is `δτ ∈ ℝ^m`, one real number per rotatable bond. By construction
it is SE(3)-invariant and parity-equivariant (δτ(−x) = −δτ(x)), which are the
required symmetries for the torsional score on the hypertorus.

Batching:
    The network takes flat per-atom and per-bond tensors plus `atom_batch` /
    `bond_batch` index tensors (PyG-style). It expects `edge_index` to already
    respect molecule boundaries — no cross-molecule edges. The caller is
    responsible for building the trunk neighborhood (e.g. via a radius graph
    inside the cutoff) and the rotatable-bond list.
"""

from __future__ import annotations

import math

import torch as pt
import torch.nn as nn

from e3nn import o3

from chemdm.MLP import MultiLayerPerceptron

from attention import EquivariantAttentionLayer
from pseudotorque import PseudotorqueHead


class TorsionalScoreNetwork(nn.Module):
    """
    Parameters
    ----------
    irreps_node_str:
        Trunk node-feature irreps. Default carries 0e + 0o + 1o + 1e so that
        the pseudotorque head can synthesise 0o through low-order TP paths.
    irreps_qk_str:
        Q and K irreps for the attention layers.
    irreps_v_str:
        V irreps. Defaults to irreps_node.
    n_layers:
        Number of trunk attention layers.
    d_cutoff:
        Trunk neighbor-search cutoff [Å] (used only by the trunk's RBFs).
    head_cutoff:
        Per-bond head neighborhood cutoff [Å].
    n_rbf:
        RBF count for the trunk and the head.
    n_pseudo:
        Per-atom pseudoscalar bank width in the head.
    time_dim:
        Width of the sinusoidal σ-embedding.
    z_embed_dim:
        Width of the atomic-number embedding used by the head's query MLP.
        Independent of the trunk's atom embedding.
    n_elements:
        Size of the atomic-number embedding tables.
    sigma_min, sigma_max:
        Noise schedule bounds [rad]. Stored on the module for the trainer to
        reuse; not used internally.
    """

    def __init__(
        self,
        irreps_node_str: str = "64x0e + 32x0o + 16x1o + 16x1e",
        irreps_qk_str: str = "16x0e + 8x0o + 8x1o + 8x1e",
        irreps_v_str: str | None = None,
        n_layers: int = 4,
        d_cutoff: float = 5.0,
        head_cutoff: float = 5.0,
        n_rbf: int = 16,
        n_pseudo: int = 8,
        time_dim: int = 32,
        z_embed_dim: int = 16,
        n_elements: int = 120,
        sigma_min: float = 0.01 * math.pi,
        sigma_max: float = math.pi,
    ) -> None:
        super().__init__()

        self.irreps_node = o3.Irreps(irreps_node_str)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.time_dim = time_dim

        # 0e block size — where the atom and time embeddings live before lift.
        self.irreps_0e = o3.Irreps(
            [(mul, ir) for mul, ir in self.irreps_node if ir.l == 0 and ir.p == 1]
        )
        self.scalar_dim = self.irreps_0e.dim

        # Atom embedding: Z -> 0e features of width scalar_dim.
        self.z_embed = nn.Embedding(n_elements, self.scalar_dim)

        # σ-embedding MLP. The raw sinusoidal embedding is (B, time_dim); we
        # project it to (B, scalar_dim) for adding to atom 0e features, and
        # also pass the raw (B, time_dim) embedding to the head's query MLP.
        self.time_mlp = MultiLayerPerceptron(
            [time_dim, 2 * time_dim, self.scalar_dim],
            nn.GELU,
            "td_time_mlp",
        )

        # Lift 0e features to irreps_node. Equivariant o3.Linear: only 0e -> 0e
        # paths have weights, so 0o / 1o / 1e blocks start at exactly zero and
        # are populated by the trunk attention layers below.
        self.lift = o3.Linear(self.irreps_0e, self.irreps_node)

        # Trunk.
        self.layers = nn.ModuleList([
            EquivariantAttentionLayer(
                irreps_node_str=irreps_node_str,
                irreps_qk_str=irreps_qk_str,
                irreps_v_str=irreps_v_str,
                d_cutoff=d_cutoff,
                n_rbf=n_rbf,
            )
            for _ in range(n_layers)
        ])

        # Head.
        self.head = PseudotorqueHead(
            irreps_node_str=irreps_node_str,
            n_pseudo=n_pseudo,
            head_cutoff=head_cutoff,
            n_rbf=n_rbf,
            time_dim=time_dim,
            z_embed_dim=z_embed_dim,
            n_elements=n_elements,
        )

    def _sigma_embedding(self, sigma: pt.Tensor) -> pt.Tensor:
        """
        Sinusoidal embedding of log(σ). The noise schedule is geometric in σ,
        so log σ varies linearly with t and is the natural quantity to embed.

        Parameters
        ----------
        sigma : (B,)

        Returns
        -------
        (B, time_dim)
        """
        half = self.time_dim // 2
        device = sigma.device
        dtype = sigma.dtype
        log_sigma = pt.log(sigma).unsqueeze(-1)              # (B, 1)
        freqs = pt.exp(
            pt.linspace(0.0, math.log(1000.0), half, device=device, dtype=dtype)
        )                                                    # (half,)
        args = log_sigma * freqs                             # (B, half)
        return pt.cat([pt.sin(args), pt.cos(args)], dim=-1)  # (B, 2*half)

    def forward(
        self,
        Z: pt.Tensor,                # (N,)
        x: pt.Tensor,                # (N, 3)
        sigma: pt.Tensor,            # (B,)
        edge_index: pt.Tensor,       # (2, E)
        bonds: pt.Tensor,            # (m, 2)
        atom_batch: pt.Tensor,       # (N,)
        bond_batch: pt.Tensor,       # (m,)
    ) -> pt.Tensor:
        """
        Returns
        -------
        delta_tau : (m,) — predicted score per rotatable bond.
        """
        # Per-molecule sinusoidal σ-embedding, projected to 0e width.
        sigma_emb = self._sigma_embedding(sigma)             # (B, time_dim)
        time_scalar = self.time_mlp(sigma_emb)               # (B, scalar_dim)

        # Atom 0e features: Z embedding + per-atom σ scalar.
        f0e = self.z_embed(Z) + time_scalar[atom_batch]      # (N, scalar_dim)

        # Lift into the trunk irreps (higher-l blocks start at zero).
        f = self.lift(f0e)                                   # (N, irreps_node.dim)

        # Attention trunk.
        for layer in self.layers:
            f = layer(f, x, edge_index)

        # Pseudotorque head: per-rotatable-bond δτ_i, sign-flips under parity.
        return self.head(
            f=f,
            x=x,
            Z=Z,
            bonds=bonds,
            atom_batch=atom_batch,
            bond_batch=bond_batch,
            time_emb_per_mol=sigma_emb,
        )
