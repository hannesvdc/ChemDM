"""
PseudotorqueHead — per-rotatable-bond pseudoscalar (0o) readout for the
torsional-diffusion score model.

For each rotatable bond (b, c) we look at atoms a in the same molecule within
head_cutoff of the bond midpoint m_bc, and produce a single real number δτ_i
that is invariant under SE(3) and *flips sign* under parity x → -x.

The construction is a two-stage tensor product (conceptually "torque around the
bond axis built from atom-side messages"):

    Stage 1:
        radial_weight_a = MLP( RBF( |x_a - m_bc| ) )
        inter_a = TP( f_a, Y(r̂_{a→m_bc}), radial_weight_a )

    Stage 2:
        msg_a = TP( inter_a, Y(r̂_{b→c}) )

The output irreps of stage 2 are chosen as `n_pseudo x 0o + n_pseudo x 0e`:
    - the 0o block holds the pseudoscalar values (V in the attention)
    - the 0e block holds parity-even scalars used as attention keys

Attention then aggregates per bond:

    Q_bond = MLP_query( Z_b emb, Z_c emb, |x_c - x_b|, σ-embedding )   ∈ R^{n_pseudo}
    K_a, V_a = split(msg_a)                                            ∈ R^{n_pseudo}
    α_a = softmax_a( <Q_bond, K_a> / sqrt(n_pseudo) )
    v_bond = Σ_a α_a · V_a                                             ∈ R^{n_pseudo}, still 0o

Final readout: a linear map `n_pseudo → 1` with no bias and no nonlinearity
preserves the 0o nature of v_bond → δτ_i ∈ R.

Equivariance:
    SE(3) invariance follows because every operation is built from o3.Linear,
    o3.FullyConnectedTensorProduct, spherical harmonics of unit vectors, and
    attention with an SO(3)-invariant inner product.

    Parity equivariance (δτ(−x) = −δτ(x)) holds because:
      - Y_l → (-1)^l Y_l under x → -x, so Stage 1 + Stage 2 combined send
        the output's parity to (parity of input feature) × (-1)^{l_1} ×
        (-1)^{l_2}, and the path that lands in the 0o block has total
        parity -1 by construction.
      - The final linear map is bias-free and parity-preserving.
"""

from __future__ import annotations

import math

import torch as pt
import torch.nn as nn

from e3nn import o3

from chemdm.MLP import MultiLayerPerceptron
from chemdm.DistanceRBFEmbedding import DistanceRBFEmbedding
from chemdm.E3AttentionLayer import segment_softmax


class PseudotorqueHead(nn.Module):
    """
    Parameters
    ----------
    irreps_node_str:
        Irreps of the trunk node features that feed this head.
    n_pseudo:
        Width of the per-atom pseudoscalar bank. Also the size of the
        attention query.
    irreps_inter_str:
        Irreps of the Stage-1 intermediate. Should contain enough parity
        mixture to support paths to 0o in Stage 2. The default has both
        parities at l ∈ {0, 1, 2}.
    lmax:
        Maximum spherical-harmonic degree for both Y(r̂_{a→m}) and Y(r̂_{bc}).
    head_cutoff:
        Distance cutoff in Å between bond midpoint and atoms contributing
        to the head.
    n_rbf:
        Number of RBF kernels for the radial weights in Stage 1.
    time_dim:
        Width of the sinusoidal σ-embedding fed into the query MLP.
    z_embed_dim:
        Width of the per-element embedding used in the query MLP. (Independent
        of any embedding used by the trunk — this is for bond-level priors.)
    n_elements:
        Size of the atomic-number embedding table (covers Z up to n_elements−1).
    """

    def __init__(
        self,
        irreps_node_str: str = "64x0e + 32x0o + 16x1o + 16x1e",
        n_pseudo: int = 8,
        irreps_inter_str: str = "16x0e + 16x0o + 8x1o + 8x1e + 4x2e + 4x2o",
        lmax: int = 2,
        head_cutoff: float = 5.0,
        n_rbf: int = 16,
        time_dim: int = 32,
        z_embed_dim: int = 16,
        n_elements: int = 120,
    ) -> None:
        super().__init__()

        self.irreps_node = o3.Irreps(irreps_node_str)
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax)
        self.irreps_inter = o3.Irreps(irreps_inter_str)

        self.head_cutoff = head_cutoff
        self.n_pseudo = n_pseudo

        self.rbf = DistanceRBFEmbedding(0.0, head_cutoff, n_rbf=n_rbf)

        # Stage 1: f_a ⊗ Y(r̂_{a→m}) with per-pair radial weights.
        self.tp1 = o3.FullyConnectedTensorProduct(
            self.irreps_node, self.irreps_sh, self.irreps_inter,
            shared_weights=False,
        )
        self.radial1 = MultiLayerPerceptron(
            [self.rbf.out_dim, 128, 128, self.tp1.weight_numel],
            nn.GELU,
            "pseudotorque_radial",
        )

        # Stage 2: inter ⊗ Y(r̂_{bc}). Output layout MATTERS: we depend on
        # `n_pseudo x 0o` coming first, then `n_pseudo x 0e`, when we split
        # along the feature dim below.
        self.irreps_out = o3.Irreps(f"{n_pseudo}x0o + {n_pseudo}x0e")
        self.tp2 = o3.FullyConnectedTensorProduct(
            self.irreps_inter, self.irreps_sh, self.irreps_out,
            shared_weights=True,
        )

        # Bond-level query (a parity-even scalar in R^{n_pseudo}) used to
        # attention-weight the contributions of nearby atoms.
        self.z_embed = nn.Embedding(n_elements, z_embed_dim)
        self.query_mlp = MultiLayerPerceptron(
            [2 * z_embed_dim + 1 + time_dim, 128, 128, n_pseudo],
            nn.GELU,
            "pseudotorque_query",
        )

        # Final readout: n_pseudo → 1, no bias, no nonlinearity (so 0o
        # input maps to 0o output).
        self.final = nn.Linear(n_pseudo, 1, bias=False)

        self.score_scale = 1.0 / math.sqrt(n_pseudo)

    def forward(
        self,
        f: pt.Tensor,                # (N, irreps_node.dim)
        x: pt.Tensor,                # (N, 3)
        Z: pt.Tensor,                # (N,)
        bonds: pt.Tensor,            # (m, 2)  -- global (b, c) atom indices
        atom_batch: pt.Tensor,       # (N,)    -- molecule index per atom
        bond_batch: pt.Tensor,       # (m,)    -- molecule index per bond
        time_emb_per_mol: pt.Tensor, # (B, time_dim)
    ) -> pt.Tensor:
        """
        Returns
        -------
        delta_tau : (m,) — pseudoscalar score per rotatable bond.
        """
        device = x.device
        dtype = x.dtype

        b_idx = bonds[:, 0]
        c_idx = bonds[:, 1]

        bond_vec = x[c_idx] - x[b_idx]
        bond_len = pt.linalg.norm(bond_vec, dim=-1, keepdim=True).clamp_min(1.0e-8)
        bond_dir = bond_vec / bond_len
        midpoints = 0.5 * (x[b_idx] + x[c_idx])              # (m, 3)

        # Per-bond neighborhood: atoms in the same molecule within head_cutoff
        # of the bond midpoint. O(m * N) per batch — fine for drug-like sizes;
        # can be replaced with a proper radius graph if it ever becomes a
        # bottleneck.
        dist_ma = pt.cdist(midpoints, x)                     # (m, N)
        same_mol = bond_batch[:, None] == atom_batch[None, :]
        mask = (dist_ma < self.head_cutoff) & same_mol
        bond_idx, atom_idx = mask.nonzero(as_tuple=True)     # (P,), (P,)

        # Per-pair geometry.
        r_am = midpoints[bond_idx] - x[atom_idx]             # (P, 3)
        d_am = pt.linalg.norm(r_am, dim=-1, keepdim=True).clamp_min(1.0e-8)
        rhat_am = r_am / d_am

        # Spherical harmonics.
        Y_am = o3.spherical_harmonics(
            self.irreps_sh, rhat_am, normalize=True, normalization="component"
        )                                                    # (P, irreps_sh.dim)

        Y_bc_per_bond = o3.spherical_harmonics(
            self.irreps_sh, bond_dir, normalize=True, normalization="component"
        )                                                    # (m, irreps_sh.dim)
        Y_bc = Y_bc_per_bond[bond_idx]                       # (P, irreps_sh.dim)

        # Stage 1: f_a ⊗ Y(r̂_{a→m}) with per-pair radial weights.
        rbf = self.rbf(d_am)                                 # (P, n_rbf)
        w1 = self.radial1(rbf)                               # (P, tp1.weight_numel)
        inter = self.tp1(f[atom_idx], Y_am, w1)              # (P, irreps_inter.dim)

        # Stage 2: inter ⊗ Y(r̂_{bc}).
        msg = self.tp2(inter, Y_bc)                          # (P, irreps_out.dim)

        # Split: irreps_out is layed out as `n_pseudo x 0o + n_pseudo x 0e`,
        # so the first n_pseudo columns are 0o (values), the next n_pseudo are
        # 0e (attention keys).
        V = msg[:, : self.n_pseudo]                          # (P, n_pseudo), 0o
        K = msg[:, self.n_pseudo : 2 * self.n_pseudo]        # (P, n_pseudo), 0e

        # Bond-level query in R^{n_pseudo} (a 0e scalar bank).
        Z_b = self.z_embed(Z[b_idx])                         # (m, z_embed_dim)
        Z_c = self.z_embed(Z[c_idx])                         # (m, z_embed_dim)
        time_per_bond = time_emb_per_mol[bond_batch]         # (m, time_dim)
        query_in = pt.cat([Z_b, Z_c, bond_len, time_per_bond], dim=-1)
        Q = self.query_mlp(query_in)                         # (m, n_pseudo)
        Q_per_pair = Q[bond_idx]                             # (P, n_pseudo)

        # Attention.
        score = (Q_per_pair * K).sum(dim=-1, keepdim=True) * self.score_scale  # (P, 1)
        alpha = segment_softmax(score, bond_idx, n_segments=bonds.shape[0])    # (P, 1)

        weighted_V = alpha * V                                # (P, n_pseudo)
        agg = pt.zeros(bonds.shape[0], self.n_pseudo, dtype=dtype, device=device)
        agg.index_add_(0, bond_idx, weighted_V)               # (m, n_pseudo)

        # Final readout: linear, no bias, no nonlinearity → preserves 0o.
        return self.final(agg).squeeze(-1)                    # (m,)
