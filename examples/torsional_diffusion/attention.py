"""
EquivariantAttentionLayer — SE(3)-Transformer-style Q/K/V attention layer
for the torsional-diffusion score model trunk.

Same Q/K/V design as examples/transition1x_attention/EquivariantTransformerLayer:

    Q lives on the destination node and is geometry-free,
    K and V are tensor products of the source node features with the
    spherical harmonics of the edge direction, conditioned on a small
    scalar context built from distance RBFs and the source/destination
    scalar features.

Differences from the transition-path version:

    - no reactant/product endpoint conditioning
    - no reaction-status edge flags
    - no coordinate update; positions are the input data and are held
      fixed throughout the forward pass
    - simpler edge scalar context (distance, inverse distance, RBF)

The layer is stateless across molecules: it operates on a flat (N, .)
node tensor with a precomputed edge_index that already respects
molecule boundaries.
"""

from __future__ import annotations

import math

import torch as pt
import torch.nn as nn
import torch.nn.functional as F

from e3nn import o3
from e3nn import nn as e3nn_nn

from chemdm.MLP import MultiLayerPerceptron
from chemdm.DistanceRBFEmbedding import DistanceRBFEmbedding
from chemdm.E3AttentionLayer import segment_softmax


class EquivariantAttentionLayer(nn.Module):
    """
    SE(3)-Transformer attention layer for torsional diffusion.

    Parameters
    ----------
    irreps_node_str:
        Per-node feature irreps. Must contain at least one 0e block.
    irreps_qk_str:
        Irreps for Q and K. Every (l, p) here must also appear in
        irreps_node, since q_proj is an o3.Linear.
    irreps_v_str:
        Irreps for V. Defaults to irreps_node.
    d_cutoff, n_rbf:
        Neighbor-search cutoff and RBF count.
    feature_residual_scale:
        Scaling of the residual update added back to f.
    """

    def __init__(
        self,
        irreps_node_str: str = "64x0e + 32x0o + 16x1o + 16x1e",
        irreps_qk_str: str = "16x0e + 8x0o + 8x1o + 8x1e",
        irreps_v_str: str | None = None,
        d_cutoff: float = 5.0,
        n_rbf: int = 16,
        feature_residual_scale: float = 0.2,
    ) -> None:
        super().__init__()

        self.irreps_node = o3.Irreps(irreps_node_str)
        self.irreps_qk = o3.Irreps(irreps_qk_str)
        self.irreps_v = o3.Irreps(irreps_v_str) if irreps_v_str is not None else self.irreps_node

        self.d_cutoff = d_cutoff
        self.feature_residual_scale = feature_residual_scale

        self.lmax = max(ir.l for _, ir in self.irreps_node)
        self.irreps_sh = o3.Irreps.spherical_harmonics(self.lmax)

        # 0e block of node features — used for scalar edge context.
        self.irreps_0e = o3.Irreps( [(mul, ir) for mul, ir in self.irreps_node if ir.l == 0 and ir.p == 1] )
        assert self.irreps_0e.dim > 0, "Need at least one 0e block in irreps_node."

        # Edge scalar context: dist/cutoff, inv_dist, RBF.
        self.rbf = DistanceRBFEmbedding( 0.0, d_cutoff, n_rbf=n_rbf )
        self.eps = 0.02
        self.n_edge_scalar = 2 + self.rbf.out_dim
        self.radial_context_dim = self.n_edge_scalar + 2 * self.irreps_0e.dim

        # Q must be representable by an o3.Linear of irreps_node.
        node_types = {(ir.l, ir.p) for _, ir in self.irreps_node}
        qk_types = {(ir.l, ir.p) for _, ir in self.irreps_qk}
        missing = qk_types - node_types
        assert not missing, f"irreps_qk has (l, p) types {missing} not in irreps_node {irreps_node_str}."

        self.q_proj = o3.Linear( self.irreps_node, self.irreps_qk )

        # K and V can be any equivariant linea rmap
        self.k_tp = o3.FullyConnectedTensorProduct( self.irreps_node, self.irreps_sh, self.irreps_qk, shared_weights=False )
        self.v_tp = o3.FullyConnectedTensorProduct( self.irreps_node, self.irreps_sh, self.irreps_v, shared_weights=False )
        self.k_radial = MultiLayerPerceptron(
            [self.radial_context_dim, 128, 128, self.k_tp.weight_numel],
            nn.GELU,
            "td_attn_k_radial",
        )
        self.v_radial = MultiLayerPerceptron(
            [self.radial_context_dim, 128, 128, self.v_tp.weight_numel],
            nn.GELU,
            "td_attn_v_radial",
        )

        self.scalar_readout = o3.Linear( self.irreps_node, self.irreps_0e )
        self.out_proj = o3.Linear( self.irreps_v, self.irreps_node )

        self._setup_gate()

        self.score_scale = 1.0 / math.sqrt(self.irreps_qk.dim)

    def _setup_gate( self ) -> None:
        self.irreps_gate_scalars = o3.Irreps( [(mul, ir) for mul, ir in self.irreps_node if ir.l == 0 and ir.p == 1] )
        self.irreps_gate_gated = o3.Irreps( [(mul, ir) for mul, ir in self.irreps_node if not (ir.l == 0 and ir.p == 1)] )
        n_gates = sum( mul for mul, _ in self.irreps_gate_gated )
        self.irreps_gate_gates = o3.Irreps( f"{n_gates}x0e" )

        # Construct Gate first, then project pre_gate into Gate's *own* irreps_in.
        # Gate internally sorts (irreps_scalars + irreps_gates + irreps_gated) by
        # (l, p) ascending, which permutes blocks once 0o enters the layout
        # (0o sorts before 0e at the same l). Projecting to a manually built
        # `irreps_scalars + irreps_gates + irreps_gated` would then mis-align the
        # data with what Gate's internal Extract expects, leaking 0e content into
        # the 0o gated channel and silently breaking parity-equivariance.
        self.gate = e3nn_nn.Gate(
            self.irreps_gate_scalars, [F.silu],
            self.irreps_gate_gates, [pt.sigmoid],
            self.irreps_gate_gated,
        )
        self.pre_gate = o3.Linear(self.irreps_node, self.gate.irreps_in)

    def forward( self, f: pt.Tensor, x: pt.Tensor, edge_index: pt.Tensor ) -> pt.Tensor:
        """
        Parameters
        ----------
        f:           (N, irreps_node.dim) — node equivariant features
        x:           (N, 3)               — atom positions (held fixed)
        edge_index:  (2, E)               — directed pairs [src, dst]; assumed
                                            to already respect molecule
                                            boundaries when batching.

        Returns
        -------
        f_new : (N, irreps_node.dim)
        """
        src, dst = edge_index[0], edge_index[1]

        edge_vec = x[dst] - x[src]
        dist_raw = pt.linalg.norm( edge_vec, dim=-1, keepdim=True ).clamp_min(1.0e-8)
        edge_dir = edge_vec / dist_raw

        dist = dist_raw / self.d_cutoff
        inv_dist = self.eps / pt.sqrt(dist**2 + self.eps**2)
        rbf = self.rbf(dist_raw)
        edge_scalars = pt.cat( [dist, inv_dist, rbf], dim=-1 )

        node_scalars = self.scalar_readout(f)
        radial_context = pt.cat( [edge_scalars, node_scalars[src], node_scalars[dst]], dim=-1 )

        edge_sh = o3.spherical_harmonics( self.irreps_sh, edge_dir, normalize=True, normalization="component" )

        Q_all = self.q_proj( f )
        Q_dst = Q_all[dst]

        k_w = self.k_radial( radial_context )
        v_w = self.v_radial( radial_context )
        K_ij = self.k_tp( f[src], edge_sh, k_w )
        V_ij = self.v_tp( f[src], edge_sh, v_w )

        # SO(3)-invariant inner product per edge, scaled by 1/sqrt(d_qk).
        score = (Q_dst * K_ij).sum(dim=-1, keepdim=True) * self.score_scale
        alpha = segment_softmax( score, dst, n_segments=f.shape[0] )

        weighted_V = alpha * V_ij
        agg_v = pt.zeros( f.shape[0], self.irreps_v.dim, dtype=f.dtype, device=f.device )
        agg_v.index_add_( 0, dst, weighted_V )
        agg = self.out_proj( agg_v )

        f_update = self.gate( self.pre_gate(agg) )
        return f + self.feature_residual_scale * f_update
