"""
EquivariantTransformerLayer — SE(3)-Transformer-style attention with fully
equivariant Q, K, V.

This is an experimental drop-in alternative to chemdm.E3AttentionLayer's
attention mechanism. E3AttentionLayer uses GAT-style attention (a scalar MLP
score over edge scalar features and node 0e features). This layer uses real
transformer-style Q/K/V attention where K and V are tensor products of the
source node features with the spherical harmonics Y(r_ij) of the edge
direction — so the score and the value both see the relative geometry of the
edge, not just scalar features of the endpoints.

Design (single head):

    Q_i      = q_proj(f_i)                              # node-wise, equivariant
    K_ij     = k_tp(f_src=f_j, Y(r_ij), w_K_ij)         # geometry-conditioned
    V_ij     = v_tp(f_src=f_j, Y(r_ij), w_V_ij)         # geometry-conditioned
    score_ij = <Q_i, K_ij> / sqrt(d_qk)                 # invariant scalar
    alpha_ij = softmax_j(score_ij)                      # over incoming edges
    agg_j    = Σ_i alpha_ij · V_ij                      # equivariant
    out_j    = out_proj(agg_j)                          # mix back to irreps_node

Equivariance:
    - Q ∈ irreps_qk and K_ij ∈ irreps_qk are equivariant under SO(3).
    - For two tensors in the same irreps, Σ_m Q_m K_m is the SO(3)-invariant
      contraction: vector dot for l=1, Frobenius inner product for l=2, etc.
      Each l-block contributes an invariant; summing across blocks keeps
      invariance. So `score` is a scalar invariant per edge.
    - alpha = softmax(invariant scores) is invariant.
    - V_ij is equivariant.
    - agg = Σ alpha · V is invariant scalar x equivariant tensor =
      equivariant.

Subclassing E3AttentionLayer keeps the diff localised: edge feature
construction, the gated equivariant nonlinearity in _update_features, and
the coordinate update in _coordinate_update are inherited unchanged. Only
__init__ and _aggregate_messages differ.

Multi-head note (future work):
    Use irreps_qk and irreps_v with multiplicity n_heads, run softmax per
    head, then concatenate head outputs before out_proj. The catch is that
    e3nn's memory layout interleaves blocks by l, not by head, so a per-head
    reshape requires either separate Linear/TP modules per head or a custom
    block-aware reshape. Single-head is implemented here for clarity.
"""

from __future__ import annotations

import math

import torch as pt
import torch.nn as nn

from e3nn import o3

from chemdm.MLP import MultiLayerPerceptron
from chemdm.E3AttentionLayer import (
    E3AttentionLayer,
    EdgeData,
    segment_softmax,
)


class EquivariantTransformerLayer(E3AttentionLayer):
    """
    SE(3)-Transformer attention layer.

    Drop-in replacement for E3AttentionLayer: same forward signature, same
    E3State / EdgeData dataclasses, same coordinate update. The change is in
    _aggregate_messages (now Q/K/V attention) and __init__ (sets up q_proj,
    k_tp, v_tp, k_radial, v_radial, out_proj; discards the GAT-style
    modules that the parent class set up).

    Parameters
    ----------
    irreps_node_str:
        Irreps of the per-node feature tensor f. Same meaning as in
        E3AttentionLayer.
    irreps_qk_str:
        Irreps for both Q and K. They must share an irreps layout so the
        elementwise product is a well-defined SO(3)-invariant contraction.
        For Q to be representable, every (l, p) in irreps_qk must also
        appear in irreps_node (because q_proj is an o3.Linear and l/p are
        preserved across the projection).
    irreps_v_str:
        Irreps for V. Defaults to irreps_node so V can carry the same
        equivariant information as the input features. The output projection
        at the end maps irreps_v back to irreps_node.
    d_cutoff, n_rbf:
        Neighbor-search cutoff and RBF count for the edge features. Same as
        parent.
    feature_residual_scale, self_interaction_init_scale:
        Forwarded to the parent class.
    """

    def __init__(
        self,
        irreps_node_str: str,
        irreps_qk_str: str = "16x0e + 8x1o + 4x1e",
        irreps_v_str: str | None = None,
        d_cutoff: float = 5.0,
        n_rbf: int = 10,
        feature_residual_scale: float = 0.2,
        self_interaction_init_scale: float = 0.0,
    ) -> None:
        # Build the parent. This sets up the shared infrastructure we keep:
        # irreps_node, irreps_0e, irreps_sh, lmax, rbf, n_edge_scalar,
        # radial_context_dim, scalar_readout, self_interaction, the gated
        # nonlinearity (pre_gate + gate), and the coordinate update modules
        # (coord_head, coordinate_gate_network, edge_coordinate_network).
        #
        # It also builds the parent's GAT-style attention modules (tp,
        # radial_network, edge_message_scalar_gate, edge_attention_score),
        # which we drop immediately below so they don't appear in the
        # optimizer's parameter list as ghost parameters.
        super().__init__(
            irreps_node_str=irreps_node_str,
            d_cutoff=d_cutoff,
            n_rbf=n_rbf,
            self_interaction_init_scale=self_interaction_init_scale,
            feature_residual_scale=feature_residual_scale,
        )

        del self.tp
        del self.radial_network
        del self.edge_attention_score
        del self.edge_message_scalar_gate

        self.irreps_qk = o3.Irreps(irreps_qk_str)
        self.irreps_v = o3.Irreps(irreps_v_str) if irreps_v_str is not None else self.irreps_node

        # Q must be representable as an o3.Linear of irreps_node, so every
        # (l, p) in irreps_qk must appear in irreps_node. Same for V via
        # the TP, although TPs can mix irreps so this is less strict.
        node_types = {(ir.l, ir.p) for _, ir in self.irreps_node}
        qk_types = {(ir.l, ir.p) for _, ir in self.irreps_qk}
        missing = qk_types - node_types
        assert not missing, (
            f"irreps_qk has (l, p) types {missing} that are not present in "
            f"irreps_node {irreps_node_str}; q_proj would be identically zero "
            f"on those blocks."
        )

        # Q: node-wise equivariant projection. No geometry dependence — Q
        # encodes "what kind of neighbor does this atom want to attend to."
        self.q_proj = o3.Linear(self.irreps_node, self.irreps_qk)

        # K and V are geometry-conditioned tensor products of f_src and the
        # spherical harmonics of the edge direction. Per-edge weights come
        # from radial networks of the same scalar context the parent uses.
        self.k_tp = o3.FullyConnectedTensorProduct(
            self.irreps_node,
            self.irreps_sh,
            self.irreps_qk,
            shared_weights=False,
        )
        self.v_tp = o3.FullyConnectedTensorProduct(
            self.irreps_node,
            self.irreps_sh,
            self.irreps_v,
            shared_weights=False,
        )

        self.k_radial = MultiLayerPerceptron(
            [self.radial_context_dim, 256, 256, self.k_tp.weight_numel],
            nn.GELU,
            "equivariant_transformer_k_radial",
        )
        self.v_radial = MultiLayerPerceptron(
            [self.radial_context_dim, 256, 256, self.v_tp.weight_numel],
            nn.GELU,
            "equivariant_transformer_v_radial",
        )

        # Mix the aggregated V (in irreps_v) back into irreps_node so the
        # parent's _update_features residual block sees an irreps_node
        # update. If irreps_v == irreps_node this is a square equivariant
        # linear — still useful (and load-bearing if you later add multi-
        # head, where this is where the heads get mixed).
        self.out_proj = o3.Linear(self.irreps_v, self.irreps_node)

        # 1 / sqrt(d_qk) attention scaling. Standard transformer choice. The
        # approximation here is that we treat all irreps_qk components as
        # roughly IID; in reality each l-block has its own scale, but the
        # learnable radial networks absorb the discrepancy.
        self.score_scale = 1.0 / math.sqrt(self.irreps_qk.dim)

    def _aggregate_messages(self, f: pt.Tensor, edges: EdgeData) -> pt.Tensor:
        """
        SE(3)-Transformer attention.

        For each directed edge src -> dst:
            Q_dst    = q_proj(f_dst)                       # equivariant
            K_ij     = k_tp(f_src, Y(r_ij), w_K_ij)        # equivariant
            V_ij     = v_tp(f_src, Y(r_ij), w_V_ij)        # equivariant
            score_ij = <Q_dst, K_ij> / sqrt(d_qk)          # invariant
            alpha_ij = softmax_j(score_ij)                 # per dst
            agg_dst  = Σ_src alpha_ij · V_ij               # equivariant

        Returns
        -------
        agg : (N, irreps_node.dim)
            Aggregated attention output, ready for the parent's residual
            feature update.
        """
        # 1. Spherical harmonics of the edge direction. These carry the
        #    rotational structure of r_ij into the K/V tensor products.
        edge_sh = o3.spherical_harmonics(
            self.irreps_sh,
            edges.edge_dir,
            normalize=True,
            normalization="component",
        )

        # 2. Scalar edge context for the K/V radial networks. Same context
        #    the parent class used for its GAT score: edge scalars (bond
        #    flags, distances, RBFs, reaction-status) plus source and
        #    destination scalar node features. Conditioning the radial
        #    networks on this lets K_ij and V_ij specialize for, e.g.,
        #    bonded-vs-forming-vs-breaking edges.
        node_scalars = self.scalar_readout( f )
        radial_context = self._edge_context( f, edges, scalar_features=node_scalars )

        # 3. Q lives on the destination node. Computed for all nodes once,
        #    then indexed by edges.dst.
        Q_all = self.q_proj( f )             # (N, irreps_qk.dim)
        Q_dst = Q_all[ edges.dst ]           # (E, irreps_qk.dim)

        # 4. K and V are per-edge tensor products of f_src with Y(r_ij).
        k_weights = self.k_radial( radial_context )
        v_weights = self.v_radial( radial_context )
        K_ij = self.k_tp( f[edges.src], edge_sh, k_weights )   # (E, irreps_qk.dim)
        V_ij = self.v_tp( f[edges.src], edge_sh, v_weights )   # (E, irreps_v.dim)

        # 5. Equivariant inner product <Q, K> = Σ_m Q_m K_m.
        #
        #    For each l-block of irreps_qk:
        #        l=0:  Σ Q_c K_c                  (scalar product)
        #        l=1:  Σ Q_c · K_c                (3D dot product per channel)
        #        l=2:  Σ tr(Q_c K_c)              (Frobenius inner product)
        #        ...
        #    All SO(3)-invariant. Summing across blocks preserves invariance,
        #    so `score` is a scalar per edge.
        score = (Q_dst * K_ij).sum( dim=-1, keepdim=True ) * self.score_scale  # (E, 1)

        # 6. Softmax over incoming edges per destination node.
        alpha = segment_softmax( score, edges.dst, n_segments=f.shape[0] )  # (E, 1)

        # 7. Weighted aggregation of V into destination nodes.
        weighted_V = alpha * V_ij          # (E, irreps_v.dim)
        agg_v = pt.zeros( f.shape[0], self.irreps_v.dim, dtype=f.dtype, device=f.device )
        agg_v.index_add_( 0, edges.dst, weighted_V )

        # 8. Mix V's irreps back into the node feature space for the residual
        #    update.
        return self.out_proj(agg_v)
