"""
EquivariantTransformer — refinement network that uses EquivariantTransformerLayer
as the shared refinement operator.

This is a drop-in subclass of chemdm.E3Transformer. The parent class handles:

    - TPMoleculeEmbedding for endpoint-conditioned scalar node init
    - ArcLengthEmbedding for arclength s
    - 0e / 1o / 1e / 2e initial feature assembly
    - The K-step refinement loop in refine_state(...) with the
      4 * s * (1-s) endpoint-preservation envelope
    - The forward signature (xA, xB, s) -> (Molecule, list[E3State])

We only swap one thing: the refinement layer. The parent builds an
E3AttentionLayer (GAT-style attention); we replace it with an
EquivariantTransformerLayer (real SE(3)-Transformer Q/K/V attention,
with K and V conditioned on the spherical harmonics of the edge direction).

Because EquivariantTransformerLayer subclasses E3AttentionLayer, the
forward signature is identical (xA, xB, s, state) -> (f_new, dx), so
refine_state in the parent doesn't need to know anything has changed.
"""

from __future__ import annotations

from typing import Optional

from chemdm.E3Transformer import E3Transformer

from EquivariantTransformerLayer import EquivariantTransformerLayer


class EquivariantTransformer(E3Transformer):
    """
    Refinement network with SE(3)-Transformer attention.

    Accepts every kwarg of E3Transformer, plus irreps_qk_str and
    irreps_v_str for the transformer layer.

    Parameters
    ----------
    irreps_node_str, n_refinement_steps, d_cutoff, n_freq, n_rbf,
    tp_embedding_dim, tp_embedding_hidden_dim, tp_embedding_hidden_layers,
    initial_edge_feature_dim:
        Forwarded to E3Transformer unchanged.

    irreps_qk_str:
        Irreps for both Q and K in the transformer layer. Every (l, p) in
        irreps_qk_str must also appear in irreps_node_str (q_proj is an
        o3.Linear and preserves irrep types).

    irreps_v_str:
        Irreps for V in the transformer layer. Defaults to irreps_node_str
        so V can carry the same equivariant information as the node features.
    """

    def __init__(
        self,
        irreps_node_str: str = "64x0e + 16x1o + 8x1e",
        n_refinement_steps: int = 7,
        d_cutoff: float = 5.0,
        n_freq: int = 8,
        n_rbf: int = 10,
        tp_embedding_dim: int = 64,
        tp_embedding_hidden_dim: int = 128,
        tp_embedding_hidden_layers: int = 2,
        initial_edge_feature_dim: int = 0,
        irreps_qk_str: str = "16x0e + 8x1o + 4x1e",
        irreps_v_str: Optional[str] = None,
    ) -> None:
        # Build the parent. It constructs everything we need (embeddings,
        # arclength encoder, 0e/1o lifts, irreps blocks for 1e/2e) and also
        # creates an E3AttentionLayer at self.refinement_layer that we
        # overwrite immediately below.
        super().__init__(
            irreps_node_str=irreps_node_str,
            n_refinement_steps=n_refinement_steps,
            d_cutoff=d_cutoff,
            n_freq=n_freq,
            n_rbf=n_rbf,
            tp_embedding_dim=tp_embedding_dim,
            tp_embedding_hidden_dim=tp_embedding_hidden_dim,
            tp_embedding_hidden_layers=tp_embedding_hidden_layers,
            initial_edge_feature_dim=initial_edge_feature_dim,
        )

        # nn.Module.__setattr__ replaces the old binding cleanly: the
        # parent's E3AttentionLayer is dropped from self._modules and
        # garbage-collected, so its parameters do not show up in
        # self.parameters() afterwards.
        self.refinement_layer = EquivariantTransformerLayer(
            irreps_node_str=irreps_node_str,
            irreps_qk_str=irreps_qk_str,
            irreps_v_str=irreps_v_str,
            d_cutoff=d_cutoff,
            n_rbf=n_rbf,
        )
