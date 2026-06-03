from __future__ import annotations

from dataclasses import dataclass

import torch as pt
import torch.nn as nn
import torch.nn.functional as F

from e3nn import o3
from e3nn import nn as e3nn_nn

from chemdm.MLP import MultiLayerPerceptron
from chemdm.DistanceRBFEmbedding import DistanceRBFEmbedding
from chemdm.MoleculeGraph import Molecule, findAllNeighborsReactantProduct


@dataclass
class E3State:
    """
    State propagated through the e3nn transition-path network.

    f : node features in irreps representation, shape (N, irreps_node.dim)
    x : coordinates, shape (N, 3)
    """

    f: pt.Tensor
    x: pt.Tensor


@dataclass
class EdgeData:
    """
    Edge-local data used by the layer.

    src, dst:
        Edge indices, shape (E,)

    edge_dir:
        Unit edge direction x_dst - x_src, shape (E, 3)

    edge_features:
        Scalar edge features, shape (E, n_edge_scalar)

    all_edges:
        Directed edge list, shape (E, 2)

    is_bond_A / is_bond_B:
        Float flags, shape (E,)

    is_persistent / is_forming / is_breaking / is_changed:
        Float reaction-status flags, shape (E,)
    """

    src: pt.Tensor
    dst: pt.Tensor
    edge_dir: pt.Tensor
    edge_features: pt.Tensor

    all_edges: pt.Tensor

    is_bond_A: pt.Tensor
    is_bond_B: pt.Tensor

    is_endpoint_bond: pt.Tensor
    is_distance_only: pt.Tensor
    is_persistent: pt.Tensor
    is_forming: pt.Tensor
    is_breaking: pt.Tensor
    is_changed: pt.Tensor


class E3AttentionLayer(nn.Module):
    """
    e3nn transition-path layer for the Newton-like refinement method.

    The layer:
      1. builds neighbors from current x, plus endpoint bonds from xA/xB
      2. builds scalar edge features
      3. computes e3nn tensor-product messages
      4. updates hidden irreps features using a gated equivariant nonlinearity
      5. updates coordinates using:
            - hidden 1o readout
            - direct neighbor-position update
            - endpoint anchors
    """

    def __init__( self,
                  irreps_node_str: str,
                  d_cutoff: float = 5.0,
                  n_rbf: int = 10,
                  self_interaction_init_scale: float = 0.0,
                  feature_residual_scale: float = 0.2,
    ) -> None:
        super().__init__()

        self.irreps_node = o3.Irreps(irreps_node_str)

        self.d_cutoff = d_cutoff
        self.n_rbf = n_rbf
        self.self_interaction_init_scale = self_interaction_init_scale
        self.feature_residual_scale = feature_residual_scale

        # Small inverse-distance smoothing parameter.
        self.eps = 0.02

        self.lmax = max(ir.l for _, ir in self.irreps_node)
        self.irreps_sh = o3.Irreps.spherical_harmonics(self.lmax)

        # Scalar edge features.
        #
        # Bond/status features:
        #   bondA
        #   bondB
        #   endpoint_bond
        #   distance_only
        #   persistent
        #   forming
        #   breaking
        #   changed
        #       = 8
        #
        # Current geometry:
        #   dist
        #   dist^2
        #   inv_dist
        #       = 3
        #
        # Endpoint geometry:
        #   dist_A
        #   inv_dist_A
        #   dist_B
        #   inv_dist_B
        #       = 4
        #
        # Distance deltas:
        #   dist_B - dist_A
        #   abs(dist_B - dist_A)
        #   dist - dist_A
        #   dist_B - dist
        #       = 4
        #
        # RBF:
        #   rbf(dist_raw)
        #   rbf(dist_A_raw)
        #   rbf(dist_B_raw)
        #       = 3 * n_rbf
        #
        # Total:
        #   8 + 3 + 4 + 4 + 3*n_rbf
        #   = 19 + 3*n_rbf
        self.rbf = DistanceRBFEmbedding( 0.0, d_cutoff, n_rbf=n_rbf )
        self.n_edge_scalar = 19 + 3 * self.rbf.out_dim

        # Tensor product for edge messages:
        #
        #   context_ij = concat(
        #       edge_scalar_ij,
        #       scalar_0e(f_src),
        #       scalar_0e(f_dst),
        #   )
        #
        #   weights_ij = radial_network(context_ij)
        #
        #   message_ij = TP(
        #       f_src,
        #       Y(r_ij),
        #       weights_ij,
        #   )
        #
        # f_dst enters through scalar context for the weights, not directly through the tensor product input.
        self.tp = o3.FullyConnectedTensorProduct( self.irreps_node, self.irreps_sh, self.irreps_node, shared_weights=False )

        # Scalar 0e subspace used for edge context.
        self.irreps_0e = o3.Irreps(
            [
                (mul, ir)
                for mul, ir in self.irreps_node
                if ir.l == 0 and ir.p == 1
            ]
        )
        assert self.irreps_0e.dim > 0, "NewtonE3NNLayer expects at least one 0e block."

        # Edge context = scalar edge features + source node scalars + destination node scalars.
        self.radial_context_dim = self.n_edge_scalar + 2 * self.irreps_0e.dim

        # Radial MLP produces tensor-product weights from scalar edge context.
        self.radial_network = MultiLayerPerceptron(
            [self.radial_context_dim, 256, 256, 256, self.tp.weight_numel],
            nn.GELU,
            "e3nn_radial_network",
        )

        # Residual scalar gate on each raw edge message before aggregation.
        # Zero-ish init behavior depends on MultiLayerPerceptron initialization,
        # but tanh gate keeps this bounded.
        self.edge_message_scalar_gate = MultiLayerPerceptron(
            [self.radial_context_dim, 128, 128, 1],
            nn.GELU,
            "e3nn_edge_message_scalar_gate",
        )

        # attention gating of the edge messages so they are normalized
        self.edge_attention_score = MultiLayerPerceptron(
            [self.radial_context_dim, 128, 128, 1],
            nn.GELU,
            "e3nn_edge_attention_score",
        )

        # Equivariant self-interaction after aggregation.
        self.self_interaction = o3.Linear( self.irreps_node, self.irreps_node )
        with pt.no_grad():
            for p in self.self_interaction.parameters():
                p.mul_(self.self_interaction_init_scale)

        # Gated equivariant nonlinearity for hidden feature updates.
        self._setup_gate()

        # Coordinate gates:
        #   gate_delta_x    : hidden 1o coordinate readout
        #   gate_neighbor   : direct neighbor coordinate update
        #   gate_xA         : reactant anchor
        #   gate_xB         : product anchor
        self.scalar_readout = o3.Linear( self.irreps_node, self.irreps_0e )
        self.coordinate_gate_network = MultiLayerPerceptron(
            [self.irreps_0e.dim, 128, 128, 4],
            nn.GELU,
            "e3nn_coordinate_gates",
        )

        # Project node features to one polar vector 1o for coordinate updates.
        self.coord_head = o3.Linear( self.irreps_node, o3.Irreps("1x1o") )

        # Direct neighbor-position update coefficient.
        # edge_position_message_ij = -edge_step_ij * edge_dir_ij
        self.edge_coordinate_network = MultiLayerPerceptron(
            [self.radial_context_dim, 128, 128, 1],
            nn.GELU,
            "e3nn_edge_coordinate_network",
        )

    def _setup_gate(self) -> None:
        """
        Set up an e3nn gated nonlinearity.

        Gate input irreps:
            scalars + gates + gated

        Gate output irreps:
            scalars + gated

        Example:
            node:     64x0e + 16x1o + 8x1e
            scalars:  64x0e
            gated:    16x1o + 8x1e
            gates:    24x0e
        """
        self.irreps_gate_scalars = o3.Irreps(
            [
                (mul, ir)
                for mul, ir in self.irreps_node
                if ir.l == 0 and ir.p == 1
            ]
        )

        self.irreps_gate_gated = o3.Irreps(
            [
                (mul, ir)
                for mul, ir in self.irreps_node
                if not (ir.l == 0 and ir.p == 1)
            ]
        )

        n_gates = sum(mul for mul, _ in self.irreps_gate_gated)

        self.irreps_gate_gates = o3.Irreps(f"{n_gates}x0e")

        self.irreps_pre_gate = (
            self.irreps_gate_scalars
            + self.irreps_gate_gates
            + self.irreps_gate_gated
        )

        self.pre_gate = o3.Linear(
            self.irreps_node,
            self.irreps_pre_gate,
        )

        self.gate = e3nn_nn.Gate(
            self.irreps_gate_scalars,
            [F.silu],
            self.irreps_gate_gates,
            [pt.sigmoid],
            self.irreps_gate_gated,
        )

    def _check_inputs( self, xA: Molecule, xB: Molecule, s: pt.Tensor, state: E3State ) -> pt.Tensor:
        """
        Validate shapes and return flattened s on the same device/dtype as x.
        """
        f = state.f
        x = state.x
        N = x.shape[0]

        assert xA.Z.shape == xB.Z.shape
        assert pt.equal( xA.Z, xB.Z ), "`xA` and `xB` must have the same atoms in the same ordering."

        assert xA.x.shape == xB.x.shape
        assert xA.x.shape == x.shape

        assert f.ndim == 2
        assert f.shape == (N, self.irreps_node.dim)

        assert x.ndim == 2
        assert x.shape == (N, 3)

        s = s.flatten().to(device=x.device, dtype=x.dtype)
        assert s.numel() == N, "`s` must have one value per atom."

        return s


    def _get_edges_and_bond_flags( self, xA: Molecule, xB: Molecule, x: pt.Tensor ) -> tuple[pt.Tensor, pt.Tensor, pt.Tensor]:
        """
        Return all_edges, is_bond_A, is_bond_B.
        """
        device = x.device
        dtype = x.dtype

        all_edges, is_bond_A, is_bond_B = findAllNeighborsReactantProduct( xA, xB, x, self.d_cutoff )
        all_edges = all_edges.to(device=device).long()
        is_bond_A = is_bond_A.to(device=device, dtype=dtype)
        is_bond_B = is_bond_B.to(device=device, dtype=dtype)

        return all_edges, is_bond_A, is_bond_B


    def _build_edge_features( self, xA: Molecule, xB: Molecule, x: pt.Tensor ) -> EdgeData:
        """
        Neighbor search and scalar edge-feature construction.

        Edges are directed pairs that are either:

            - bonds in xA,
            - bonds in xB,
            - or current distance neighbors under x.

        The endpoint-bond flags are used to compute:

            persistent
            forming
            breaking
            changed
            distance_only
        """
        all_edges, is_bond_A, is_bond_B = self._get_edges_and_bond_flags( xA, xB, x )

        src = all_edges[:, 0].long()
        dst = all_edges[:, 1].long()

        dtype = x.dtype
        device = x.device

        # Bond / reaction status features.
        bondA = is_bond_A[:, None].to(device=device, dtype=dtype)
        bondB = is_bond_B[:, None].to(device=device, dtype=dtype)

        endpoint_bond = pt.maximum(bondA, bondB)
        distance_only = 1.0 - endpoint_bond

        persistent = bondA * bondB
        forming = (1.0 - bondA) * bondB
        breaking = bondA * (1.0 - bondB)
        changed = pt.abs(bondB - bondA)

        # Current edge geometry.
        #
        # This is differentiable with respect to current x.
        edge_vec = x[dst] - x[src]  # (E, 3)
        dist_raw = pt.sqrt( (edge_vec * edge_vec).sum(dim=1, keepdim=True).clamp_min(1.0e-8) )
        edge_dir = edge_vec / dist_raw
        dist = dist_raw / self.d_cutoff
        inv_dist = self.eps / pt.sqrt(dist**2 + self.eps**2)

        # Endpoint edge geometry.
        xA_pos = xA.x.to(device=device, dtype=dtype)
        xB_pos = xB.x.to(device=device, dtype=dtype)

        edge_vec_A = xA_pos[dst] - xA_pos[src]
        dist_A_raw = pt.sqrt(
            (edge_vec_A * edge_vec_A)
            .sum(dim=1, keepdim=True)
            .clamp_min(1.0e-8)
        )
        dist_A = dist_A_raw / self.d_cutoff
        inv_dist_A = self.eps / pt.sqrt(dist_A**2 + self.eps**2)

        edge_vec_B = xB_pos[dst] - xB_pos[src]
        dist_B_raw = pt.sqrt(
            (edge_vec_B * edge_vec_B)
            .sum(dim=1, keepdim=True)
            .clamp_min(1.0e-8)
        )
        dist_B = dist_B_raw / self.d_cutoff
        inv_dist_B = self.eps / pt.sqrt(dist_B**2 + self.eps**2)

        # Endpoint/current distance changes.
        dist_delta_AB = dist_B - dist_A
        abs_dist_delta_AB = pt.abs(dist_delta_AB)

        dist_delta_current_A = dist - dist_A
        dist_delta_B_current = dist_B - dist

        # RBF features.
        rbf_current = self.rbf(dist_raw)
        rbf_A = self.rbf(dist_A_raw)
        rbf_B = self.rbf(dist_B_raw)

        # Final scalar edge features.
        edge_features = pt.cat(
            (
                # Bond / reaction status.
                bondA,
                bondB,
                endpoint_bond,
                distance_only,
                persistent,
                forming,
                breaking,
                changed,

                # Current geometry.
                dist,
                dist**2,
                inv_dist,

                # Endpoint geometry.
                dist_A,
                inv_dist_A,
                dist_B,
                inv_dist_B,

                # Distance-change geometry.
                dist_delta_AB,
                abs_dist_delta_AB,
                dist_delta_current_A,
                dist_delta_B_current,

                # RBF features.
                rbf_current,
                rbf_A,
                rbf_B,
            ),
            dim=1,
        )

        assert edge_features.shape[1] == self.n_edge_scalar, (
            f"Expected edge feature dimension {self.n_edge_scalar}, "
            f"got {edge_features.shape[1]}."
        )

        return EdgeData(
            src=src,
            dst=dst,
            edge_dir=edge_dir,
            edge_features=edge_features,
            all_edges=all_edges,
            is_bond_A=is_bond_A,
            is_bond_B=is_bond_B,
            is_endpoint_bond=endpoint_bond[:, 0],
            is_distance_only=distance_only[:, 0],
            is_persistent=persistent[:, 0],
            is_forming=forming[:, 0],
            is_breaking=breaking[:, 0],
            is_changed=changed[:, 0],
        )

    def _edge_context( self, f: pt.Tensor, edges: EdgeData, *, scalar_features: pt.Tensor | None = None ) -> pt.Tensor:
        """
        Build scalar edge context:

            edge scalar features
            source node scalar 0e features
            destination node scalar 0e features
        """
        if scalar_features is None:
            scalar_features = self.scalar_readout( f )

        return pt.cat( ( edges.edge_features, scalar_features[edges.src], scalar_features[edges.dst] ), dim=1 ) # type: ignore

    def _aggregate_messages(self, f: pt.Tensor, edges: EdgeData) -> pt.Tensor:
        """
        e3nn tensor-product message passing with destination-wise edge attention.

        For each edge i -> j:

            context_ij = concat(edge_features_ij, scalar_0e(f_i), scalar_0e(f_j))
            weights_ij = radial_network(context_ij)
            message_ij = TP(f_i, Y(r_ij), weights_ij)

        Then incoming messages are attention-weighted per destination node j.

        We use degree-scaled softmax attention:

            alpha_ij = softmax_i(score_ij over incoming edges i -> j)
            multiplier_ij = degree_j * alpha_ij

        so the average incoming multiplier stays close to 1.
        """
        edge_attr = o3.spherical_harmonics(
            self.irreps_sh,
            edges.edge_dir,
            normalize=True,
            normalization="component",
        )

        # Scalar context for tensor-product weights and attention scores.
        node_scalars = self.scalar_readout(f)

        radial_context = self._edge_context( f, edges, scalar_features=node_scalars )

        # Edge message from src to dst.
        weights = self.radial_network(radial_context)
        edge_messages = self.tp( f[edges.src], edge_attr, weights )

        # Destination-wise attention scores.
        attention_logits = self.edge_attention_score(radial_context)
        alpha = segment_softmax( attention_logits, edges.dst, n_segments=f.shape[0] )  # (E, 1), sums to 1 over incoming edges per dst

        # Degree-scaled attention keeps the scale comparable to sum aggregation.
        degree = pt.zeros( (f.shape[0], 1), dtype=f.dtype, device=f.device )
        degree.index_add_( 0, edges.dst, pt.ones_like(alpha), )

        attention_multiplier = alpha * degree[edges.dst]
        edge_messages = attention_multiplier * edge_messages

        # Optional residual scalar gate after attention.
        gate_logits = self.edge_message_scalar_gate(radial_context)
        gate_residual = 0.5 * pt.tanh(gate_logits)
        edge_messages = (1.0 + gate_residual) * edge_messages

        agg = pt.zeros_like(f)
        agg.index_add_(0, edges.dst, edge_messages)

        return agg

    def _update_features( self, f: pt.Tensor, agg: pt.Tensor ) -> pt.Tensor:
        """
        Residual equivariant feature update.

        We apply the gate to the update, not the whole residual state:

            f_new = f + scale * Gate(PreGate(SelfInteraction(agg)))
        """
        f_update = self.self_interaction(agg)
        f_update = self.gate(self.pre_gate(f_update))

        f_new = f + self.feature_residual_scale * f_update

        return f_new

    def _coordinate_update(
        self,
        xA: Molecule,
        xB: Molecule,
        s: pt.Tensor,
        x: pt.Tensor,
        f_new: pt.Tensor,
        edges: EdgeData,
    ) -> pt.Tensor:
        """
        Coordinate update from:
          1. hidden 1o readout
          2. direct neighbor displacement update
          3. endpoint anchors
        """
        dtype = x.dtype
        device = x.device

        xA_pos = xA.x.to(device=device, dtype=dtype)
        xB_pos = xB.x.to(device=device, dtype=dtype)

        # Hidden-state coordinate readout.
        delta_x = self.coord_head( f_new )  # (N, 3)

        # Scalar node features for gates and edge-coordinate coefficient.
        scalar_features = self.scalar_readout( f_new )  # (N, irreps_0e.dim)

        edge_coord_context = self._edge_context(
            f_new,
            edges,
            scalar_features=scalar_features,
        )

        # Direct neighbor-position update.
        edge_step = self.edge_coordinate_network(edge_coord_context)  # (E, 1)

        # Move dst using messages from src along edge direction.
        # edge_dir = x_dst - x_src normalized.
        # Negative sign pulls dst toward src if edge_step is positive.
        edge_position_messages = -edge_step * edges.edge_dir  # (E, 3)

        neighbor_update = pt.zeros_like(x)
        neighbor_update.index_add_(
            0,
            edges.dst,
            edge_position_messages,
        )

        # Nodewise coordinate gates.
        coord_gates = pt.tanh( self.coordinate_gate_network( scalar_features ) )  # (N, 4)

        gate_delta_x = coord_gates[:, 0:1]
        gate_neighbor = coord_gates[:, 1:2]
        gate_xA = coord_gates[:, 2:3]
        gate_xB = coord_gates[:, 3:4]

        dx = (
            gate_delta_x * delta_x
            + gate_neighbor * neighbor_update
            + gate_xA * (1.0 - s[:, None]) * (xA_pos - x)
            + gate_xB * s[:, None] * (xB_pos - x)
        )

        return dx

    def forward( self, xA: Molecule, xB: Molecule, s: pt.Tensor, state: E3State ) -> tuple[pt.Tensor, pt.Tensor]:
        """
        Apply one Newton/e3nn refinement layer.

        Parameters
        ----------
        xA:
            Reactant molecule.
        xB:
            Product molecule.
        s:
            Per-atom arclength coordinate, shape (N,).
        state:
            Current E3State.

        Returns
        -------
        f_new:
            Updated irreps node features.
        dx:
            Coordinate update, shape (N, 3).
        """
        s = self._check_inputs(xA, xB, s, state)

        f = state.f
        x = state.x

        edges = self._build_edge_features( xA, xB, x )

        agg = self._aggregate_messages(f, edges)
        f_new = self._update_features(f, agg)
        dx = self._coordinate_update(xA, xB, s, x, f_new, edges)

        return f_new, dx
    
def segment_softmax( scores: pt.Tensor, index: pt.Tensor, n_segments: int ) -> pt.Tensor:
    """
    Softmax over variable-size groups.

    scores:
        Shape (E, 1)

    index:
        Shape (E,), group index for each edge.
        For edge attention, this is edges.dst.

    n_segments:
        Number of groups. Usually number of nodes.

    Returns
    -------
    alpha:
        Shape (E, 1), with alpha summing to 1 over each group.
    """
    assert scores.ndim == 2 and scores.shape[1] == 1
    assert index.ndim == 1
    assert scores.shape[0] == index.shape[0]

    scores_flat = scores[:, 0]
    max_per_segment = pt.full( (n_segments,), -pt.inf, dtype=scores.dtype, device=scores.device )

    max_per_segment.scatter_reduce_( 0, index, scores_flat, reduce="amax", include_self=True )

    shifted = scores_flat - max_per_segment[index]
    exp_scores = pt.exp(shifted)

    denom = pt.zeros( (n_segments,), dtype=scores.dtype, device=scores.device )

    denom.scatter_add_( 0, index, exp_scores )
    alpha = exp_scores / denom[index].clamp_min(1.0e-12)

    return alpha[:, None]