import torch as pt
import torch.nn as nn
import torch.nn.functional as F

from e3nn import o3
from e3nn import nn as e3nn_nn
from dataclasses import dataclass

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

    src, dst       : edge indices, shape (E,)
    edge_dir       : unit edge direction, shape (E, 3)
    dx_to_src      : x[src] - x[dst], shape (E, 3)
    edge_features  : scalar edge features, shape (E, n_edge_scalar)
    """
    src: pt.Tensor
    dst: pt.Tensor
    edge_dir: pt.Tensor
    dx_to_src: pt.Tensor
    edge_features: pt.Tensor

class TransitionPathE3NNLayer(nn.Module):
    """
    e3nn transition-path layer.

    The layer:
      1. builds neighbors from current x
      2. builds scalar edge features
      3. computes e3nn tensor-product messages
      4. updates hidden irreps features using a gated equivariant nonlinearity
      5. updates coordinates using:
            - hidden 1o readout
            - direct neighbor-position update
            - endpoint anchors
    """

    def __init__(
        self,
        irreps_node_str: str,
        d_cutoff: float = 5.0,
        n_rbf: int = 10,
        self_interaction_init_scale: float = 0.1,
        feature_residual_scale : float = 0.2
    ) -> None:
        super().__init__()

        self.irreps_node = o3.Irreps(irreps_node_str)
        self.d_cutoff = d_cutoff
        self.n_rbf = n_rbf
        self.self_interaction_init_scale = self_interaction_init_scale
        self.feature_residual_scale = feature_residual_scale

        self.lmax = max(ir.l for _, ir in self.irreps_node)
        self.irreps_sh = o3.Irreps.spherical_harmonics(self.lmax)

        # Scalar edge features:
        #   bondA, bondB, d, d^2, dA, dB, dA-dB, RBF(d), RBF(dA), RBF(dB)
        self.rbf = DistanceRBFEmbedding(0.0, d_cutoff, n_rbf=n_rbf)
        self.n_edge_scalar = 7 + 3 * self.rbf.out_dim

        # Tensor product for edge messages:
        #   message_ij = TP( f_j, Y(r_ij); weights(edge_scalar) )
        self.tp = o3.FullyConnectedTensorProduct( self.irreps_node, self.irreps_sh, self.irreps_node, shared_weights=False)

        # Radial MLP produces tensor-product weights from scalar edge features
        self.irreps_0e = o3.Irreps([
            (mul, ir) for mul, ir in self.irreps_node
            if ir.l == 0 and ir.p == 1
        ])
        assert self.irreps_0e.dim > 0, "TransitionPathE3NNLayer expects at least one 0e block."
        self.radial_context_dim = self.n_edge_scalar + 2 * self.irreps_0e.dim
        self.radial_network = MultiLayerPerceptron(
            [self.radial_context_dim, 64, 64, self.tp.weight_numel], # type: ignore
            nn.GELU, "e3nn_radial_network" )

        # A simple equivariant self-interaction after aggregation.
        # Initialize with zeros for stability.
        self.self_interaction = o3.Linear(self.irreps_node, self.irreps_node)
        with pt.no_grad():
            for p in self.self_interaction.parameters():
                p.mul_( self.self_interaction_init_scale )

        # Gated equivariant nonlinearity for hidden feature updates.
        self._setup_gate()

        # Coordinate gates:
        #   gate_delta_x    : hidden 1o coordinate readout
        #   gate_neighbor   : direct neighbor coordinate update
        #   gate_xA         : reactant anchor
        #   gate_xB         : product anchor
        self.scalar_readout = o3.Linear(self.irreps_node, self.irreps_0e)
        self.coordinate_gate_network = MultiLayerPerceptron(
            [self.irreps_0e.dim, 64, 64, 4],
            nn.GELU,
            "e3nn_coordinate_gates",
        )

        # Project node features to one polar vector (1o) for coordinate updates.
        self.coord_head = o3.Linear(self.irreps_node, o3.Irreps("1x1o"))

        # Direct neighbor-position update coefficient.
        # edge_update_ij = edge_step_ij * (x_src - x_dst)
        self.edge_coordinate_network = MultiLayerPerceptron(
            [self.n_edge_scalar + 2 * self.irreps_0e.dim, 64, 64, 1],
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

        For example:
            node:     64x0e + 16x1o + 8x1e
            scalars:  64x0e
            gated:    16x1o + 8x1e
            gates:    24x0e
        """
        self.irreps_gate_scalars = o3.Irreps([
            (mul, ir) for mul, ir in self.irreps_node
            if ir.l == 0 and ir.p == 1
        ])
        self.irreps_gate_gated = o3.Irreps([
            (mul, ir) for mul, ir in self.irreps_node
            if not (ir.l == 0 and ir.p == 1)
        ])

        n_gates = sum(mul for mul, _ in self.irreps_gate_gated)
        self.irreps_gate_gates = o3.Irreps(f"{n_gates}x0e")

        self.irreps_pre_gate = self.irreps_gate_scalars + self.irreps_gate_gates + self.irreps_gate_gated
        self.pre_gate = o3.Linear(self.irreps_node, self.irreps_pre_gate)

        self.gate = e3nn_nn.Gate(
            self.irreps_gate_scalars,
            [F.silu],
            self.irreps_gate_gates,
            [pt.sigmoid],
            self.irreps_gate_gated,
        )

    def _check_inputs( self, xA: Molecule, xB: Molecule, s: pt.Tensor, state: E3State, ) -> pt.Tensor:
        """
        Validate shapes and return flattened s.
        """
        f = state.f
        x = state.x
        N = x.shape[0]

        assert pt.all(xA.Z == xB.Z), "`xA` and `xB` must have the same atoms in the same ordering."

        s = s.flatten()

        assert f.ndim == 2 and f.shape[0] == N and f.shape[1] == self.irreps_node.dim
        assert x.ndim == 2 and x.shape == (N, 3)
        assert s.numel() == N, "`s` must have one value per atom."

        return s
    
    def _build_edges( self, xA: Molecule, xB: Molecule, x: pt.Tensor, ) -> EdgeData:
        """
        Neighbor search and scalar edge-feature construction.
        """
        all_edges, is_bond_A, is_bond_B = findAllNeighborsReactantProduct( xA, xB, x, self.d_cutoff )
        src = all_edges[:, 0]
        dst = all_edges[:, 1]

        # Current edge geometry.
        edge_vec = x[dst] - x[src]  # (E, 3)
        dist_raw = pt.sqrt((edge_vec * edge_vec).sum(dim=1, keepdim=True).clamp_min(1e-8))
        edge_dir = edge_vec / dist_raw
        dist = dist_raw / self.d_cutoff

        # Old sign convention for direct coordinate update:
        # move dst along vector pointing from dst to src.
        dx_to_src = x[src] - x[dst]  # (E, 3)

        # Endpoint edge geometry.
        edge_vec_A = xA.x[dst] - xA.x[src]
        dist_A_raw = pt.sqrt((edge_vec_A * edge_vec_A).sum(dim=1, keepdim=True).clamp_min(1e-8))
        dist_A = dist_A_raw / self.d_cutoff

        edge_vec_B = xB.x[dst] - xB.x[src]
        dist_B_raw = pt.sqrt((edge_vec_B * edge_vec_B).sum(dim=1, keepdim=True).clamp_min(1e-8))
        dist_B = dist_B_raw / self.d_cutoff

        bondA = is_bond_A[:, None].to(x.dtype)
        bondB = is_bond_B[:, None].to(x.dtype)

        edge_features = pt.cat( ( bondA, bondB, dist, dist ** 2, dist_A, dist_B,
                dist_A - dist_B, self.rbf(dist_raw), self.rbf(dist_A_raw), self.rbf(dist_B_raw), ), dim=1, )

        return EdgeData( src, dst, edge_dir, dx_to_src, edge_features )
    
    def _aggregate_messages( self, f: pt.Tensor, edges: EdgeData ) -> pt.Tensor:
        """
        e3nn tensor-product message passing.
        """
        edge_attr = o3.spherical_harmonics(
            self.irreps_sh,
            edges.edge_dir,
            normalize=True,
            normalization="component",
        )
        node_scalars = self.scalar_readout(f)  # (N, irreps_0e.dim)

        radial_context = pt.cat( ( edges.edge_features, node_scalars[edges.src], node_scalars[edges.dst], ), dim=1, )  # (E, n_edge_scalar + 2 * irreps_0e.dim)
        weights = self.radial_network(radial_context)

        edge_messages = self.tp( f[edges.src], edge_attr, weights, )
        agg = pt.zeros_like(f)
        agg.index_add_(0, edges.dst, edge_messages)

        return agg
    
    def _update_features( self, f: pt.Tensor, agg: pt.Tensor, ) -> pt.Tensor:
        """
        Residual equivariant feature update with optional gated nonlinearity.

        We apply the gate to the update, not to the whole residual state:

            f_new = f + scale * Gate(PreGate(SelfInteraction(agg)))

        This preserves the residual/identity structure.
        """
        f_update = self.self_interaction(agg)
        f_update = self.gate(self.pre_gate(f_update))
        f_new = f + self.feature_residual_scale * f_update
        return f_new
    
    def _coordinate_update( self, xA: Molecule, xB: Molecule, s: pt.Tensor, x: pt.Tensor, f_new: pt.Tensor, edges: EdgeData, ) -> pt.Tensor:
        """
        Coordinate update from:
          1. hidden 1o readout
          2. direct neighbor displacement update
          3. endpoint anchors
        """
        # Hidden-state coordinate readout.
        delta_x = self.coord_head(f_new)  # (N, 3)

        # Scalar features for gates and edge-coordinate coefficient.
        scalar_features = self.scalar_readout(f_new)  # (N, irreps_0e.dim)

        # Direct neighbor-position update.
        edge_coord_context = pt.cat( ( edges.edge_features, scalar_features[edges.src], scalar_features[edges.dst], ), dim=1, )

        edge_step = self.edge_coordinate_network(edge_coord_context)  # (E, 1)
        edge_position_messages = edge_step * edges.dx_to_src          # (E, 3)

        neighbor_update = pt.zeros_like(x)
        neighbor_update.index_add_(0, edges.dst, edge_position_messages)

        # Nodewise coordinate gates.
        coord_gates = self.coordinate_gate_network(scalar_features)  # (N, 4)

        gate_delta_x = coord_gates[:, 0:1]
        gate_neighbor = coord_gates[:, 1:2]
        gate_xA = coord_gates[:, 2:3]
        gate_xB = coord_gates[:, 3:4]

        x_new = (
            x
            + gate_delta_x * delta_x
            + gate_neighbor * neighbor_update
            + gate_xA * (1.0 - s[:, None]) * (xA.x - x)
            + gate_xB * s[:, None] * (xB.x - x)
        )

        return x_new

    def forward( self, xA: Molecule, xB: Molecule, s: pt.Tensor, state: E3State, ) -> E3State:
        s = self._check_inputs(xA, xB, s, state)

        f = state.f
        x = state.x

        edges = self._build_edges(xA, xB, x)
        agg = self._aggregate_messages(f, edges)
        f_new = self._update_features(f, agg)
        x_new = self._coordinate_update(xA, xB, s, x, f_new, edges)

        return E3State(f=f_new, x=x_new)