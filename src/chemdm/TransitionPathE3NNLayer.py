import torch as pt
import torch.nn as nn

from e3nn import o3
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


class TransitionPathE3NNLayer(nn.Module):
    """
    First-pass e3nn transition-path layer.

    This layer performs:
      1. neighbor search from current x
      2. scalar edge feature construction
      3. spherical harmonics from edge directions
      4. equivariant message passing with e3nn tensor product
      5. node feature update
      6. coordinate update from the 1o part of the node features
    """

    def __init__( self,
        irreps_node_str : str,
        d_cutoff: float = 5.0,
        n_rbf: int = 10,
    ) -> None:
        super().__init__()

        self.irreps_node = o3.Irreps(irreps_node_str)
        self.d_cutoff = d_cutoff
        self.n_rbf = n_rbf

        self.lmax = max(ir.l for _, ir in self.irreps_node)
        self.irreps_sh = o3.Irreps.spherical_harmonics(self.lmax)

        # Scalar edge features:
        #   bondA, bondB, d, d^2, dA, dB, dA-dB, RBF(d), RBF(dA), RBF(dB)
        self.rbf = DistanceRBFEmbedding(0.0, d_cutoff, n_rbf=n_rbf)
        self.n_edge_scalar = 7 + 3 * self.rbf.out_dim

        # Tensor product for edge messages:
        #   message_ij = TP( f_j, Y(r_ij); weights(edge_scalar) )
        self.tp = o3.FullyConnectedTensorProduct( self.irreps_node, self.irreps_sh, self.irreps_node, shared_weights=False)
        print('Number of TPFullyConnected weights: ', self.tp.weight_numel)

        # Radial MLP produces tensor-product weights from scalar edge features
        self.radial_network = MultiLayerPerceptron(
            [self.n_edge_scalar, 64, 64, self.tp.weight_numel], # type: ignore
            nn.GELU, "e3nn_radial_network" )

        # A simple equivariant self-interaction after aggregation.
        # Initialize with zeros for stability.
        self.self_interaction = o3.Linear(self.irreps_node, self.irreps_node)
        with pt.no_grad():
            for p in self.self_interaction.parameters():
                p.mul_( 0.2 )

        # Read out scalar-even channels for coordinate gates
        self.irreps_0e = o3.Irreps([ 
            (mul, ir) for mul, ir in self.irreps_node
            if ir.l == 0 and ir.p == 1
        ])
        self.scalar_readout = o3.Linear(self.irreps_node, self.irreps_0e)
        self.coordinate_gate_network = MultiLayerPerceptron(
            [self.irreps_0e.dim, 64, 64, 4],
            nn.GELU, "e3nn_coordinate_gates")

        # Project node features to one polar vector (1o) for coordinate updates.
        self.coord_head = o3.Linear(self.irreps_node, o3.Irreps("1x1o"))

        # Direct neighbor-position update coefficient.
        #   edge_update_ij = alpha_ij * (x_src - x_dst)
        # The input is scalar-only, so this remains equivariant.
        self.edge_coordinate_network = MultiLayerPerceptron(
            [self.n_edge_scalar + 2 * self.irreps_0e.dim, 64, 64, 1],
            nn.GELU, "e3nn_edge_coordinate_network", )

    def forward( self, xA: Molecule, xB: Molecule, s: pt.Tensor,  state: E3State ) -> E3State:
        # Unpack the equivariant state.
        f = state.f
        x = state.x
        N = x.shape[0]

        # Some essential checks
        assert pt.all(xA.Z == xB.Z), "`xA` and `xB` must have the same atoms in the same ordering."
        s = s.flatten()
        assert f.ndim == 2 and f.shape[0] == N and f.shape[1] == self.irreps_node.dim
        assert x.ndim == 2 and x.shape == (N, 3)
        assert s.numel() == N, "`s` must have one value per atom."

        # 1. Neighbor search from current positions
        all_edges, is_bond_A, is_bond_B = findAllNeighborsReactantProduct(xA, xB, x, self.d_cutoff)
        src = all_edges[:, 0]
        dst = all_edges[:, 1]

        # 2. Edge geometry
        edge_vec = x[dst] - x[src]  # (E, 3)
        dist = pt.sqrt( (edge_vec * edge_vec).sum(dim=1, keepdim=True).clamp_min(1e-8) )
        edge_dir = edge_vec / dist

        edge_vec_A = xA.x[dst] - xA.x[src]
        dist_A = pt.sqrt( (edge_vec_A * edge_vec_A).sum(dim=1, keepdim=True).clamp_min(1e-8) )

        edge_vec_B = xB.x[dst] - xB.x[src]
        dist_B = pt.sqrt( (edge_vec_B * edge_vec_B).sum(dim=1, keepdim=True).clamp_min(1e-8) )

        bondA = is_bond_A[:, None].to(x.dtype)
        bondB = is_bond_B[:, None].to(x.dtype)

        edge_scalar = pt.cat( ( bondA, bondB, dist, dist**2, dist_A, dist_B, 
                               dist_A - dist_B, self.rbf(dist), self.rbf(dist_A), self.rbf(dist_B), ), dim=1, )  # (E, n_edge_scalar)

        # 3. Spherical harmonics on edge directions
        edge_attr = o3.spherical_harmonics( self.irreps_sh, edge_dir, normalize=True, normalization="component", )  # (E, irreps_sh.dim)

        # 4. Radial weights from scalar edge features
        weights = self.radial_network(edge_scalar)  # (E, tp.weight_numel)

        # 5. Equivariant messages
        # source node features live on src
        edge_messages = self.tp( f[src], edge_attr, weights )  # (E, irreps_node.dim)
        agg = pt.zeros_like(f)
        agg.index_add_( 0, dst, edge_messages )

        # 6. Node feature update
        f_new = f + self.self_interaction( agg )

        # 7. Coordinate update from the 1o part of f_new
        delta_x = self.coord_head( f_new )  # (N, 3)
        scalar_features = self.scalar_readout( f_new )  # (N, irreps_0e.dim)
        dx_to_src = x[src] - x[dst]  # (E, 3)
        edge_coord_context = pt.cat( (
                edge_scalar,
                scalar_features[src],
                scalar_features[dst], ), dim=1 )  # (E, n_edge_scalar + 2*irreps_0e.dim)

        edge_alpha = self.edge_coordinate_network(edge_coord_context)  # (E, 1)
        edge_position_messages = edge_alpha * dx_to_src                # (E, 3)
        neighbor_update = pt.zeros_like(x)
        neighbor_update.index_add_(0, dst, edge_position_messages)     # (N, 3)

        coord_gates = self.coordinate_gate_network(scalar_features)    # (N, 4)
        gate_delta_x = coord_gates[:, 0:1]
        gate_neighbor = coord_gates[:, 1:2]
        gate_xA = coord_gates[:, 2:3]
        gate_xB = coord_gates[:, 3:4]

        x_new = x + gate_delta_x * delta_x + gate_neighbor * neighbor_update \
            + gate_xA * (1.0 - s[:, None]) * (xA.x - x) \
            + gate_xB * s[:, None] * (xB.x - x)

        return E3State(f=f_new, x=x_new)