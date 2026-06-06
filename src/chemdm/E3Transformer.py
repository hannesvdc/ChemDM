from __future__ import annotations

from dataclasses import dataclass

import torch as pt
import torch.nn as nn

from e3nn import o3

from chemdm.TransitionPathMoleculeEmbedding import TPMoleculeEmbedding
from chemdm.embedding import ArcLengthEmbedding
from chemdm.MoleculeGraph import Molecule, findFixedUnionNeighbors
from chemdm.E3AttentionLayer import E3AttentionLayer, E3State



class E3Transformer(nn.Module):
    """
    Transition path network using:

      - endpoint-conditioned scalar node initialization from TPMoleculeEmbedding
      - separate arclength embedding
      - vector initialization from endpoint geometry
      - shared e3nn-based refinement operator
      - optional static edge features passed into each refinement layer

    Interpretation:
        x^(0) = linear interpolation between xA and xB
        x^(k+1) = learned refinement of x^(k)

    The same NewtonE3NNLayer is reused at every refinement step.
    """

    def __init__( self,
                  irreps_node_str: str = "64x0e + 16x1o + 8x1e",
                  n_refinement_steps: int = 7,
                  d_cutoff: float = 5.0,
                  n_freq: int = 8,
                  n_rbf: int = 10,
                  tp_embedding_dim: int = 64,
                  tp_embedding_hidden_dim: int = 128,
                  tp_embedding_hidden_layers: int = 2,
                  initial_edge_feature_dim: int = 0,
                  use_fixed_neighbors: bool = True,
    ) -> None:
        super().__init__()

        self.irreps_node = o3.Irreps( irreps_node_str )

        self.n_refinement_steps = n_refinement_steps
        self.d_cutoff = d_cutoff
        self.n_freq = n_freq
        self.n_rbf = n_rbf

        # When True (default), forward() computes one neighbor graph from the
        # endpoints (xA, xB) and reuses it across every refinement layer.
        # When False, each layer recomputes a current-x distance graph from
        # scratch (legacy behavior — heavier but adapts to coordinate updates).
        self.use_fixed_neighbors = use_fixed_neighbors

        self.tp_embedding_dim = tp_embedding_dim
        self.initial_edge_feature_dim = initial_edge_feature_dim

        # Endpoint-conditioned scalar node embedding.
        self.tp_molecule_embedding = TPMoleculeEmbedding(
            embedding_dim=tp_embedding_dim,
            hidden_dim=tp_embedding_hidden_dim,
            n_hidden_layers=tp_embedding_hidden_layers,
        )

        # Arclength embedding.
        self.arclength_embedding = ArcLengthEmbedding(self.n_freq)

        # Initial scalar feature dimension before lifting into 0e irreps.
        self.scalar_init_dim = self.tp_embedding_dim + self.arclength_embedding.getNumberOfFeatures()

        # 0e output block.
        self.irreps_0e_out = o3.Irreps(
            [
                (mul, ir)
                for mul, ir in self.irreps_node
                if ir.l == 0 and ir.p == 1
            ]
        )
        assert self.irreps_0e_out.dim > 0, "Expected at least one 0e block in irreps_node."

        self.irreps_0e_init = o3.Irreps(f"{self.scalar_init_dim}x0e")
        self.initial_0e_lift = o3.Linear( self.irreps_0e_init, self.irreps_0e_out )

        # 1o output block. Initial vector channels:
        #   1) xB - xA
        #   2) x  - xA
        #   3) x  - xB
        self.irreps_1o_out = o3.Irreps(
            [
                (mul, ir)
                for mul, ir in self.irreps_node
                if ir.l == 1 and ir.p == -1
            ]
        )

        self.irreps_1o_init = o3.Irreps("3x1o")
        self.initial_1o_lift = o3.Linear( self.irreps_1o_init, self.irreps_1o_out )

        # 1e and 2e output blocks. These are zero-initialized for now.
        self.irreps_1e_out = o3.Irreps(
            [
                (mul, ir)
                for mul, ir in self.irreps_node
                if ir.l == 1 and ir.p == 1
            ]
        )

        self.irreps_2e_out = o3.Irreps(
            [
                (mul, ir)
                for mul, ir in self.irreps_node
                if ir.l == 2 and ir.p == 1
            ]
        )

        # Shared e3nn refinement layer, which accepts initial (molecular) edge features.
        self.refinement_layer = E3AttentionLayer(
            irreps_node_str=irreps_node_str,
            d_cutoff=self.d_cutoff,
            n_rbf=n_rbf,
        )

    def initialize_state( self, xA: Molecule, xB: Molecule, s: pt.Tensor ) -> E3State:
        """
        Initialize node features and coordinates.

        Scalar 0e features:
            TPMoleculeEmbedding(xA, xB) + arclength embedding

        Coordinates:
            linear interpolation between xA and xB

        Vector 1o features:
            xB - xA
            x  - xA
            x  - xB
        """
        N = len(xA.Z)

        # Endpoint-conditioned scalar node embedding.
        tp_embed = self.tp_molecule_embedding( xA, xB )  # (N, c_tp)

        # Arclength embedding.
        s_embed = self.arclength_embedding(s)
        if s_embed.ndim == 1:
            s_embed = s_embed[None, :].expand(N, -1)

        # Linear-path coordinate initialization.
        x = (1.0 - s[:, None]) * xA.x + s[:, None] * xB.x

        # Scalar initial 0e features.
        scalar_init = pt.cat( [ tp_embed, s_embed ], dim=1 )
        assert scalar_init.shape == (N, self.scalar_init_dim), (
            f"Expected scalar_init shape {(N, self.scalar_init_dim)}, "
            f"got {tuple(scalar_init.shape)}"
        )
        f_0e = self.initial_0e_lift(scalar_init)

        # Vector initial 1o features.
        v1 = xB.x - xA.x
        v2 = x - xA.x
        v3 = x - xB.x

        vector_init = pt.stack((v1, v2, v3), dim=1)  # (N, 3, 3)
        vector_init = vector_init.reshape(N, -1)    # (N, 9), interpreted as 3x1o

        f_1o = self.initial_1o_lift(vector_init)

        # Higher/parity features are zero-initialized for now.
        f_1e = pt.zeros(
            N,
            self.irreps_1e_out.dim,
            device=x.device,
            dtype=x.dtype,
        )

        f_2e = pt.zeros(
            N,
            self.irreps_2e_out.dim,
            device=x.device,
            dtype=x.dtype,
        )

        # This assumes irreps_node is ordered as:
        #   0e blocks, then 1o blocks, then 1e blocks, then 2e blocks.
        #
        # This is true for:
        #   "64x0e + 16x1o + 8x1e"
        #
        # If irreps_node order changes, replace this with irreps-aware assembly.
        f = pt.cat((f_0e, f_1o, f_1e, f_2e), dim=1)

        assert f.shape == (N, self.irreps_node.dim), (
            f"Expected f shape {(N, self.irreps_node.dim)}, "
            f"got {tuple(f.shape)}"
        )

        return E3State(f=f, x=x)
    

    def refine_state( self, xA: Molecule, xB: Molecule, s: pt.Tensor, state: E3State,
        *,
        fixed_neighbors: tuple[pt.Tensor, pt.Tensor, pt.Tensor] | None = None,
    ) -> tuple[E3State, list[E3State]]:
        """
        Apply the shared refinement layer multiple times.

        During training, returning all states is useful because it allows losses
        on intermediate refinements.

        `fixed_neighbors`, if given, is passed unchanged to every layer call so
        the neighbor search runs once for the whole forward pass instead of
        once per refinement step.
        """
        initial_state = E3State(f=state.f, x=state.x)
        state = initial_state

        states: list[E3State] = [state]

        for _ in range(self.n_refinement_steps):
            f_new, dx = self.refinement_layer( xA, xB, s, state, fixed_neighbors=fixed_neighbors )

            # Endpoint preservation. The factor 4 makes the maximum multiplier 1,
            # since max_s s(1-s) = 0.25.
            endpoint_mask = 4.0 * s[:, None] * (1.0 - s[:, None])
            x_new = state.x + endpoint_mask * dx

            state = E3State(f=f_new, x=x_new)
            states.append(state)

        return state, states

    def forward( self,  xA: Molecule, xB: Molecule, s: pt.Tensor ) -> tuple[Molecule, list[E3State]]:
        """
        Run K refinement steps starting from the endpoint-initialised state.

        Neighbor graph handling:
            - If `fixed_neighbors` is explicitly passed, use it (override path,
              useful for tests/debug).
            - Otherwise, if `self.use_fixed_neighbors` is True (default),
              compute the endpoint-union graph internally once and reuse it
              across every refinement layer.
            - Otherwise, each layer rebuilds its own current-x distance graph.
        """
        assert xA.Z.shape == xB.Z.shape
        assert pt.equal(xA.Z, xB.Z), "`xA` and `xB` must have the same atoms in the same ordering."
        assert xA.x.shape == xB.x.shape
        assert xA.x.device == xB.x.device
        assert xA.x.dtype == xB.x.dtype

        s = s.flatten().to(device=xA.x.device, dtype=xA.x.dtype)

        assert s.numel() == len(xA.Z), "`s` must have the same number of elements as the number of atoms."

        # Decide which neighbor graph to use. Explicit kwarg wins; otherwise
        # the flag decides whether to precompute the endpoint-union graph.
        if self.use_fixed_neighbors:
            fixed_neighbors = findFixedUnionNeighbors( xA, xB, self.d_cutoff )
        else:
            fixed_neighbors = None

        # Initialize node state from endpoints and arclength.
        state = self.initialize_state(xA, xB, s)

        # Repeatedly refine using the same learned update operator.
        state, states = self.refine_state( xA, xB, s, state, fixed_neighbors=fixed_neighbors )

        # Put in the molecule framework and return.
        x_molecule = xA.copyWithNewPositions(state.x)

        return x_molecule, states