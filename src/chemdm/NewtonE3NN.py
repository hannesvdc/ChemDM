import torch as pt
import torch.nn as nn

from e3nn import o3

from chemdm.AtomicOnlyInformation import AtomicInformation
from chemdm.MLP import MultiLayerPerceptron
from chemdm.MolecularEmbeddingNetwork import MolecularEmbeddingGNN
from chemdm.embedding import ArcLengthEmbedding
from chemdm.MoleculeGraph import Molecule
from chemdm.NewtonE3NNLayer import NewtonE3NNLayer, E3State


class NewtonE3NN(nn.Module):
    """
    Transition path network using:
      - scalar/invariant endpoint encoders for xA and xB
      - one shared e3nn-based refinement operator
      - repeated learned relaxation of the path coordinates

    Interpretation:
        x^(0) = linear interpolation between xA and xB
        x^(k+1) = learned refinement of x^(k)

    The same TransitionPathE3NNLayer is reused at every refinement step.
    This makes the model closer to a learned iterative solver than to a
    conventional fixed-depth GNN.
    """

    def __init__(
        self,
        xA_embedding_network: MolecularEmbeddingGNN,
        xB_embedding_network: MolecularEmbeddingGNN,
        irreps_node_str: str = "64x0e + 16x1o + 8x1e",
        n_refinement_steps: int = 7,
        d_cutoff: float = 5.0,
        n_freq: int = 8,
        n_rbf: int = 10,
        reinitialize_features_each_step: bool = False,
    ) -> None:
        super().__init__()

        self.xA_embedding_network = xA_embedding_network
        self.xB_embedding_network = xB_embedding_network

        self.irreps_node = o3.Irreps(irreps_node_str)

        self.n_refinement_steps = n_refinement_steps
        self.d_cutoff = d_cutoff
        self.n_freq = n_freq

        self.reinitialize_features_each_step = reinitialize_features_each_step

        # Atomic scalar information
        self.atom_information = AtomicInformation()
        self.atomic_information_outputs = 64
        info_neurons_per_layer = [
            self.atom_information.numberOfOutputs(),
            64,
            self.atomic_information_outputs,
        ]
        self.atom_info_embedding = MultiLayerPerceptron(
            info_neurons_per_layer,
            nn.GELU,
            "embedding_atomic_info",
        )

        # Arclength embedding
        self.arclength_embedding = ArcLengthEmbedding(self.n_freq)

        # Initial scalar feature dimension before lifting into irreps space
        self.scalar_init_dim = (
            self.atomic_information_outputs
            + self.xA_embedding_network.state_size
            + self.xB_embedding_network.state_size
            + self.arclength_embedding.getNumberOfFeatures()
        )

        # 0e input: scalar initialization features
        self.irreps_0e_out = o3.Irreps([
            (mul, ir) for mul, ir in self.irreps_node
            if ir.l == 0 and ir.p == 1
        ])
        assert self.irreps_0e_out.dim > 0, "Expected at least one 0e block in irreps_node."

        self.irreps_0e_init = o3.Irreps(f"{self.scalar_init_dim}x0e")
        self.initial_0e_lift = o3.Linear(self.irreps_0e_init, self.irreps_0e_out)

        # 1o input: three vector channels
        #   1) xB - xA
        #   2) x - xA
        #   3) x - xB
        self.irreps_1o_out = o3.Irreps([
            (mul, ir) for mul, ir in self.irreps_node
            if ir.l == 1 and ir.p == -1
        ])

        self.irreps_1o_init = o3.Irreps("3x1o")
        self.initial_1o_lift = o3.Linear(self.irreps_1o_init, self.irreps_1o_out)

        # 1e input: all zeros for now
        self.irreps_1e_out = o3.Irreps([
            (mul, ir) for mul, ir in self.irreps_node
            if ir.l == 1 and ir.p == 1
        ])

        # 2e input: all zeros for now
        self.irreps_2e_out = o3.Irreps([
            (mul, ir) for mul, ir in self.irreps_node
            if ir.l == 2 and ir.p == 1
        ])

        # Shared e3nn refinement layer.
        #
        # This replaces the previous ModuleList of distinct layers.
        self.refinement_layer = NewtonE3NNLayer(
            irreps_node_str=irreps_node_str,
            d_cutoff=self.d_cutoff,
            n_rbf=n_rbf,
        )

    def initialize_state( self,
                          xA: Molecule,
                          xB: Molecule,
                          s: pt.Tensor,
                        ) -> E3State:
        """
        Initialize:
          - scalar node features from atom info + endpoint embeddings + arclength
          - coordinates as the linear interpolation path
          - irreps features from scalar and vector initial features
        """
        N = len(xA.Z)

        # Calculate all embedding information.
        Z_info = self.atom_information(xA)                 # (N, info_dim)
        atom_embedding = self.atom_info_embedding(Z_info)  # (N, c_atom)

        hA = self.xA_embedding_network(xA)                 # (N, c_A)
        hB = self.xB_embedding_network(xB)                 # (N, c_B)

        s_embed = self.arclength_embedding(s)              # (N, c_s) or (c_s,)
        if s_embed.ndim == 1:
            s_embed = s_embed[None, :].expand(N, -1)

        # Linear-path coordinate initialization.
        x = (1.0 - s[:, None]) * xA.x + s[:, None] * xB.x  # (N, 3)

        # Scalar initial features.
        scalar_init = pt.cat((atom_embedding, hA, hB, s_embed), dim=1)
        f_0e = self.initial_0e_lift(scalar_init)

        # Vector initial features: 3 x 1o.
        v1 = xB.x - xA.x
        v2 = x - xA.x
        v3 = x - xB.x

        vector_init = pt.stack((v1, v2, v3), dim=1)  # (N, 3, 3)
        vector_init = vector_init.reshape(N, -1)    # (N, 9), interpreted as 3x1o
        f_1o = self.initial_1o_lift(vector_init)

        # Higher/parity vector features: zero-initialized for now.
        f_1e = pt.zeros( N, self.irreps_1e_out.dim, device=x.device, dtype=x.dtype, )
        f_2e = pt.zeros( N, self.irreps_2e_out.dim, device=x.device, dtype=x.dtype, )

        # This assumes irreps_node is ordered as:
        #   0e blocks, then 1o blocks, then 1e blocks, then 2e blocks.
        #
        # This is true for the default:
        #   "64x0e + 16x1o + 8x1e"
        #
        # If you later change the order of irreps_node, this concatenation should
        # be replaced by an explicit irreps-aware assembly.
        f = pt.cat((f_0e, f_1o, f_1e, f_2e), dim=1)

        assert f.shape == (N, self.irreps_node.dim)

        return E3State(f=f, x=x)

    def refine_state( self,
                      xA: Molecule,
                      xB: Molecule,
                      s: pt.Tensor,
                      state: E3State,
                    ) -> tuple[E3State, list[E3State]]:
        """
        Apply the shared refinement layer multiple times.

        During training, keeping return_all_states=True is useful because it lets
        you put losses on intermediate refinements.
        """
        initial_state = E3State( f=state.f, x=state.x )
        state = initial_state
        states: list[E3State] = [ state ]

        for _ in range(self.n_refinement_steps):
            f_new, dx = self.refinement_layer(xA, xB, s, state)
            x_new = state.x + 4.0 * s[:, None] * (1.0 - s[:, None]) * dx # Ensure end-point preservation. s(1-s) max. value is 0.25
           
            if self.reinitialize_features_each_step:
                f_next = initial_state.f
            else:
                f_next = f_new

            state = E3State( f=f_next, x=x_new )
            states.append(state)

        return state, states

    def forward( self,
                 xA: Molecule,
                 xB: Molecule,
                 s: pt.Tensor,
                ) -> tuple[Molecule, list[E3State]]:
        assert pt.all(xA.Z == xB.Z), "`xA` and `xB` must have the same atoms in the same ordering."
        s = s.flatten()
        assert s.numel() == len(xA.Z), "`s` must have the same number of elements as the number of atoms."

        # Initialize state from the linear path.
        state = self.initialize_state( xA, xB, s )

        # Repeatedly refine using the same learned update operator.
        state, states = self.refine_state( xA, xB, s, state )

        # Put in the molecule framework and return. CoM is taken care of.
        x_molecule = xA.copyWithNewPositions( state.x )

        return x_molecule, states