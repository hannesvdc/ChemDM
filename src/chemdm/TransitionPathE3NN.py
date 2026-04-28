import torch as pt
import torch.nn as nn

from e3nn import o3

from chemdm.AtomicOnlyInformation import AtomicInformation
from chemdm.MLP import MultiLayerPerceptron
from chemdm.MolecularEmbeddingNetwork import MolecularEmbeddingGNN
from chemdm.embedding import ArcLengthEmbedding
from chemdm.MoleculeGraph import Molecule, recenterMolecule
from chemdm.TransitionPathE3NNLayer import TransitionPathE3NNLayer, E3State

class TransitionPathE3NN(nn.Module):
    """
    Transition path network using:
      - scalar/invariant endpoint encoders for xA and xB
      - an e3nn-based equivariant path network for the actual path evolution

    The e3nn layers update:
      - node irreps features f
      - coordinates x

    This archictecture is more expressive than TransitionPathGNN (hopefully). 
    """

    def __init__(
        self,
        xA_embedding_network: MolecularEmbeddingGNN,
        xB_embedding_network: MolecularEmbeddingGNN,
        irreps_node_str: str = "64x0e + 16x1o + 8x1e",
        n_layers: int = 4,
        d_cutoff: float = 5.0,
        n_freq: int = 8,
        n_rbf: int = 10,
    ) -> None:
        super().__init__()

        self.xA_embedding_network = xA_embedding_network
        self.xB_embedding_network = xB_embedding_network
        self.irreps_node = o3.Irreps(irreps_node_str)
        self.n_layers = n_layers
        self.d_cutoff = d_cutoff
        self.n_freq = n_freq

        # Atomic scalar information
        self.atom_information = AtomicInformation()
        self.atomic_information_outputs = 64
        info_neurons_per_layer = [ self.atom_information.numberOfOutputs(), 64, self.atomic_information_outputs, ]
        self.atom_info_embedding = MultiLayerPerceptron( info_neurons_per_layer, nn.GELU, "embedding_atomic_info", )

        # Arclength embedding
        self.arclength_embedding = ArcLengthEmbedding(self.n_freq)

        # Initial scalar feature dimension before lifting into irreps space
        self.scalar_init_dim = (
            self.atomic_information_outputs
            + self.xA_embedding_network.state_size
            + self.xB_embedding_network.state_size
            + self.arclength_embedding.getNumberOfFeatures() )

        # 0e input: scalar initialization features
        self.irreps_0e_out = o3.Irreps([
            (mul, ir) for mul, ir in self.irreps_node
            if ir.l == 0 and ir.p == 1
        ])
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

        # 1e input: all zeros
        self.irreps_1e_out = o3.Irreps([
            (mul, ir) for mul, ir in self.irreps_node
            if ir.l == 1 and ir.p == 1
        ])

        # 2e input: all zeros
        self.irreps_2e_out = o3.Irreps([
            (mul, ir) for mul, ir in self.irreps_node
            if ir.l == 2 and ir.p == 1
        ])

        # e3nn layers
        self.layers = nn.ModuleList( [  
            TransitionPathE3NNLayer(
                    irreps_node_str=irreps_node_str,
                    d_cutoff=self.d_cutoff,
                    n_rbf=n_rbf,
                ) for _ in range(self.n_layers) ]  )

    def initialize_state( self, xA: Molecule, xB: Molecule, s: pt.Tensor ) -> E3State:
        """
        Initialize:
          - scalar node features from atom info + endpoint embeddings + arclength
          - coordinates as the linear interpolation path
          - then lift scalar features into irreps node features
        """
        N = len( xA.Z )

        # Calculate all embedding information
        Z_info = self.atom_information(xA)                # (N, info_dim)
        atom_embedding = self.atom_info_embedding(Z_info) # (N, c_atom)

        hA = self.xA_embedding_network(xA)                # (N, c_A)
        hB = self.xB_embedding_network(xB)                # (N, c_B)

        s_embed = self.arclength_embedding(s)             # (N, c_s)
        if s_embed.ndim == 1:
            s_embed = s_embed[None, :]

        # Linear-path coordinate initialization
        x = (1.0 - s[:, None]) * xA.x + s[:, None] * xB.x  # (N, 3)

        # Initialize the irreps from scalar and vector features
        scalar_init = pt.cat((atom_embedding, hA, hB, s_embed), dim=1)  # (N, scalar_init_dim)
        f_0e = self.initial_0e_lift( scalar_init )

        # Vector initial features (1o)
        v1 = xB.x - xA.x   # (N, 3)
        v2 = x - xA.x      # (N, 3)
        v3 = x - xB.x      # (N, 3)
        vector_init = pt.stack((v1, v2, v3), dim=1)   # (N, 3, 3)
        vector_init = vector_init.reshape(N, -1)       # (N, 9), interpreted as 3x1o
        f_1o = self.initial_1o_lift(vector_init)   # (N, irreps_1o_out.dim)

        # Vector initial features (1e). All zeros for now.
        # Ideally, this is where we include torsion information between bonds.
        f_1e = pt.zeros( N, self.irreps_1e_out.dim, device=x.device, dtype=x.dtype, )

        f_2e = pt.zeros(N, self.irreps_2e_out.dim, device=x.device, dtype=x.dtype)
        
        # Concatenate in the same order as irreps_node = " ... 0e + ... 1o + ... 1e + ... 2e"
        f = pt.cat((f_0e, f_1o, f_1e, f_2e), dim=1)
        return E3State(f=f, x=x)

    def forward( self, xA: Molecule, xB: Molecule, s: pt.Tensor ) -> Molecule:
        assert pt.all(xA.Z == xB.Z), "`xA` and `xB` must have the same atoms in the same ordering."
        s = s.flatten()
        assert s.numel() == len(xA.Z), "`s` must have the same number of elements as the number of atoms."

        # Initialize state
        state = self.initialize_state(xA, xB, s)

        # Evolve path state through e3nn layers
        for layer in self.layers:
            state = layer(xA, xB, s, state)

        # Final Dirichlet-style correction over the linear path
        base = (1.0 - s[:, None]) * xA.x + s[:, None] * xB.x
        correction = state.x - base
        x_final = base + s[:, None] * (1.0 - s[:, None]) * correction

        # Put in the molecule framework
        x_molecule = xA.copyWithNewPositions(x_final)
        return x_molecule