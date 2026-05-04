import torch as pt
import torch.nn as nn

from e3nn import o3

from chemdm.AtomicOnlyInformation import AtomicInformation
from chemdm.MLP import MultiLayerPerceptron
from chemdm.MolecularEmbeddingNetwork import MolecularEmbeddingGNN
from chemdm.MoleculeGraph import Molecule
from chemdm.NewtonE3NNLayer import NewtonE3NNLayer, E3State
from chemdm.embedding import SinusoidalEmbedding

class ResidualDiffusionE3NN(nn.Module):
    """
    Conditional residual diffusion model around a deterministic Newton path.
    
    The Newton path is treated as the base path:

        x_base = x_Newton

    The diffusion variable c_t lives in residual space:

        x_noisy = x_base + gate(s) * residual_scale * c_t

    where:

        gate(s) = 4 s (1 - s)

    The model predicts the Gaussian noise eps added to the clean residual.
    """

    def __init__(
        self,
        xA_embedding_network: MolecularEmbeddingGNN,
        xB_embedding_network: MolecularEmbeddingGNN,
        irreps_node_str: str = "48x0e + 16x1o + 16x1e + 8x2e",
        n_denoising_steps: int = 3,
        d_cutoff: float = 5.0,
        n_arclength_freq: int = 8,
        n_rbf: int = 10,
        n_time_freq: int = 8,
        time_embedding_outputs: int = 64,
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__()

        self.xA_embedding_network = xA_embedding_network
        self.xB_embedding_network = xB_embedding_network

        self.irreps_node = o3.Irreps(irreps_node_str)

        self.n_denoising_steps = n_denoising_steps
        self.d_cutoff = d_cutoff
        self.n_arclength_freq = n_arclength_freq
        self.n_rbf = n_rbf
        self.residual_scale = residual_scale

        # Atomic scalar information.
        self.atom_information = AtomicInformation()
        self.atomic_information_outputs = 64

        self.atom_info_embedding = MultiLayerPerceptron(
            [ self.atom_information.numberOfOutputs(), 64, self.atomic_information_outputs ],
            nn.GELU,
            "diffusion_embedding_atomic_info",
        )

        # Arclength embedding.
        self.arclength_embedding = SinusoidalEmbedding( self.n_arclength_freq, include_endpoints=True )

        # Diffusion-time embedding.
        self.time_embedding = SinusoidalEmbedding( n_time_freq, include_endpoints=True )
        self.time_embedding_mlp = MultiLayerPerceptron( 
            [ self.time_embedding.n_embeddings, 128, time_embedding_outputs ],
            nn.GELU,
            "diffusion_time_embedding",
        )

        # Scalar initialization features.
        self.scalar_init_dim = (
            self.atomic_information_outputs
            + self.xA_embedding_network.state_size
            + self.xB_embedding_network.state_size
            + self.arclength_embedding.getNumberOfFeatures()
            + time_embedding_outputs
        )

        self.irreps_0e_out = o3.Irreps([
            (mul, ir) for mul, ir in self.irreps_node
            if ir.l == 0 and ir.p == 1
        ])

        self.irreps_0e_init = o3.Irreps(f"{self.scalar_init_dim}x0e")
        self.initial_0e_lift = o3.Linear(
            self.irreps_0e_init,
            self.irreps_0e_out,
        )

        # Vector initialization features.
        #
        # 1) xB - xA
        # 2) x_noisy - xA
        # 3) x_noisy - xB
        # 4) x_noisy - x_base
        # 5) residual_scale * c_t
        #
        # All are polar vectors, so 1o.
        self.irreps_1o_out = o3.Irreps([
            (mul, ir) for mul, ir in self.irreps_node
            if ir.l == 1 and ir.p == -1
        ])

        self.irreps_1o_init = o3.Irreps("5x1o")
        self.initial_1o_lift = o3.Linear(
            self.irreps_1o_init,
            self.irreps_1o_out,
        )

        # Optional higher/parity vector features, zero-initialized.
        self.irreps_1e_out = o3.Irreps([
            (mul, ir) for mul, ir in self.irreps_node
            if ir.l == 1 and ir.p == 1
        ])
        self.irreps_2e_out = o3.Irreps([
            (mul, ir) for mul, ir in self.irreps_node
            if ir.l == 2 and ir.p == 1
        ])

        # Denoising layers.
        #
        # v1: use one shared NewtonE3NNLayer repeatedly.
        # This mirrors the Newton model and keeps the parameter count down.
        self.denoising_layer = NewtonE3NNLayer(
            irreps_node_str=irreps_node_str,
            d_cutoff=d_cutoff,
            n_rbf=n_rbf,
        )

    def initialize_state( self, xA: Molecule,
                                xB: Molecule,
                                s: pt.Tensor,
                                x_base: pt.Tensor,
                                x_noisy: pt.Tensor,
                                c_t: pt.Tensor,
                                t: pt.Tensor, ) -> E3State:
        """
        Initialize the equivariant denoising state. Coordinates in the state are the noisy physical path:

            state.x = x_noisy

        The noisy residual, Newton/base path, endpoints, s, and t all enter
        through scalar/vector features.
        """
        N = len(xA.Z)

        assert x_base.shape == (N, 3)
        assert x_noisy.shape == (N, 3)
        assert c_t.shape == (N, 3)
        assert pt.all((0.0 <= t) & (t <= 1.0)), "`t` should be normalized to [0,1]."

        device = x_noisy.device
        dtype = x_noisy.dtype

        # Scalar molecular/atomic context.
        Z_info = self.atom_information(xA)
        atom_embedding = self.atom_info_embedding(Z_info)

        hA = self.xA_embedding_network(xA)
        hB = self.xB_embedding_network(xB)

        s_embed = self.arclength_embedding(s)
        if s_embed.ndim == 1:
            s_embed = s_embed[None, :].expand(N, -1)

        t_embed = self.time_embedding( t )
        t_embed = self.time_embedding_mlp( t_embed )

        scalar_init = pt.cat( ( atom_embedding, hA, hB, s_embed, t_embed, ), dim=1, )
        f_0e = self.initial_0e_lift(scalar_init)

        # Vector features.
        v1 = xB.x - xA.x
        v2 = x_noisy - xA.x
        v3 = x_noisy - xB.x
        v4 = x_noisy - x_base
        v5 = self.residual_scale * c_t

        vector_init = pt.stack((v1, v2, v3, v4, v5), dim=1)  # (N, 5, 3)
        vector_init = vector_init.reshape(N, -1)             # (N, 15)

        f_1o = self.initial_1o_lift(vector_init)

        f_1e = pt.zeros(
            N,
            self.irreps_1e_out.dim,
            device=device,
            dtype=dtype,
        )

        f_2e = pt.zeros(
            N,
            self.irreps_2e_out.dim,
            device=device,
            dtype=dtype,
        )

        # Assumes irreps_node order:
        #   0e, 1o, 1e, 2e
        f = pt.cat((f_0e, f_1o, f_1e, f_2e), dim=1)

        assert f.shape == (N, self.irreps_node.dim)

        return E3State(f=f, x=x_noisy)

    def forward( self, xA: Molecule,
                       xB: Molecule,
                       s: pt.Tensor,
                       x_base: Molecule,
                       c_t: pt.Tensor,
                       t: pt.Tensor, ) -> pt.Tensor:
        """
        Predict diffusion noise eps from noisy residual c_t.

        Arguments
        ---------
        xA, xB:
            Endpoint molecule/path batches.
        s:
            Path coordinate per atom, shape (N,).
        x_base:
            Newton path coordinates, either tensor shape (N, 3)
            or Molecule/BatchedMoleculeGraph with field .x.
        c_t:
            Noisy residual latent, shape (N, 3).
        t:
            Diffusion time. Must be shape (N,)

        Returns
        -------
        eps_pred:
            Predicted Gaussian noise, shape (N, 3).
        """
        assert pt.all(xA.Z == xB.Z), "`xA` and `xB` must have the same atoms in the same ordering."
        s = s.flatten()
        N = len(xA.Z)

        assert s.numel() == N
        assert c_t.shape == (N, 3)

        x_base_tensor = x_base.x

        # Endpoint gate. Vanishes at s=0 and s=1; max value is 1 at s=0.5.
        gate = 4.0 * s[:, None] * (1.0 - s[:, None])

        # Physical noisy path corresponding to residual latent c_t.
        x_noisy = x_base_tensor + gate * self.residual_scale * c_t
        state = self.initialize_state( xA=xA, xB=xB, s=s, x_base=x_base_tensor, x_noisy=x_noisy, c_t=c_t, t=t )

        eps_pred = pt.empty_like( state.x )
        for k in range(self.n_denoising_steps):
            f_new, dx = self.denoising_layer(xA, xB, s, state)

            eps_pred = dx
            x_next = state.x
            state = E3State(f=f_new, x=x_next)

        # Put in the molecule framework and return. CoM is taken care of.
        return eps_pred