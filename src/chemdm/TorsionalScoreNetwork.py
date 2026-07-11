"""
TorsionalScoreNetwork — the full score model for torsional diffusion.

Composes:
    1. Atom embedding: Z -> 0e scalar features.
    2. Sinusoidal embedding of diffusion time t -> 0e scalar features added per atom.
    3. Lift 0e scalars to the trunk irreps (higher-l blocks initialise to 0;
       they will be populated by the attention layers via TPs with Y(r̂_ij)).
    4. K layers of EquivariantAttentionLayer (SE(3)-Transformer Q/K/V attention).
    5. PseudotorqueHead: per-rotatable-bond pseudoscalar (0o) readout.

The output is `δτ ∈ ℝ^m`, one real number per rotatable bond. By construction
it is SE(3)-invariant and parity-equivariant (δτ(-x) = −δτ(x)), which are the
required symmetries for the torsional score on the hypertorus.

Batching:
    The network takes flat per-atom and per-bond tensors plus `atom_batch` /
    `bond_batch` index tensors (PyG-style). It expects `edge_index` to already
    respect molecule boundaries — no cross-molecule edges. The caller is
    responsible for building the trunk neighborhood (e.g. via a radius graph
    inside the cutoff) and the rotatable-bond list.
"""

from __future__ import annotations

import math

import torch as pt
import torch.nn as nn

from e3nn import o3

from chemdm.MLP import MultiLayerPerceptron
from chemdm.embedding import SinusoidalEmbedding
from chemdm.MoleculeGraph import BatchedMoleculeGraph

from chemdm.attention import EquivariantAttentionLayer
from chemdm.pseudotorque import PseudotorqueHead


class TorsionalScoreNetwork(nn.Module):
    """
    Parameters
    ----------
    irreps_node_str:
        Trunk node-feature irreps. Default carries 0e + 0o + 1o + 1e so that
        the pseudotorque head can synthesise 0o through low-order TP paths.
    irreps_qk_str:
        Q and K irreps for the attention layers.
    irreps_v_str:
        V irreps. Defaults to irreps_node.
    n_layers:
        Number of trunk attention layers.
    d_cutoff:
        Trunk neighbor-search cutoff [Å] (used only by the trunk's RBFs).
    head_cutoff:
        Per-bond head neighborhood cutoff [Å].
    n_rbf:
        RBF count for the trunk and the head.
    n_pseudo:
        Per-atom pseudoscalar bank width in the head.
    time_n_freq:
        Number of sinusoidal frequencies in the time embedding. The resulting
        embedding width is 2*time_n_freq + 2 (the +2 = raw `t` and `1-t`).
    z_embed_dim:
        Width of the atomic-number embedding used by the head's query MLP.
        Independent of the trunk's atom embedding.
    n_elements:
        Size of the atomic-number embedding tables.
    sigma_min, sigma_max:
        Noise schedule bounds [rad]. Stored on the module for the trainer to
        reuse; not used internally.
    """

    def __init__( self,
                 irreps_node_str: str = "64x0e + 32x0o + 16x1o + 16x1e",
                 irreps_qk_str: str = "16x0e + 8x0o + 8x1o + 8x1e",
                 irreps_v_str: str | None = None,
                 n_layers: int = 4,
                 d_cutoff: float = 5.0,
                 head_cutoff: float = 5.0,
                 n_rbf: int = 16,
                 n_pseudo: int = 8,
                 time_n_freq: int = 16,
                 z_embed_dim: int = 16,
                 n_elements: int = 120,
                 sigma_min: float = 0.01 * math.pi,
                 sigma_max: float = math.pi,
    ) -> None:
        super().__init__()

        self.irreps_node = o3.Irreps(irreps_node_str)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        # 0e block size — where the atom and time embeddings live before lift.
        self.irreps_0e = o3.Irreps( [(mul, ir) for mul, ir in self.irreps_node if ir.l == 0 and ir.p == 1] )
        self.scalar_dim = self.irreps_0e.dim

        # Atom embedding: Z -> 0e features of width scalar_dim.
        # Let the network learn its own atomic embedding, perhaps more general than
        # chemdm.MoleculeInformation
        self.z_embed = nn.Embedding( n_elements, self.scalar_dim )

        # Time embedding: t ∈ [0, 1] -> (B, time_dim). include_endpoints=True
        # appends the raw `t` and `1-t` to the sin/cos bank so the network has
        # boundary-aware features at no expressivity cost.
        self.time_embed = SinusoidalEmbedding( n_freq=time_n_freq, include_endpoints=True )
        self.time_dim = self.time_embed.n_embeddings   # 2*n_freq + 2

        # Time-embedding MLP: project the raw (B, time_dim) embedding to the 0e
        # width so it can be added to atom features. The raw embedding is also
        # passed through (unprojected) to the head's query MLP.
        time_mlp_neurons = [self.time_dim, 2 * self.time_dim, self.scalar_dim]
        self.time_mlp = MultiLayerPerceptron( time_mlp_neurons, nn.GELU, "td_time_mlp" )

        # Lift 0e features to irreps_node. Equivariant o3.Linear: only 0e -> 0e
        # paths have weights, so 0o / 1o / 1e blocks start at exactly zero and
        # are populated by the trunk attention layers below.
        self.lift = o3.Linear( self.irreps_0e, self.irreps_node )

        # Trunk. Identical layer repeated.
        self.layers = nn.ModuleList([
            EquivariantAttentionLayer(
                irreps_node_str=irreps_node_str,
                irreps_qk_str=irreps_qk_str,
                irreps_v_str=irreps_v_str,
                d_cutoff=d_cutoff,
                n_rbf=n_rbf,
            )
            for _ in range(n_layers)
        ])

        # Torque bond head.
        self.head = PseudotorqueHead(
            irreps_node_str=irreps_node_str,
            n_pseudo=n_pseudo,
            head_cutoff=head_cutoff,
            n_rbf=n_rbf,
            time_dim=self.time_dim,
            z_embed_dim=z_embed_dim,
            n_elements=n_elements,
        )

    def forward( self,
                 mol: BatchedMoleculeGraph,   # carries Z (N,), x (N, 3), molecule_id (N,)
                 t: pt.Tensor,                # (B,)  diffusion time ∈ [0, 1]
                 neighbors: pt.Tensor,        # (2, E)  trunk's radius+bond graph
                 is_bond: pt.Tensor,          # (E,)    1.0 if edge is a covalent bond
                 rotatable_bonds: pt.Tensor,  # (m, 2)
                 bond_batch: pt.Tensor,       # (m,)
    ) -> pt.Tensor:
        """
        Returns
        -------
        delta_tau : (m,) — predicted score per rotatable bond.
        """
        Z = mol.Z
        x = mol.x
        atom_batch = mol.molecule_id

        # Per-molecule sinusoidal time embedding, projected to 0e width.
        time_emb = self.time_embed( t )                     # (B, time_dim)
        time_scalar = self.time_mlp( time_emb )                # (B, scalar_dim)

        # Atom 0e features: Z embedding + per-atom time scalar.
        # Add together because each is already the output of an MLP.
        # This does not reduce expressiveness.
        f0e = self.z_embed( Z ) + time_scalar[atom_batch]      # (N, scalar_dim)

        # Lift into the trunk irreps (higher-l blocks start at zero).
        f = self.lift( f0e )                                   # (N, irreps_node.dim)

        # Attention trunk. Layer signature stays flat (f, x, neighbors, is_bond)
        # — the SE(3) attention isn't molecule-specific, it's a generic graph op.
        for layer in self.layers:
            f = layer( f, x, neighbors, is_bond )

        # Pseudotorque head: per-rotatable-bond δτ_i, sign-flips under parity.
        return self.head(
            f=f,
            mol=mol,
            rotatable_bonds=rotatable_bonds,
            bond_batch=bond_batch,
            time_emb_per_mol=time_emb,
        )
