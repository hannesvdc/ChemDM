from __future__ import annotations

from typing import Optional

import torch as pt
import torch.nn as nn

from chemdm.MoleculeGraph import Molecule
from chemdm.MoleculeInformation import (
    computeMoleculeInformation,
    DEFAULT_ATOMIC_NUMBERS,
    DEFAULT_RING_SIZES,
)
from chemdm.MLP import MultiLayerPerceptron



class TPMoleculeEmbedding(nn.Module):
    """
    Endpoint-conditioned scalar node embedding for the transition-path model.

    This module replaces the old scalar-node initialization pieces:

        atom_information(xA)
        atom_info_embedding(...)
        xA_embedding_network(xA)
        xB_embedding_network(xB)

    It returns one scalar embedding per atom:

        tp_embedding(xA, xB): (N, embedding_dim)
    """

    def __init__( self,
                 *,
                 embedding_dim: int,
                 hidden_dim: int = 128,
                 allowed_atomic_numbers: Optional[list[int]] = None,
                 allowed_ring_sizes: Optional[list[int]] = None,
                 include_ring_information: bool = True,
                 n_hidden_layers: int = 2,
    ) -> None:
        super().__init__()

        if allowed_atomic_numbers is None:
            allowed_atomic_numbers = DEFAULT_ATOMIC_NUMBERS

        if allowed_ring_sizes is None:
            allowed_ring_sizes = DEFAULT_RING_SIZES

        self.allowed_atomic_numbers = list(allowed_atomic_numbers)
        self.allowed_ring_sizes = list(allowed_ring_sizes)
        self.include_ring_information = include_ring_information

        self.raw_feature_dim = self._raw_feature_dim()

        embedding_layers = [ self.raw_feature_dim ] + [ hidden_dim ] * n_hidden_layers + [ embedding_dim ]
        self.embedding_network = MultiLayerPerceptron( embedding_layers, nn.SiLU )

    def _common_feature_dim( self ) -> int:
        return (
            len(self.allowed_atomic_numbers)
            + 1  # atomic_mass_scaled
            + 1  # is_hydrogen
            + 1  # is_heavy_atom
        )

    def _topology_feature_dim( self ) -> int:
        return (
            1  # degree
            + 1  # atom_in_ring
            + 1  # atom_ring_count
            + len(self.allowed_ring_sizes)
        )

    def _reaction_feature_dim( self ) -> int:
        return (
            1  # displacement_norm ||xB_i - xA_i||
            + 1  # degree_change
            + 1  # abs_degree_change
            + 1  # ring_count_change
            + 1  # abs_ring_count_change
            + 1  # local_bond_change_count
            + 1  # is_reactive_atom
            + 1  # is_reactive_hydrogen
        )

    def _raw_feature_dim(self) -> int:
        return (
            self._common_feature_dim()
            + 2 * self._topology_feature_dim()
            + self._reaction_feature_dim()
        )
    
    def build_common_features( self, info ) -> pt.Tensor:
        """ Features shared by reactants and products. """
        atoms = info.atoms
        dtype = atoms.atom_type_one_hot.dtype

        is_hydrogen = (info.atoms.Z == 1)
        is_heavy_atom = (info.atoms.Z > 1)

        return pt.cat( [
                atoms.atom_type_one_hot,
                atoms.atomic_mass_scaled[:, None].to(dtype=dtype),
                is_hydrogen[:,None].to(dtype=dtype),
                is_heavy_atom[:,None].to(dtype=dtype),
            ], dim=1 
        )

    def build_topology_features( self, info ) -> pt.Tensor:
        atoms = info.atoms
        dtype = atoms.atom_type_one_hot.dtype

        return pt.cat(
            [
                atoms.degree[:, None].to(dtype=dtype),
                atoms.atom_in_ring.to(dtype=dtype)[:, None],
                atoms.atom_ring_count.to(dtype=dtype)[:, None],
                atoms.atom_ring_size_flags.to(dtype=dtype),
            ],
            dim=1,
        )

    @pt.no_grad()
    def build_reaction_features( self,
                                 xA: Molecule,
                                 xB: Molecule,
                                 *,
                                 dtype: pt.dtype,
                                 device: pt.device,
    ) -> pt.Tensor:
        """
        Build per-atom scalar features describing graph/topology changes
        between xA and xB.

        This is intentionally no_grad because it is discrete/topological.
        """
        n_atoms = int( xA.Z.shape[0] )

        neighbors_A = neighbor_sets(n_atoms, xA.edge_index)
        neighbors_B = neighbor_sets(n_atoms, xB.edge_index)

        local_bond_change_count = pt.zeros((n_atoms, 1), dtype=dtype, device=device)
        is_reactive_atom = pt.zeros((n_atoms, 1), dtype=dtype, device=device)

        for i in range(n_atoms):
            removed = neighbors_A[i] - neighbors_B[i]
            added = neighbors_B[i] - neighbors_A[i]

            n_changed = len(removed) + len(added)

            local_bond_change_count[i, 0] = float(n_changed)
            is_reactive_atom[i, 0] = 1.0 if n_changed > 0 else 0.0

        Z = xA.Z.to(device=device).long()
        is_hydrogen = (Z == 1).to(dtype=dtype)[:, None]
        is_reactive_hydrogen = is_hydrogen * is_reactive_atom

        return pt.cat(
            [
                local_bond_change_count,
                is_reactive_atom,
                is_reactive_hydrogen,
            ],
            dim=-1,
        )

    def build_raw_features( self, xA: Molecule, xB: Molecule ) -> pt.Tensor:
        """
        Build raw scalar endpoint/change features before the learnable MLP.

        Shape:
            (N, raw_feature_dim)
        """
        assert pt.equal( xA.Z, xB.Z ), f"`xA` and `xB` must have the same atoms in the same order."
        assert xA.x.shape == xB.x.shape
        assert xA.x.device == xB.x.device, "`xA` and `xB` must be on the same device."
        assert xA.x.dtype == xB.x.dtype, "`xA` and `xB` must have the same dtype."

        device = xA.x.device
        dtype = xA.x.dtype
        n_atoms = int(xA.Z.shape[0])

        info_A = computeMoleculeInformation(
            xA,
            allowed_atomic_numbers=self.allowed_atomic_numbers,
            allowed_ring_sizes=self.allowed_ring_sizes,
            include_ring_information=self.include_ring_information,
        )

        info_B = computeMoleculeInformation(
            xB,
            allowed_atomic_numbers=self.allowed_atomic_numbers,
            allowed_ring_sizes=self.allowed_ring_sizes,
            include_ring_information=self.include_ring_information,
        )

        # Start with the common features
        common_features = self.build_common_features( info_A )

        # Then append reactant and product topology information
        base_A = self.build_topology_features( info_A )
        base_B = self.build_topology_features( info_B )

        # Some topology mixing information. Requires alignment!
        displacement = xB.x.to(device=device, dtype=dtype) - xA.x.to(device=device, dtype=dtype)
        displacement_norm = pt.linalg.norm(displacement, dim=-1, keepdim=True)

        # Finally add reaction features
        degree_change = (
            info_B.atoms.degree.to(device=device, dtype=dtype)
            - info_A.atoms.degree.to(device=device, dtype=dtype)
        )[:, None]
        abs_degree_change = pt.abs( degree_change )

        ring_count_change = (
            info_B.atoms.atom_ring_count.to(device=device, dtype=dtype)
            - info_A.atoms.atom_ring_count.to(device=device, dtype=dtype)
        )[:, None]
        abs_ring_count_change = pt.abs( ring_count_change )

        reaction_features = self.build_reaction_features( xA, xB, dtype=dtype, device=device )

        # Actually append everything
        raw = pt.cat( [
                common_features,
                base_A,
                base_B,
                displacement_norm,
                degree_change,
                abs_degree_change,
                ring_count_change,
                abs_ring_count_change,
                reaction_features,
            ],
            dim=-1,
        )

        assert raw.shape == (n_atoms, self.raw_feature_dim), (
            f"Expected raw feature shape {(n_atoms, self.raw_feature_dim)}, "
            f"got {tuple(raw.shape)}"
        )

        return raw

    def forward( self, xA: Molecule, xB: Molecule ) -> pt.Tensor:
        """
        Important: assumes the reactants and products are aligned and have zero CoM! """
        raw_features = self.build_raw_features(xA, xB)
        return self.embedding_network(raw_features)


@pt.no_grad()
def neighbor_sets( n_atoms: int, edge_index: pt.Tensor ) -> list[set[int]]:
    """
    Return undirected neighbor sets from directed or undirected edge_index.
    """
    neighbors = [set() for _ in range(n_atoms)]

    if edge_index.numel() == 0:
        return neighbors

    for i, j in edge_index.detach().cpu().long().tolist():
        if i == j:
            continue
        neighbors[i].add(j)
        neighbors[j].add(i)

    return neighbors