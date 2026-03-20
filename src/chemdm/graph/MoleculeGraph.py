import torch as pt

from typing import List

class MoleculeGraph:
    Z : pt.Tensor
    x : pt.Tensor
    state : pt.Tensor
    edge_index : pt.Tensor

    def __init__( self, Z : pt.Tensor,
                        x : pt.Tensor,
                        bonds : pt.Tensor  ) -> None:
        Z = Z.flatten()
        bonds = bonds.to( dtype=pt.long )
        assert bonds.ndim == 2 and bonds.shape[1] == 2, f"`bonds` must be an integer tensor with shape `(n_bonds,2)` but got {bonds.shape}"
        assert bonds.numel() == 0 or (pt.min(bonds) >= 0 and pt.max(bonds) < len(Z)), f"Bond indices must be nonnegative and cannot exceed `len(Z)`."
        assert x.ndim == 2 and x.shape[1] == 3, f"`x` must have shape (N_atoms, 3) but got {x.shape}"
        self.Z = Z
        self.x = x
        self.edge_index = bonds

class BatchedMoleculeGraph:
    def __init__(self, molecules: List[MoleculeGraph]) -> None:
        # Concatenate atomic state and positions.
        self.Z = pt.cat([mol.Z for mol in molecules], dim=0)
        self.x = pt.cat([mol.x for mol in molecules], dim=0)

        # Calculate the total offset per molecule.
        n_atoms = pt.tensor([mol.Z.shape[0] for mol in molecules], dtype=pt.long)
        self.molecule_id = pt.repeat_interleave( pt.arange(len(molecules), dtype=pt.long), n_atoms )
        offsets = pt.cumsum( pt.cat([pt.tensor([0], dtype=pt.long), n_atoms[:-1]]), dim=0 )

        # Merge the edge indices.
        edge_list = [ mol.edge_index + offset for mol, offset in zip(molecules, offsets) ]
        self.edge_index = pt.cat(edge_list, dim=0)   # shape: (total_edges, 2)

def batchMolecules(molecules: List[MoleculeGraph]) -> BatchedMoleculeGraph:
    return BatchedMoleculeGraph(molecules)