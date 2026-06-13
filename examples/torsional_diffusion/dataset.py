"""
TorsionalDataset + collate_torsional for torsional-diffusion training.

The dataset is backed by the preparsed .pt files produced by
`preparse_qm9.py`. Each training example is one (molecule, conformer) pair:
sampling is therefore uniform over conformers, while the split itself was
done at the molecule level so train / val / test molecules are disjoint.

Per-example layout uses chemdm's `MoleculeGraph` for the atoms-and-positions
part. The collator stacks them via `BatchedMoleculeGraph`, which carries a
`molecule_id` per atom — that automatically plays the role of `atom_batch`,
and `chemdm.MoleculeGraph.findAllDistanceNeighbors` uses it natively to keep
the radius graph within molecule boundaries.

Rotatable-bond / side-mask metadata is concatenated into flat tensors with
the same offset trick as before. Those aren't part of the chemistry graph,
so they live alongside the molecule rather than inside it.

Loading strategy
----------------
The preparsed .pt fits comfortably in RAM (~450 MB for train), so we just
load it fully on construction and `__getitem__` is pure-memory indexing.

`torch.load` is invoked with `weights_only=False` because the preparsed file
is a dict containing Python lists, which the safe loader rejects.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import torch as pt
from torch.utils.data import Dataset

from chemdm.MoleculeGraph import MoleculeGraph, BatchedMoleculeGraph, batchMolecules


class TorsionalDataset( Dataset ):
    """
    One example = one (molecule, conformer) pair from a single preparsed split.

    Returned by __getitem__(idx):

        mol              MoleculeGraph    Z, conformer positions, empty chemistry bonds
        rotatable_bonds  (m, 2)  long     rotatable bond endpoints, b < c, atom-local indices
        side_atom_idx    (P,)    long     atoms on the c-side of each bond, atom-local indices
        side_bond_idx    (P,)    long     which bond (0..m-1) each side-atom belongs to

    Topology fields (Z, rotatable bonds, side_*) are shared across all conformers
    of a given molecule — `__getitem__` looks them up by `mol_id[idx]` rather than
    storing one copy per conformer.
    """

    def __init__( self, split_pt: Path ):
        data = pt.load( split_pt, weights_only=False )
        self.mol_smiles          = data["mol_smiles"]
        self.mol_Z               = data["mol_Z"]
        self.mol_edge_index      = data["mol_edge_index"]   # covalent bonds per mol, (E_i, 2)
        self.mol_rotatable_bonds = data["mol_rotatable_bonds"]
        self.mol_side_atom       = data["mol_side_atom"]
        self.mol_side_bond       = data["mol_side_bond"]
        self.x_flat         = data["x_flat"]    # (sum_c N_c, 3)  float32
        self.x_offset       = data["x_offset"]  # (C+1,)          long
        self.mol_id         = data["mol_id"]    # (C,)            long

    def __len__( self ) -> int:
        return int( self.mol_id.numel() )

    def __getitem__( self, idx: int ) -> dict:
        m = int( self.mol_id[idx] )
        s = int( self.x_offset[idx] )
        e = int( self.x_offset[idx + 1] )
        return {
            "mol":             MoleculeGraph(
                                    Z=self.mol_Z[m],
                                    x=self.x_flat[s:e],   # view into x_flat, no copy
                                    bonds=self.mol_edge_index[m],
                                ),
            "rotatable_bonds": self.mol_rotatable_bonds[m],
            "side_atom_idx":   self.mol_side_atom[m],
            "side_bond_idx":   self.mol_side_bond[m],
        }


def collate_torsional( batch: List[dict] ) -> dict:
    """
    Pack a list of per-example dicts into a BatchedMoleculeGraph plus the
    rotatable-bond + side-mask metadata, with atom and bond offsets applied so
    every index is globally valid.

    Returns a dict with:

        mol              BatchedMoleculeGraph     Z (N_total,), x (N_total, 3), molecule_id (N_total,)
        rotatable_bonds  (m_total, 2)   long      atom indices, globally offset
        side_atom_idx    (P_total,)     long      atom indices, globally offset
        side_bond_idx    (P_total,)     long      bond indices, globally offset
        bond_batch       (m_total,)     long      molecule index per bond  (0..B-1)

    `mol.molecule_id` serves as the per-atom batch index — no separate
    `atom_batch` tensor needed.
    """
    rot_chunks: list[pt.Tensor] = []
    sa_chunks:  list[pt.Tensor] = []
    sb_chunks:  list[pt.Tensor] = []
    bond_batch: list[pt.Tensor] = []

    atom_offset = 0
    bond_offset = 0
    for i, ex in enumerate( batch ):
        N = int( ex["mol"].Z.numel() )
        m = int( ex["rotatable_bonds"].shape[0] )

        rot_chunks.append( ex["rotatable_bonds"] + atom_offset )
        sa_chunks.append ( ex["side_atom_idx"]   + atom_offset )
        sb_chunks.append ( ex["side_bond_idx"]   + bond_offset )
        bond_batch.append( pt.full( (m,), i, dtype=pt.long ) )

        atom_offset += N
        bond_offset += m

    batched_mol = batchMolecules( [ ex["mol"] for ex in batch ] )

    return {
        "mol":             batched_mol,
        "rotatable_bonds": pt.cat( rot_chunks ),
        "side_atom_idx":   pt.cat( sa_chunks ),
        "side_bond_idx":   pt.cat( sb_chunks ),
        "bond_batch":      pt.cat( bond_batch ),
    }


if __name__ == '__main__':
    """ Simple testing routine. """
    from dotenv import load_dotenv
    load_dotenv()
    import os
    qm9_folder = Path( os.environ["QM9_FOLDER"] )

    data_folder = qm9_folder.parent / "parsed" / "train.pt"
    ds = TorsionalDataset( data_folder )
    data1 = ds[1001]
    data2 = ds[2]
    collated = collate_torsional( [data1, data2] )

    print( "ex1 N:", int(data1["mol"].Z.numel()), "  ex2 N:", int(data2["mol"].Z.numel()) )
    print( "batch N_total:", int(collated["mol"].Z.numel()) )
    print( "batch m_total:", int(collated["rotatable_bonds"].shape[0]) )
    print( "molecule_id unique:", pt.unique(collated["mol"].molecule_id).tolist() )
