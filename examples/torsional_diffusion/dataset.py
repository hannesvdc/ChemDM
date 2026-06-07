"""
TorsionalDataset + collate_torsional for torsional-diffusion training.

The dataset is backed by the preparsed .pt files produced by
`preparse_qm9.py`. Each training example is one (molecule, conformer) pair:
sampling is therefore uniform over conformers, while the split itself was
done at the molecule level so train / val / test molecules are disjoint.

The collator concatenates per-example tensors into flat batched tensors with
PyG-style batch indices (`atom_batch`, `bond_batch`), and offsets atom-level
and bond-level index tensors (`bonds`, `side_atom_idx`, `side_bond_idx`) so
they remain globally valid. The result is the exact layout the score
network's forward takes (plus the side-mask COO arrays the torsion-update
operator will use).

Loading strategy
----------------
The preparsed .pt fits comfortably in RAM (~450 MB for train), so we just
load it fully on construction and `__getitem__` is pure-memory indexing.
If we ever need to stream, the same layout supports memory-mapping with a
small adapter (each `x_list` entry is independent).

`torch.load` is invoked with `weights_only=False` because the preparsed file
is a dict containing Python lists, which the safe loader rejects.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import torch as pt
from torch.utils.data import Dataset


class TorsionalDataset( Dataset ):
    """
    One example = one (molecule, conformer) pair from a single preparsed split.

    Returned by __getitem__(idx):

        Z              (N,)       long      atomic numbers
        x              (N, 3)     float32   conformer positions (Å)
        bonds          (m, 2)     long      rotatable bond endpoints, b < c
        side_atom_idx  (P,)       long      atoms on the c-side of each bond
        side_bond_idx  (P,)       long      which bond (0..m-1) each side-atom belongs to

    Topology fields (Z, bonds, side_*) are shared across all conformers of a
    given molecule — `__getitem__` looks them up by mol_id[idx] rather than
    storing one copy per conformer.
    """

    def __init__( self, split_pt: Path ):
        data = pt.load( split_pt, weights_only=False )
        self.mol_smiles = data["mol_smiles"]
        self.mol_Z = data["mol_Z"]
        self.mol_bonds = data["mol_bonds"]
        self.mol_side_atom = data["mol_side_atom"]
        self.mol_side_bond = data["mol_side_bond"]
        self.x_flat = data["x_flat"]    # (sum_c N_c, 3)  float32
        self.x_offset = data["x_offset"]  # (C+1,)  long
        self.mol_id = data["mol_id"]    # (C,) long

    def __len__( self ) -> int:
        return int( self.mol_id.numel() )

    def __getitem__( self, idx: int ) -> dict:
        m = int( self.mol_id[idx] )
        s = int( self.x_offset[idx] )
        e = int( self.x_offset[idx+1] )
        return {
            "Z":             self.mol_Z[m],
            "x":             self.x_flat[s:e],     # view into x_flat, no copy
            "bonds":         self.mol_bonds[m],
            "side_atom_idx": self.mol_side_atom[m],
            "side_bond_idx": self.mol_side_bond[m],
        }


def collate_torsional( batch: List[dict] ) -> dict:
    """
    Pack a list of per-example dicts into flat batched tensors.

    Layout follows the score network's forward signature plus the COO side-mask
    pair used by `apply_torsion_update`. All index tensors are globally valid:
    `bonds` and `side_atom_idx` use atom indices in the flat (N_total,) layout;
    `side_bond_idx` uses bond indices in the flat (m_total,) layout.

    Returns a dict with:

        Z              (N_total,)        long
        x              (N_total, 3)      float32
        bonds          (m_total, 2)      long      atom indices, globally offset
        side_atom_idx  (P_total,)        long      atom indices, globally offset
        side_bond_idx  (P_total,)        long      bond indices, globally offset
        atom_batch     (N_total,)        long      molecule index per atom (0..B-1)
        bond_batch     (m_total,)        long      molecule index per bond  (0..B-1)
    """
    Z_chunks:     list[pt.Tensor] = []
    x_chunks:     list[pt.Tensor] = []
    bonds_chunks: list[pt.Tensor] = []
    sa_chunks:    list[pt.Tensor] = []
    sb_chunks:    list[pt.Tensor] = []
    atom_batch:   list[pt.Tensor] = []
    bond_batch:   list[pt.Tensor] = []

    atom_offset = 0
    bond_offset = 0
    for i, ex in enumerate( batch ):
        N = int( ex["Z"].numel() )
        m = int( ex["bonds"].shape[0] )

        Z_chunks.append( ex["Z"] )
        x_chunks.append( ex["x"] )
        bonds_chunks.append( ex["bonds"]         + atom_offset )
        sa_chunks.append( ex["side_atom_idx"] + atom_offset )
        sb_chunks.append( ex["side_bond_idx"] + bond_offset )
        atom_batch.append( pt.full( (N,), i, dtype=pt.long ) )
        bond_batch.append( pt.full( (m,), i, dtype=pt.long ) )

        atom_offset += N
        bond_offset += m

    return {
        "Z": pt.cat( Z_chunks ),
        "x": pt.cat( x_chunks ),
        "bonds": pt.cat( bonds_chunks ),
        "side_atom_idx": pt.cat( sa_chunks ),
        "side_bond_idx": pt.cat( sb_chunks ),
        "atom_batch": pt.cat( atom_batch ),
        "bond_batch": pt.cat( bond_batch ),
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

    print( len(data1["Z"]), len(data2["Z"]) )
    print( len(collated["Z"]) )
    print( collated )