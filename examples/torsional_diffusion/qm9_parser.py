"""
Parser for the GEOM-QM9 pickles (https://doi.org/10.7910/DVN/JNGTDF), as
distributed under `~/rdkit_folder/qm9/*.pickle`.

Each pickle file is one SMILES. Its top-level dict has:
    smiles            : str
    conformers        : list[dict], one entry per conformer with keys:
        rd_mol            : rdkit.Chem.Mol — one internal conformer carrying
                            the 3D positions for this snapshot.
        totalenergy       : float — Hartree, absolute energy
        relativeenergy    : float — kcal/mol above the lowest conformer
        boltzmannweight   : float — normalised Boltzmann weight at 298.15 K
        geom_id, set, degeneracy, conformerweights : misc bookkeeping
    totalconfs        : int — sum of all degeneracies
    uniqueconfs       : int — len(conformers) (each entry is unique up to
                        symmetry / RMSD merging in CREST)
    lowestenergy      : float — Hartree
    poplowestpct      : float — % Boltzmann population on the lowest conf
    charge            : int
    temperature       : float — 298.15 K
    ensembleenergy    : float
    ensembleentropy   : float
    ensemblefreeenergy: float

The topology (atom types, bonds) is shared across all conformers of a given
SMILES — only the 3D positions differ between conformers. This module exposes
a single function `load_qm9_molecule` that returns a `MoleculeData` carrying
the shared topology + a list of per-conformer positions/energies, all as
torch tensors in the layout the score network expects.

Rotatable-bond definition (from the torsional-diffusion paper, Section 3):
    a bond (u, v) is *freely rotatable* iff severing it splits the molecular
    graph into exactly two connected components, each with at least two
    atoms.

That definition is graph-topological, includes double bonds, and excludes
ring bonds and terminal bonds (e.g. C-H, terminal methyl). It is what the
score model is meant to predict updates for.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import networkx as nx
import torch as pt

from chemdm.TorsionalDiffusionData import TorsionalDiffusionData, find_rotatable_bonds


def load_qm9_molecule( path: Path ) -> TorsionalDiffusionData:
    """
    Load one QM9 pickle into model-ready tensors.
    """
    with open( path, "rb" ) as f:
        raw = pickle.load(f)

    confs = raw["conformers"]
    if len(confs) == 0:
        raise ValueError(f"{path}: empty `conformers` list")

    # Topology comes from the first conformer's rd_mol (shared across all).
    mol0 = confs[0]["rd_mol"]
    N = mol0.GetNumAtoms()

    Z = pt.tensor( [a.GetAtomicNum() for a in mol0.GetAtoms()], dtype=pt.long )

    # Covalent edges, both directions, layout (E, 2) to match `rotatable_bonds`.
    src, dst = [], []
    for b in mol0.GetBonds():
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        src += [u, v]
        dst += [v, u]
    edge_index = pt.tensor(list(zip(src, dst)), dtype=pt.long).reshape(-1, 2)

    rot, sides = find_rotatable_bonds( mol0 )
    rotatable_bonds = pt.tensor( rot, dtype=pt.long ).reshape(-1, 2)

    # COO-style flatten of the per-bond c-side atom sets:
    #   side_atom_idx[k] = atom index on the c-side of bond `side_bond_idx[k]`.
    # Empty molecule (no rotatable bonds) gracefully yields empty (0,) tensors.
    side_atom_flat: list[int] = []
    side_bond_flat: list[int] = []
    for i, atoms in enumerate(sides):
        side_atom_flat.extend( atoms )
        side_bond_flat.extend( [i] * len(atoms) )
    side_atom_idx = pt.tensor( side_atom_flat, dtype=pt.long )
    side_bond_idx = pt.tensor( side_bond_flat, dtype=pt.long )

    conformers = []
    for c in confs:
        m = c["rd_mol"]
        if m.GetNumAtoms() != N:
            raise ValueError( f"{path}: conformer has {m.GetNumAtoms()} atoms, expected {N}" )
        pos = m.GetConformer(0).GetPositions()  # numpy (N, 3) float64
        conformers.append({
            "x": pt.tensor(pos, dtype=pt.float64),
            "totalenergy":     float(c["totalenergy"]),
            "relativeenergy":  float(c["relativeenergy"]),
            "boltzmannweight": float(c["boltzmannweight"]),
        })

    return TorsionalDiffusionData(
        smiles=raw["smiles"],
        Z=Z,
        edge_index=edge_index,
        rotatable_bonds=rotatable_bonds,
        side_atom_idx=side_atom_idx,
        side_bond_idx=side_bond_idx,
        conformers=conformers,
    )
