"""
Test suite of 10 molecules spanning H2 (2 atoms) to a C60 alkane (182 atoms),
with element coverage H, C, N, O, F, S.

Geometries are generated deterministically from SMILES with RDKit (ETKDGv3
embedding + MMFF optimization, fixed random seed), so the suite is reproducible
without shipping coordinate files. MMFF-optimized geometries are *not* GFN
minima, so forces are non-trivial (good for testing force correspondence).

Each molecule is returned as (name, Z, x_A): atomic numbers and positions in
Angstrom. All are neutral closed-shell singlets.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

# Size ladder, small -> large. (name, SMILES). Approx atom counts in comments.
SUITE = [
    ("H2",           "[H][H]"),                                    # 2    H
    ("H2O",          "O"),                                         # 3    O H
    ("methanol",     "CO"),                                        # 6    C O H
    ("thiophene",    "c1ccsc1"),                                   # 9    C S H
    ("benzene",      "c1ccccc1"),                                  # 12   C H
    ("aspirin",      "CC(=O)Oc1ccccc1C(=O)O"),                     # 21   C O H
    ("caffeine",     "Cn1cnc2c1c(=O)n(C)c(=O)n2C"),                # 24   C N O H
    ("cholesterol",  "C[C@H](CCCC(C)C)[C@H]1CC[C@@H]2[C@@]1(C)"
                     "CC[C@H]3[C@H]2CC=C4[C@@]3(C)CC[C@@H](O)C4"),  # 74   C O H
    ("atorvastatin", "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)"
                     "c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)O"),  # 76   C N O F H
    ("C60-alkane",   "C" * 60),                                    # 182  C H
]

EMBED_SEED = 0xC0FFEE


def build_geometry( smiles: str, seed: int = EMBED_SEED ) -> tuple[np.ndarray, np.ndarray]:
    """SMILES -> (Z, x_A) via RDKit ETKDGv3 embedding + MMFF optimization."""
    mol = Chem.AddHs( Chem.MolFromSmiles(smiles) )
    params = AllChem.ETKDGv3( )
    params.randomSeed = seed
    if AllChem.EmbedMolecule( mol, params ) != 0:
        # Fall back to random-coordinate embedding for awkward cases.
        AllChem.EmbedMolecule( mol, useRandomCoords=True, randomSeed=seed )
    AllChem.MMFFOptimizeMolecule( mol, maxIters=500 )
    Z = np.array( [a.GetAtomicNum() for a in mol.GetAtoms()], dtype=int )
    x_A = np.asarray( mol.GetConformer().GetPositions(), dtype=float )
    return Z, x_A


def get_molecules( names: list[str] | None = None ) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Returns the suite as a list of (name, Z, x_A) tuples."""
    wanted = {n for n in names} if names is not None else None
    out = []
    for name, smi in SUITE:
        if wanted is not None and name not in wanted:
            continue
        Z, x_A = build_geometry(smi)
        out.append( (name, Z, x_A) )
    return out


if __name__ == "__main__":
    for name, Z, x in get_molecules():
        print(f"  {name:14s} n_atoms={len(Z):4d}  elements={sorted(set(Z.tolist()))}")
