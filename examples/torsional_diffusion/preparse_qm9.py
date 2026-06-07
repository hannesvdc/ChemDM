"""
One-shot preprocessing of GEOM-QM9 for torsional-diffusion training.

Three passes in one script:

    1. Walk the QM9 directory and parse every pickle. Drop molecules with
       zero rotatable bonds (untrainable for torsional diffusion).
    2. Randomly split the surviving SMILES into train / val / test
       (default 70 / 20 / 10), at the **molecule** level — so all conformers
       of a given molecule fall into the same split. This allows us
        to check if the network learned the chemistry.
    3. For each split, pack the per-molecule topology (Z, bonds,
       side_atom_idx, side_bond_idx) and the per-conformer positions into a
       single .pt file. Float32 positions; everything else `long`.
    4. side_atom_idx is a list of atoms on the 'c' side (end point) of 
       every bond, side_bond_idx maps to which bond that atom belongs. 
       Both are size (P,) with P the total number of atom on the right 
       of all bonds.

The output is small enough to load fully into RAM at training time
(~300 MB for train.pt under default settings) so the Dataset implementation
can be plain indexing.

Outputs in --out-dir:
    splits.json         metadata + the three SMILES lists, for reproducibility
    train.pt            preparsed train split (see layout below)
    val.pt              preparsed val split
    test.pt             preparsed test split

Per-split .pt file layout (a dict):

    mol_smiles    : list[str]                       length M
    mol_Z         : list[Tensor (N_i,) long]        length M
    mol_bonds     : list[Tensor (m_i, 2) long]      length M
    mol_side_atom : list[Tensor (P_i,) long]        length M  (COO)
    mol_side_bond : list[Tensor (P_i,) long]        length M  (COO)
    x_flat        : Tensor (sum_c N_c, 3) float32             — all conformer positions concatenated
    x_offset      : Tensor (C+1,) long                        — cumulative atom count; conformer k lives at
                                                                x_flat[x_offset[k] : x_offset[k+1]]
    mol_id        : Tensor (C,) long                length C  (which molecule each conformer belongs to)

`x_*` is stored flat (one tensor for all conformers) because torch.save adds
~300 B of pickle overhead per Tensor in a list — for 1.2M small conformer
tensors that overhead dominates the actual position bytes. Slicing
`x_flat[s:e]` at dataset-access time is a free view, no copy.

Run:
    /opt/homebrew/anaconda3/envs/py311/bin/python preparse_qm9.py

Pass `--max-molecules N` for a quick dev run on a subset.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch as pt

from dotenv import load_dotenv

from qm9_parser import load_qm9_molecule


def scan_and_split( qm9_dir: Path, train_frac: float, val_frac: float, seed: int, max_molecules: int | None = None ) -> dict[str, list[str]]:
    """
    Walk qm9_dir, drop 0-rotatable molecules, and randomly split the rest
    into train / val / test SMILES lists.

    The walk also serves as a parse smoke test — any pickle that fails to
    load is reported and skipped.
    """
    files = sorted( f for f in os.listdir(qm9_dir) if f.endswith(".pickle") )
    if max_molecules is not None:
        files = files[:max_molecules]
    print( f"Scanning {len(files):,} pickle files for rotatable bonds..." )

    smiles_keep:  list[str] = []
    n_zero_rot = 0
    n_failed = 0

    t0 = time.time()
    for i, fn in enumerate(files):
        if i and i % 5000 == 0:
            dt = time.time() - t0
            eta = (len(files) - i) / (i / dt)
            print( f"  scanned {i:>7,d}/{len(files):,}  ({i/dt:.0f} mol/s, eta {eta:.0f}s)" )
        try:
            d = load_qm9_molecule( qm9_dir / fn )
        except Exception as e:
            n_failed += 1
            if n_failed <= 5:
                print( f"  skip {fn}: {e}" )
            continue
        if d.bonds.shape[0] == 0:
            n_zero_rot += 1
            continue
        smiles_keep.append( d.smiles )

    print(
        f"Kept {len(smiles_keep):,} molecules with >=1 rotatable bond  "
        f"(dropped {n_zero_rot:,} zero-rotatable, {n_failed:,} parse-failed)"
    )

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(smiles_keep))
    smiles_shuffled = [smiles_keep[int(i)] for i in perm]

    n_total = len(smiles_shuffled)
    n_train = int( train_frac * n_total )
    n_val   = int( val_frac   * n_total )
    return {
        "train": smiles_shuffled[:n_train],
        "val":   smiles_shuffled[n_train:n_train + n_val],
        "test":  smiles_shuffled[n_train + n_val:],
    }


def preparse_split( qm9_dir: Path, smiles_list: list[str] ) -> dict:
    """
    Load every (mol, conformer) pair for one split and pack into the .pt layout
    documented at the top of this file.
    """
    mol_smiles : list[str] = []
    mol_Z : list[pt.Tensor] = []
    mol_bonds : list[pt.Tensor] = []
    mol_side_atom : list[pt.Tensor] = []
    mol_side_bond : list[pt.Tensor] = []

    # Accumulate per-conformer positions as individual tensors during the
    # parse loop, then cat() them into one flat tensor at the end. This
    # avoids 300 B of pickle overhead per conformer Tensor when we eventually
    # `torch.save` the result.
    # This is an example of sequential graph-batching instead of dense 
    # batching in a new dimension (typically 0).
    conf_x_chunks : list[pt.Tensor] = []
    conf_offsets  : list[int]       = [0]
    mol_id_flat   : list[int]       = []

    t0 = time.time()
    for m, smi in enumerate(smiles_list):
        if m and m % 5000 == 0:
            dt = time.time() - t0
            eta = (len(smiles_list) - m) / (m / dt)
            print( f"  parsed {m:>7,d}/{len(smiles_list):,}  ({m/dt:.0f} mol/s, eta {eta:.0f}s)" )
        d = load_qm9_molecule( qm9_dir / f"{smi}.pickle" )

        mol_smiles.append( d.smiles )
        mol_Z.append( d.Z )
        mol_bonds.append( d.bonds )
        mol_side_atom.append( d.side_atom_idx )
        mol_side_bond.append( d.side_bond_idx )

        N = int( d.Z.shape[0] )
        for conf in d.conformers:
            conf_x_chunks.append( conf["x"].to(pt.float32) )    # downcast from rdkit's float64
            conf_offsets.append( conf_offsets[-1] + N )
            mol_id_flat.append( m )

    x_flat = pt.cat( conf_x_chunks, dim=0 )            # (sum_c N_c, 3) float32
    x_offset = pt.tensor( conf_offsets, dtype=pt.long )  # (C+1,)
    mol_id = pt.tensor( mol_id_flat, dtype=pt.long )

    return {
        "mol_smiles": mol_smiles,
        "mol_Z": mol_Z,
        "mol_bonds": mol_bonds,
        "mol_side_atom": mol_side_atom,
        "mol_side_bond": mol_side_bond,
        "x_flat": x_flat,
        "x_offset": x_offset,
        "mol_id": mol_id,
    }


def main() -> None:
    load_dotenv()
    qm9_dir = Path( os.environ["QM9_FOLDER"] )

    out_dir = qm9_dir.parent / "parsed"
    out_dir.mkdir( parents=True, exist_ok=True )

    seed = 42
    train_frac = 0.70
    val_frac = 0.20
    splits = scan_and_split(
        qm9_dir=qm9_dir,
        train_frac=train_frac,
        val_frac=val_frac,
        seed=seed,
        max_molecules=None,
    )

    # Save the split definitions before doing the heavy work, so we always
    # have a reproducible record of what went where.
    with open( out_dir / "splits.json", "w" ) as f:
        json.dump(
            {
                "seed": seed,
                "train_frac": train_frac,
                "val_frac": val_frac,
                "max_molecules": None,
                "qm9_dir": str(qm9_dir),
                "splits": splits,
            },
            f, indent=2,
        )
    print(
        f"\nWrote {out_dir / 'splits.json'}: "
        f"train={len(splits['train']):,}, val={len(splits['val']):,}, test={len(splits['test']):,}"
    )

    for kind in ("train", "val", "test"):
        print( f"\nPre-parsing {kind} ({len(splits[kind]):,} molecules)..." )
        data = preparse_split( qm9_dir, splits[kind] )
        n_conf = int( data["mol_id"].numel() )
        out_file = out_dir / f"{kind}.pt"
        pt.save( data, out_file )
        size_mb = out_file.stat().st_size / 1e6
        print( f"  {kind}: {n_conf:>9,d} conformers  ->  {out_file} ({size_mb:.1f} MB)" )


if __name__ == "__main__":
    main()
