"""
Conformer matching for torsional-diffusion training data
(Jing et al. 2022, Appendix E).

For every CREST conformer in the parsed splits we:

    1. Generate N_RDKIT_FACTOR x N_CREST RDKit ETKDGv3 conformers on the
       original rd_mol (atom indexing stays identical to the dataset).
    2. MMFF94-optimise each (pulls L̂ toward an actual MM minimum, closing
       part of the RDKit-vs-CREST gap before matching).
    3. Compute torsion angles of every CREST and every RDKit conformer using
       a deterministic reference-atom choice per rotatable bond.
    4. For every (CREST i, RDKit k) pair: apply Δτ = τ_gt(i) - τ_init(k) to
       L̂_k via `apply_torsion_update`. The result is L̂_k's geometry with
       torsions matching CREST i. Heavy-atom Kabsch-aligned RMSD vs x_gt(i)
       is the matching cost.
    5. Hungarian-style bipartite assignment (scipy.optimize.linear_sum_assignment)
       pairs each CREST conformer with a unique RDKit conformer.
    6. Pack the matched geometries into the same per-split layout as
       `preparse_qm9.py` so dataset.py / train.py can swap data sources by
       flipping the parsed-dir path.

Output: <qm9_dir.parent>/conformer_matching/{train,val,test}.pt

Split definitions are re-used verbatim from <qm9_dir.parent>/parsed/splits.json
so this output is directly comparable to the CREST training set.

Sequential per-molecule loop. RDKit conformer generation (ETKDG + MMFF) is
the dominant cost — drop in a `multiprocessing.Pool` over `smiles_list` if
you need to parallelise.

Run:
    python conformer_matching.py
"""

from __future__ import annotations

import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch as pt

from dotenv import load_dotenv
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
from scipy.optimize import linear_sum_assignment

from chemdm.geometry import apply_torsion_update, kabsch_aligned_rmsd_torch
from qm9_parser import load_qm9_molecule, MoleculeData

RDLogger.DisableLog("rdApp.*")


# Config
N_RDKIT_FACTOR = 2          # generate this × N_CREST RDKit conformers per mol
MMFF = True                 # MMFF94 relax each RDKit conformer before matching
SEED = 42                   # ETKDG random seed
MAX_MOLECULES_PER_SPLIT = None  # set to int for a smoke run


def pick_reference_atoms( rotatable_bonds: pt.Tensor, edge_index: pt.Tensor, n_atoms: int ) -> tuple[pt.Tensor, pt.Tensor]:
    """
    For each rotatable bond (b, c), pick a deterministic neighbour of b (≠c)
    and a neighbour of c (≠b). These are anchor atoms for dihedral computation.

    The choice doesn't affect the action of `apply_torsion_update` — that
    function rotates the c-side by Δτ around the bond axis regardless of any
    reference. We only need the SAME reference atoms when extracting torsions
    from the CREST and RDKit geometries so Δτ is a meaningful delta.

    Rotatable bonds (per the paper's definition, enforced in
    `find_rotatable_bonds`) always have ≥ 2 atoms on each side, so both
    endpoints have at least one other neighbour.

    This function picks the smallest-index neighbor of b that isn't c. There is 
    no chemistry-awareness in this function, just a local indexing routine.
    """
    m = rotatable_bonds.shape[0]
    adj: list[set] = [set() for _ in range(n_atoms)]
    for u, v in edge_index.tolist():
        adj[u].add(v)

    ref_a = pt.empty(m, dtype=pt.long)
    ref_d = pt.empty(m, dtype=pt.long)
    for i in range(m):
        b, c = int(rotatable_bonds[i, 0]), int(rotatable_bonds[i, 1])
        ref_a[i] = sorted(adj[b] - {c})[0]
        ref_d[i] = sorted(adj[c] - {b})[0]
    return ref_a, ref_d


def torsions_for_geometry( x: pt.Tensor, rotatable_bonds: pt.Tensor, ref_a: pt.Tensor, ref_d: pt.Tensor ) -> pt.Tensor:
    """
    Vectorised dihedral a-b-c-d for each rotatable bond. Returns (m,) angles
    in radians in (-π, π].
    
    As torsions are defined per quadruple of atoms, not per bond, we need reference atoms 
    on the left of b and on the right of c to define each torsion uniquely. This choice
    does not need to be chemistry-aware - just needs to be consistent among molecules
    with the same atom ordering.

    See also: pick_reference_atoms
    """
    a = x[ref_a]
    b = x[rotatable_bonds[:, 0]]
    c = x[rotatable_bonds[:, 1]]
    d = x[ref_d]

    v1 = b - a
    v2 = c - b
    v3 = d - c

    n1 = pt.linalg.cross(v1, v2, dim=-1)
    n2 = pt.linalg.cross(v2, v3, dim=-1)
    v2_n = v2 / v2.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
    m1 = pt.linalg.cross(n1, v2_n, dim=-1)

    return pt.atan2( (m1 * n2).sum(-1), (n1 * n2).sum(-1) )


def kabsch_aligned_heavy_rmsd( x_pred: pt.Tensor, x_ref: pt.Tensor, Z: pt.Tensor ) -> float:
    """Heavy-atom Kabsch-aligned RMSD. Float64 on CPU for SVD stability."""
    heavy = (Z != 1)
    if int(heavy.sum()) < 3:
        heavy = pt.ones_like(Z, dtype=pt.bool)
    return float( kabsch_aligned_rmsd_torch(
        x_pred[heavy].cpu().to(pt.float64),
        x_ref[heavy].cpu().to(pt.float64),
    ) )


def generate_rdkit_conformers( rdmol: Chem.Mol, n_conf: int, seed: int = SEED, mmff: bool = MMFF ) -> list[pt.Tensor]:
    """
    ETKDGv3 + (optional) MMFF94. Returns a list of (N, 3) float32 tensors,
    one per successfully embedded conformer (≤ n_conf if RDKit drops any).
    """
    rdmol = Chem.Mol(rdmol)
    rdmol.RemoveAllConformers()

    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    cids = list( AllChem.EmbedMultipleConfs(rdmol, numConfs=n_conf, params=params) )

    if mmff:
        for cid in cids:
            try:
                AllChem.MMFFOptimizeMolecule(rdmol, confId=cid, maxIters=200)
            except Exception:
                pass

    return [ pt.tensor( rdmol.GetConformer(cid).GetPositions(), dtype=pt.float32 ) for cid in cids ]


def match_rdkit_to_crest( d: MoleculeData, rdmol: Chem.Mol, n_rdkit_factor: int = N_RDKIT_FACTOR, mmff: bool = MMFF ) -> list[pt.Tensor]:
    """
    Run conformer matching for one molecule. Returns a list of (N, 3) float32
    tensors — one matched geometry per CREST conformer, ordered like
    `d.conformers`.

    If RDKit can't supply enough conformers (< N_CREST), falls back to using
    the CREST positions directly so the output schema stays consistent.
    """
    n_crest = len( d.conformers )
    n_rdkit = n_rdkit_factor * n_crest

    rdkit_positions = generate_rdkit_conformers( rdmol, n_rdkit, mmff=mmff )
    n_rdkit_actual = len(rdkit_positions)
    if n_rdkit_actual < n_crest:
        # Fallback in case many rdkit conformers collapsed to the same local minimum
        print( 'Falling back. RDKit conformer set is not diverse enough.')
        return [c["x"].to(pt.float32) for c in d.conformers]

    # d.edge_index are all the molecule bonds. d
    ref_a, ref_d = pick_reference_atoms( d.rotatable_bonds, d.edge_index, len(d.Z) )

    # Compute cost matrix: every RMSD between crest and xTB conformers.
    crest_torsions = pt.stack([ torsions_for_geometry(c["x"].to(pt.float32), d.rotatable_bonds, ref_a, ref_d) for c in d.conformers ])
    rdkit_torsions = pt.stack([ torsions_for_geometry(pos, d.rotatable_bonds, ref_a, ref_d) for pos in rdkit_positions ])
    cost = np.zeros((n_crest, n_rdkit_actual), dtype=np.float64)
    aligned: list[list[pt.Tensor | None]] = [[None] * n_rdkit_actual for _ in range(n_crest)]

    # Iterate over every pair. 
    for i in range(n_crest):
        x_gt = d.conformers[i]["x"].to(pt.float32)
        for k in range(n_rdkit_actual):
            delta_tau = crest_torsions[i] - rdkit_torsions[k]
            x_aligned = apply_torsion_update( rdkit_positions[k], d.rotatable_bonds, d.side_atom_idx, d.side_bond_idx, delta_tau )
            cost[i, k] = kabsch_aligned_heavy_rmsd(x_aligned, x_gt, d.Z)
            aligned[i][k] = x_aligned

    # Optimal transport to match each CREST conformer to a unique RDKIT one.
    # Hungarian assignment solver is necessary to maintain diversity in the training
    # dataset. Simply matching each CREST conformer to its optimal RDKIT one
    # can cause an unwanted collapse.
    row_idx, col_idx = linear_sum_assignment(cost)
    matched: list[pt.Tensor] = [None] * n_crest        # type: ignore[list-item]
    for i, k in zip(row_idx, col_idx):
        matched[i] = aligned[i][k]
    return matched


def process_split( qm9_dir: Path, smiles_list: list[str], n_rdkit_factor: int, mmff: bool ) -> dict:
    """Walk the smiles list, run conformer matching per molecule, pack into
    the same per-split layout as `preparse_qm9.preparse_split`."""

    if MAX_MOLECULES_PER_SPLIT is not None:
        smiles_list = smiles_list[:MAX_MOLECULES_PER_SPLIT]

    mol_smiles: list[str] = []
    mol_Z: list[pt.Tensor] = []
    mol_edge_index: list[pt.Tensor] = []
    mol_rotatable_bonds: list[pt.Tensor] = []
    mol_side_atom: list[pt.Tensor] = []
    mol_side_bond: list[pt.Tensor] = []

    conf_x_chunks: list[pt.Tensor] = []
    conf_offsets: list[int] = [0]
    mol_id_flat: list[int] = []

    t0 = time.time()
    n_fallback = 0
    for m, smi in enumerate(smiles_list):
        if m > 0 and m % 10 == 0:
            dt  = time.time() - t0
            eta = (len(smiles_list) - m) / max(m / dt, 1e-9)
            print(f"  matched {m:>7,d}/{len(smiles_list):,}  ({m/dt:.1f} mol/s, eta {eta:.0f}s)")

        pickle_path = qm9_dir / f"{smi}.pickle"
        d = load_qm9_molecule( pickle_path ) # Loads all CREST conformers
        with open(pickle_path, "rb") as f:
            rdmol = pickle.load(f)["conformers"][0]["rd_mol"] # Load the representative Mol object

        try:
            matched = match_rdkit_to_crest( d, rdmol, n_rdkit_factor=n_rdkit_factor, mmff=mmff )
        except Exception as e:
            if n_fallback < 5:
                print(f"  match failed for {smi}: {type(e).__name__}: {e}; using CREST fallback")
            n_fallback += 1
            matched = [c["x"].to(pt.float32) for c in d.conformers]

        mol_smiles.append( d.smiles )
        mol_Z.append( d.Z )
        mol_edge_index.append( d.edge_index )
        mol_rotatable_bonds.append( d.rotatable_bonds )
        mol_side_atom.append( d.side_atom_idx )
        mol_side_bond.append( d.side_bond_idx )

        N = int(d.Z.shape[0])
        for x_matched in matched:
            conf_x_chunks.append(x_matched.to(pt.float32))
            conf_offsets.append(conf_offsets[-1] + N)
            mol_id_flat.append(m)

    if n_fallback > 0:
        print(f"  {n_fallback:,} molecules used CREST fallback")

    return {
        "mol_smiles": mol_smiles,
        "mol_Z": mol_Z,
        "mol_edge_index": mol_edge_index,
        "mol_rotatable_bonds": mol_rotatable_bonds,
        "mol_side_atom": mol_side_atom,
        "mol_side_bond": mol_side_bond,
        "x_flat": pt.cat(conf_x_chunks, dim=0),
        "x_offset": pt.tensor(conf_offsets, dtype=pt.long),
        "mol_id": pt.tensor(mol_id_flat, dtype=pt.long),
    }


def main() -> None:
    load_dotenv()
    qm9_dir = Path( os.environ["QM9_FOLDER"] )
    parsed_dir = qm9_dir.parent / "parsed"
    out_dir = qm9_dir.parent / "conformer_matching"
    out_dir.mkdir( parents=True, exist_ok=True )

    with open(parsed_dir / "splits.json") as f:
        splits_data = json.load(f)

    for split_name in ("train", "val", "test"):
        smiles_list = splits_data["splits"][split_name]
        print(f"\nConformer matching for {split_name} ({len(smiles_list):,} molecules, "
              f"N_RDKIT = {N_RDKIT_FACTOR} x N_CREST, MMFF={MMFF})...")
        data = process_split( qm9_dir, smiles_list, N_RDKIT_FACTOR, MMFF )
        out_file = out_dir / f"{split_name}.pt"
        pt.save( data, out_file )
        n_conf  = int(data["mol_id"].numel())
        size_mb = out_file.stat().st_size / 1e6
        print(f"  {split_name}: {n_conf:,} matched conformers -> {out_file} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
