"""
Conformer matching for torsional-diffusion training data
(Jing et al. 2022, Appendix E).

For every CREST conformer in the parsed splits we:

    1. Generate N_RDKIT_FACTOR x N_CREST RDKit ETKDGv3 conformers on the
       original rd_mol (atom indexing stays identical to the dataset).
    2. Compute torsion angles of every CREST and every RDKit conformer using
       a deterministic reference-atom choice per rotatable bond.
    3. For every (CREST i, RDKit k) pair: apply Δτ = τ_gt(i) - τ_init(k) to
       L̂_k via `apply_torsion_update`. The result is L̂_k's geometry with
       torsions matching CREST i. Heavy-atom Kabsch-aligned RMSD vs x_gt(i)
       is the matching cost.
    4. Hungarian-style bipartite assignment (scipy.optimize.linear_sum_assignment)
       pairs each CREST conformer with a unique RDKit conformer.
    5. Apply differential evolution to each machted (CREST, RDKit) conformer pair
       to fint the torsion update \delta \tau from RDKit that minimizes RMSD
       with the CREST conformer.
    5. Pack the matched and minimized geometries into the same per-split layout as
       `preparse_qm9.py` so dataset.py / train.py can swap data sources by
       flipping the parsed-dir path.

Output: <qm9_dir.parent>/conformer_matching/{train,val,test}.pt

Split definitions are re-used verbatim from <qm9_dir.parent>/parsed/splits.json
so this output is directly comparable to the CREST training set.

Sequential per-molecule loop. RDKit conformer generation (ETKDG) and differential_evolution
are the dominant costs — drop in a `multiprocessing.Pool` over `smiles_list` if
you need to parallelise.

Run:
    python conformer_matching.py
"""

from __future__ import annotations

import json
import multiprocessing as mp
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
from scipy.optimize import linear_sum_assignment, differential_evolution

from chemdm.geometry import apply_torsion_update, kabsch_aligned_rmsd_torch
from qm9_parser import load_qm9_molecule, MoleculeData

RDLogger.DisableLog("rdApp.*")


# Config
N_RDKIT_FACTOR = 2          # generate this × N_CREST RDKit conformers per mol
SEED = 42                   # ETKDG random seed
MAX_MOLECULES_PER_SPLIT = None  # set to int for a smoke run

# Differential-evolution torsion refinement (Jing et al., Appendix E)
DE_POPSIZE = 15             # population = DE_POPSIZE × n_rotatable_bonds
DE_MAXITER = 100            # max generations; warm-started, so converges fast
DE_TOL = 1e-3           # convergence tolerance on heavy-atom RMSD (Å)

N_WORKERS = os.cpu_count() # parallel processes over molecules (CPU-bound max. is os.cpu_count() )


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


def generate_rdkit_conformers( rdmol: Chem.Mol, n_conf: int, seed: int = SEED ) -> list[pt.Tensor]:
    """
    ETKDGv3 generation, no mmf94 relaxation. Returns a list of (N, 3) float32 tensors,
    one per successfully embedded conformer (≤ n_conf if RDKit drops any).
    """
    rdmol = Chem.Mol(rdmol)
    rdmol.RemoveAllConformers()

    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    cids = list( AllChem.EmbedMultipleConfs(rdmol, numConfs=n_conf, params=params) )

    return [ pt.tensor( rdmol.GetConformer(cid).GetPositions(), dtype=pt.float32 ) for cid in cids ]


def refine_torsions_de( rdkit_x: pt.Tensor,         # (N, 3) assigned RDKit local structure
                        x_gt: pt.Tensor,            # (N, 3) target CREST conformer
                        rotatable_bonds: pt.Tensor, # (m, 2)
                        side_atom_idx: pt.Tensor,
                        side_bond_idx: pt.Tensor,
                        Z: pt.Tensor,
                        delta0: pt.Tensor,          # (m,) dihedral-transfer (von Mises) warm-start delta
    ) -> pt.Tensor:
    """
    Differential-evolution torsion refinement (Jing et al. 2022, Appendix E).

    Searches the torsion update Δτ ∈ [−π, π)^m that minimises the heavy-atom
    Kabsch RMSD between `rdkit_x` rotated by Δτ and the ground-truth conformer
    `x_gt`. Plain dihedral transfer only sets each dihedral to the CREST value,
    which is NOT the RMSD optimum on a different local structure (the reference
    atoms sit differently relative to the bond axis, and the bonds couple), so
    it leaves strained geometries. DE finds the true joint optimum.

    Warm-started at `delta0` (so the result is never worse than plain transfer);
    Latin-hypercube fills the rest of the population for global escape. Returns
    the refined (N, 3) geometry.
    """
    m = int( rotatable_bonds.shape[0] )
    if m == 0:
        return rdkit_x

    def objective( delta_np: np.ndarray ) -> float:
        delta = pt.tensor( np.asarray( delta_np, dtype=np.float32 ) )
        x = apply_torsion_update( rdkit_x, rotatable_bonds, side_atom_idx, side_bond_idx, delta )
        return kabsch_aligned_heavy_rmsd( x, x_gt, Z )

    # delta0 (= τ_gt − τ_rdkit) can fall outside [−π, π); wrap it (the rotation
    # is 2π-periodic) so the warm start is a valid in-bounds point.
    d0 = pt.remainder( delta0 + np.pi, 2.0 * np.pi ) - np.pi

    result = differential_evolution(
        objective,
        bounds=[ (-np.pi, np.pi) ] * m,
        x0=d0.detach().cpu().numpy().astype( np.float64 ),
        popsize=DE_POPSIZE,
        maxiter=DE_MAXITER,
        tol=DE_TOL,
        seed=SEED,
        polish=True,
    )

    delta = pt.tensor( result.x.astype( np.float32 ) )
    return apply_torsion_update( rdkit_x, rotatable_bonds, side_atom_idx, side_bond_idx, delta )


def match_rdkit_to_crest( d: MoleculeData, rdmol: Chem.Mol, n_rdkit_factor: int = N_RDKIT_FACTOR ) -> list[pt.Tensor]:
    """
    Run conformer matching for one molecule. Returns a list of (N, 3) float32
    tensors — one matched geometry per CREST conformer, ordered like
    `d.conformers`.

    Two-stage matching (Jing et al., Appendix E):
      * a fast dihedral-transfer RMSD fills the K x N_rdkit cost matrix, and
        Hungarian assignment pairs each CREST conformer with a unique RDKit
        local structure (the bijection preserves local-structure diversity);
      * each assigned pair is then refined with differential evolution over the
        torsions to actually minimise heavy-atom RMSD (`refine_torsions_de`).

    If RDKit underproduces (fewer local structures than CREST conformers), the
    available structures are reused (tiled) so every conformer is still matched
    and DE-refined onto an RDKit local structure. If RDKit yields nothing at
    all, an empty list is returned and the molecule is dropped (it is also
    un-embeddable at inference, so it is out of scope for the method).
    """
    n_crest = len( d.conformers )
    n_rdkit = n_rdkit_factor * n_crest

    rdkit_positions = generate_rdkit_conformers( rdmol, n_rdkit )
    n_rdkit_actual = len(rdkit_positions)
    if n_rdkit_actual == 0:
        # RDKit produced nothing — no local structure to match onto. Drop the
        # molecule entirely: it is un-embeddable at inference too (eval_worker's
        # rdkit_starting_geometry returns None on embed failure), so its CREST
        # conformers would never be sampled and would only reintroduce the
        # shift. The empty list signals process_split to skip it.
        print( f'Skipping {d.smiles}: RDKit produced no conformers.')
        return []
    if n_rdkit_actual < n_crest:
        # Fewer distinct RDKit local structures than CREST conformers. Reuse
        # them (tile) instead of bailing to raw CREST geometries, so EVERY
        # conformer is still matched + DE-refined onto an RDKit local structure
        # — the training input stays the same KIND regardless of how many
        # conformers RDKit produced. Some local-structure diversity is lost,
        # which is unavoidable when RDKit underproduces.
        reps = -(-n_crest // n_rdkit_actual)   # ceil division
        rdkit_positions = rdkit_positions * reps
        n_rdkit_actual = len(rdkit_positions)

    # d.edge_index are all the molecule bonds.
    ref_a, ref_d = pick_reference_atoms( d.rotatable_bonds, d.edge_index, len(d.Z) )

    # Cost matrix: dihedral-transfer (von Mises) heavy-atom RMSD for every
    # (CREST i, RDKit k) pair. This is only the assignment proxy — the final
    # geometry comes from the DE refinement below.
    crest_torsions = pt.stack([ torsions_for_geometry(c["x"].to(pt.float32), d.rotatable_bonds, ref_a, ref_d) for c in d.conformers ])
    rdkit_torsions = pt.stack([ torsions_for_geometry(pos, d.rotatable_bonds, ref_a, ref_d) for pos in rdkit_positions ])
    cost = np.zeros((n_crest, n_rdkit_actual), dtype=np.float64)
    for i in range(n_crest):
        x_gt = d.conformers[i]["x"].to(pt.float32)
        for k in range(n_rdkit_actual):
            delta_tau = crest_torsions[i] - rdkit_torsions[k]
            x_aligned = apply_torsion_update( rdkit_positions[k], d.rotatable_bonds, d.side_atom_idx, d.side_bond_idx, delta_tau )
            cost[i, k] = kabsch_aligned_heavy_rmsd(x_aligned, x_gt, d.Z)

    # Hungarian assignment maintains diversity in the training dataset: matching
    # each CREST conformer to its individually-optimal RDKit one can collapse
    # many CREST conformers onto the same local structure.
    row_idx, col_idx = linear_sum_assignment(cost)

    # Refine each assigned pair with differential evolution over the torsions,
    # warm-started at its dihedral-transfer delta.
    matched: list[pt.Tensor] = [None] * n_crest        # type: ignore[list-item]
    for i, k in zip(row_idx, col_idx):
        x_gt = d.conformers[i]["x"].to(pt.float32)
        delta0 = crest_torsions[i] - rdkit_torsions[k]
        matched[i] = refine_torsions_de(
            rdkit_positions[k], x_gt,
            d.rotatable_bonds, d.side_atom_idx, d.side_bond_idx, d.Z,
            delta0,
        )
    return matched


def _worker_init() -> None:
    # One torch/BLAS thread per worker process so N_WORKERS processes don't
    # oversubscribe the cores (the parallelism is across processes, not threads).
    pt.set_num_threads( 1 )


def _match_one( task: tuple ):
    """Per-molecule worker, run in a separate Pool *process*. Loads the molecule,
    runs conformer matching, and returns the packed per-molecule result, or
    None if the molecule is dropped (RDKit produced no conformers)."""
    qm9_dir, smiles, n_rdkit_factor = task

    pickle_path = qm9_dir / f"{smiles}.pickle"
    d = load_qm9_molecule( pickle_path )                       # all CREST conformers
    with open(pickle_path, "rb") as f:
        rdmol = pickle.load(f)["conformers"][0]["rd_mol"]      # representative Mol

    fallback = False
    try:
        matched = match_rdkit_to_crest( d, rdmol, n_rdkit_factor=n_rdkit_factor )
    except Exception as e:
        print(f"  match failed for {smiles}: {type(e).__name__}: {e}; using CREST fallback")
        fallback = True
        matched = [c["x"].to(pt.float32) for c in d.conformers]

    if len(matched) == 0:
        return None

    return (
        d.smiles, d.Z, d.edge_index, d.rotatable_bonds,
        d.side_atom_idx, d.side_bond_idx,
        [x.to(pt.float32) for x in matched], fallback,
    )


def process_split( qm9_dir: Path, smiles_list: list[str], n_rdkit_factor: int ) -> dict:
    """Walk the smiles list, run conformer matching per molecule (parallel over
    a `multiprocessing.Pool` of N_WORKERS *processes* — CPU-bound, so processes
    not threads), pack into the same per-split layout as
    `preparse_qm9.preparse_split`."""

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

    tasks = [ (qm9_dir, smi, n_rdkit_factor) for smi in smiles_list ]
    t0 = time.time()
    n_fallback = 0
    n_dropped  = 0

    # imap_unordered streams results back as worker processes finish. Output
    # order is irrelevant (training shuffles) and each molecule is deterministic
    # (fixed SEED), so the dataset content is reproducible. The parent does the
    # packing, so mol_idx / offsets stay consistent regardless of finish order.
    with mp.Pool( processes=N_WORKERS, initializer=_worker_init ) as pool:
        for done, res in enumerate( pool.imap_unordered( _match_one, tasks ), start=1 ):
            if done % 100 == 0:
                dt  = time.time() - t0
                eta = (len(tasks) - done) / max(done / dt, 1e-9)
                print(f"  matched {done:>7,d}/{len(tasks):,}  ({done/dt:.1f} mol/s, eta {eta:.0f}s)")

            if res is None:
                n_dropped += 1
                continue

            smiles, Z, edge_index, rot, side_atom, side_bond, matched, fallback = res
            if fallback:
                n_fallback += 1

            mol_idx = len(mol_smiles)   # index this molecule will occupy
            mol_smiles.append( smiles )
            mol_Z.append( Z )
            mol_edge_index.append( edge_index )
            mol_rotatable_bonds.append( rot )
            mol_side_atom.append( side_atom )
            mol_side_bond.append( side_bond )

            N = int(Z.shape[0])
            for x_matched in matched:
                conf_x_chunks.append( x_matched )
                conf_offsets.append( conf_offsets[-1] + N )
                mol_id_flat.append( mol_idx )

    dt = time.time() - t0
    print(f"  done: {len(tasks):,} molecules in {dt:.0f}s "
          f"({len(tasks)/max(dt,1e-9):.1f} mol/s on {N_WORKERS} processes)")
    if n_fallback > 0:
        print(f"  {n_fallback:,} molecules used CREST fallback")
    if n_dropped > 0:
        print(f"  {n_dropped:,} molecules dropped (RDKit produced no conformers)")

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

    with open( parsed_dir / "splits.json" ) as f:
        splits_data = json.load(f)

    for split_name in ("train", "val", "test"):
        smiles_list = splits_data["splits"][split_name]
        print(f"\nConformer matching for {split_name} ({len(smiles_list):,} molecules, "
              f"N_RDKIT = {N_RDKIT_FACTOR} x N_CREST)...")
        data = process_split( qm9_dir, smiles_list, N_RDKIT_FACTOR )
        out_file = out_dir / f"{split_name}.pt"
        pt.save( data, out_file )
        n_conf = int(data["mol_id"].numel())
        size_mb = out_file.stat().st_size / 1e6
        print(f"  {split_name}: {n_conf:,} matched conformers -> {out_file} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
