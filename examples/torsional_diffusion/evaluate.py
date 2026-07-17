"""
Evaluate trained torsional-diffusion score models against the CREST ground
truth on GEOM-QM9.

What it measures
----------------
For each (model, split, molecule) it draws K = 2·n_crest conformers with the
reverse-SDE sampler and compares them to the molecule's CREST ensemble:

    AMR-R / AMR-P     mean min heavy-atom Kabsch RMSD, in two directions
                      (see geom_metrics for the full explanation):
                        AMR-R (recall)    — for each CREST conformer, the closest
                                            generated sample, averaged over CREST.
                                            Low = the true ensemble is covered;
                                            penalizes missing modes.
                        AMR-P (precision) — for each generated sample, the closest
                                            CREST conformer, averaged over samples.
                                            Low = every sample is realistic;
                                            penalizes spurious conformers.
    Cov-R(δ)/Cov-P(δ) same two directions, but the fraction within δ ∈
                      {0.5,0.75,1.0,1.25} Å instead of the mean.

Then it relaxes every generated sample with GFN2-xTB (the level CREST minima
were made at, via chemdm.relaxMolecule) and recomputes BOTH AMR directions on
the relaxed geometries:

    relaxed AMR-R     avg min RMSD from each CREST conformer to its closest
                      relaxed sample — is every real basin still covered after
                      relaxation? (computed only over samples that relaxed
                      successfully, so it is slightly pessimistic when some fail;
                      n_relaxed / n_failed are recorded per molecule.)
    relaxed AMR-P     avg min RMSD of a *relaxed* sample to its closest CREST
                      conformer — do the generated conformers fall into real GFN2
                      basins?
    relax shift       avg heavy-atom RMSD a sample moves under relaxation —
                      small = the model already lands near a minimum.

Relaxed coverage (Cov-R/Cov-P on relaxed geometries) is NOT computed — the
relaxed stage reports only the two AMR directions plus the shift.

The runs (all at equal sample budget K = 2·n_crest)
---------------------------------------------------
    crest     CREST-trained model, started from each real CREST local
              structure (x2 torsion inits). The first-principles test of the
              diffusion itself, with no starting-geometry shift.
    rdkit_t1  RDKit-trained model, 2·n_crest fresh ETKDG backbones x 1 torsion
              init  (backbone-diversity arm of the A/B).
    rdkit_t2  RDKit-trained model, n_crest ETKDG backbones x 2 torsion inits
              (torsion-init-diversity arm of the A/B).

t1 vs t2 isolates, at the same K, whether re-using an expensive ETKDG backbone
with a second random torsion init buys as much coverage as a fresh backbone.

Config is module-level constants (no CLI args). Results print as a side-by-side
table and are written to eval_results/{label}_{split}.json.

Run:
    python evaluate.py
"""

from __future__ import annotations

import json
import os
import pickle
import time
from pathlib import Path

import multiprocessing as mp

import numpy as np
import torch as pt

from dotenv import load_dotenv

from chemdm.TBLitePotential import TBLitePotential
from chemdm.relaxMolecule import relaxMolecule
from chemdm.Constants import KJ_MOL_TO_EV
from chemdm.TorsionalDiffusionSampling import TorsionalDiffusionData, sample_conformers, kabsch_aligned_heavy_rmsd, generate_rdkit_conformers
from chemdm.TorsionalScoreNetwork import TorsionalScoreNetwork

from qm9_parser import load_qm9_molecule


# Config
N_PER_SPLIT = 100                 # molecules (with >=1 rotatable bond) per split
SPLITS = ("train", "val", "test")
DEVICE = "cpu"                    # sampling device; cpu sidesteps the MPS variable-shape graph-cache blowup
SEED = 42
N_SDE_STEPS = 20                  # reverse-SDE steps (matches sampler default)
CUTOFF = 5.0                      # spatial-graph cutoff (matches training)
COV_DELTAS = (0.5, 0.75, 1.0, 1.25)    # coverage thresholds, Å

XTB_METHOD = "GFN2-xTB"           # relaxation level (matches how CREST minima were made)
FORCE_TOL_EV_A = 0.02             # relaxation convergence: max per-atom force (eV/Å)
PRUNE_RMSD = 0.125                # dedup before relaxing: raw samples within this heavy-atom
                                  # RMSD (Å) are treated as one conformer -> one relaxation.
                                  # Keep it well below the smallest distinct-basin separation;
                                  # too large merges real basins and drops recall.
N_RELAX_WORKERS = os.cpu_count()  # CPU-bound, embarrassingly parallel over samples

# Each run = (checkpoint, start mode, backbone/torsion factors). K = backbone_mult·n_crest · torsion_inits.
RUNS = [
    {"label": "crest",    "ckpt": "models/crest_best.pt", "start": "crest", "backbone_mult": 1, "torsion_inits": 2},
    {"label": "rdkit_t1", "ckpt": "models/rdkit_de.pt",   "start": "rdkit", "backbone_mult": 2, "torsion_inits": 1},
]

OUT_DIR = Path( "./eval_results" )
BASELINE_LABEL = "crest"          # ceiling run for the gap table: metric[run] − metric[baseline]


# Model
def load_model( ckpt_path: Path, device: pt.device ) -> TorsionalScoreNetwork:
    """Instantiate the score network and load weights. Accepts both a bare
    state_dict (best.pt) and a training checkpoint dict carrying "model"."""
    model = TorsionalScoreNetwork().to( device=device, dtype=pt.float32 )
    state = pt.load( ckpt_path, map_location=device, weights_only=True )
    model.load_state_dict( state )
    model.eval()
    return model


# Starting geometries
def build_starting_geometries( d: TorsionalDiffusionData, rd_mol, start: str, backbone_mult: int, torsion_inits: int ) -> list[np.ndarray] | None:
    """
    Return the K = (backbone_mult·n_crest)·torsion_inits starting structures the
    sampler will randomize torsions on. Only the local structure (bonds/angles)
    of each backbone matters — the sampler overwrites torsions with a uniform
    prior — so re-using a backbone with a fresh torsion init gives an
    independent draw.

    start="crest": backbones are the molecule's real CREST local structures.
    start="rdkit": backbones are fresh ETKDGv3 embeds (same atom indexing).
                   Tiled if RDKit underproduces; None if RDKit yields nothing
                   (the molecule is un-embeddable at inference too).
    """
    n_crest = len( d.conformers )
    n_backbones = 100 #backbone_mult * n_crest
    print( n_backbones )

    if start == "crest":
        base = [ c["x"].numpy().astype( np.float32 ) for c in d.conformers ]
    elif start == "rdkit":
        embeds = generate_rdkit_conformers( rd_mol, n_backbones, seed=SEED )
        if len( embeds ) == 0:
            return None
        base = [ e.numpy().astype( np.float32 ) for e in embeds ]
    else:
        raise ValueError( f"unknown start mode {start!r}" )

    # Replicate each backbone `torsion_inits` times -> K starting structures.
    # Done by cycling through the base conformers.
    backbones = [ base[i % len( base )] for i in range( n_backbones ) ]
    out = []
    for b in backbones:
        out.extend( b for _ in range(torsion_inits) )
    print( 'rdkit backbones generated')
    return out


# Metrics
def rmsd_matrix( samples: np.ndarray, gt_confs: list[dict], Z: pt.Tensor ) -> np.ndarray:
    """(K, n_gt) heavy-atom Kabsch RMSD between each sample and each CREST conformer."""
    K = samples.shape[0]
    G = len( gt_confs )
    M = np.zeros( (K, G), dtype=np.float64 )
    for k in range( K ):
        xk = pt.tensor( samples[k], dtype=pt.float32 )
        for g in range( G ):
            M[k, g] = kabsch_aligned_heavy_rmsd( xk, gt_confs[g]["x"], Z )
    return M


def geom_metrics( M: np.ndarray ) -> dict:
    """
    AMR (Average Minimum RMSD) and Coverage in both directions, from a
    (K, n_gt) matrix where M[k, g] = RMSD between generated sample k and CREST
    conformer g. Lower AMR / higher Coverage are better for both directions.

    The two directions answer opposite questions, and a model can be good at one
    and bad at the other — which is exactly why both are reported:

      Recall (R) — "did we reproduce every real conformer?"
        For each CREST conformer (column g) take the *closest generated sample*
        (min down axis 0), then average/threshold over CREST conformers.
        Penalizes MISSING modes: a real conformer with no nearby sample inflates
        AMR-R. Measures coverage of the ground-truth ensemble. A model that
        collapses to one conformer scores poorly here even if that conformer is
        perfect. Needs enough samples (we draw K = 2·n_crest) to have a fair shot
        at every CREST mode.

      Precision (P) — "is everything we generated actually realistic?"
        For each generated sample (row k) take the *closest CREST conformer*
        (min across axis 1), then average/threshold over samples. This is the
        "avg RMSD of a sample to its nearest dataset representative" the task
        asked for. Penalizes SPURIOUS samples: a garbage conformer far from every
        CREST minimum inflates AMR-P. A model that emits one perfect conformer
        and nothing else scores great here (but terribly on recall).

    The asymmetry is the min axis: recall minimizes over samples (rows) and
    averages over ground truth; precision minimizes over ground truth (columns)
    and averages over samples. Coverage is the same min vectors, but reports the
    *fraction* within δ Å instead of the mean — robust to a few large outliers
    that would dominate the AMR average.
    """
    min_per_gt = M.min( axis=0 ) # recall:    for each CREST conf, the closest sample
    min_per_k = M.min( axis=1 )  # precision: for each sample, the closest CREST conf
    out = { "amr_r": float( min_per_gt.mean() ), "amr_p": float( min_per_k.mean() ) }
    for delta in COV_DELTAS:
        out[f"cov_r_{delta}"] = float( ( min_per_gt < delta ).mean() )   # frac of CREST confs with a sample within δ
        out[f"cov_p_{delta}"] = float( ( min_per_k  < delta ).mean() )   # frac of samples within δ of some CREST conf
    return out


# Relaxation (GFN2-xTB, parallel over a process pool)
def _relax_one( task: tuple ) -> tuple[int, int, np.ndarray | None, dict | None ]:
    """Worker: GFN2-xTB-relax one sample. Pass numpy arrays / plain scalars only
    (no torch tensors) so nothing crosses the Pool via torch's shared-memory reducer.

    Also returns relaxation diagnostics read off the L-BFGS history:
      init_force  max per-atom force of the RAW sample before relaxation (eV/Å) --
                  how strained the generated conformer is. Dominated by the RDKit
                  backbone / L-shift, which torsion diffusion can't fix.
      n_iter      number of L-BFGS steps taken (distance-to-minimum, in steps).
      converged   whether it reached FORCE_TOL_EV_A before the step cap.
    """
    mol_idx, k, Z_np, x0 = task
    tol_kJ = FORCE_TOL_EV_A / KJ_MOL_TO_EV   # relaxMolecule works in kJ/mol/Å
    try:
        pot = TBLitePotential( Z=Z_np, method=XTB_METHOD )
        x_relaxed, history = relaxMolecule( pot, x0, minimizer="lbfgs",
                                            force_tol=tol_kJ,
                                            returnOptimizationHistory=True )
    except Exception as e:
        print( f"  relax failed (mol {mol_idx}, sample {k}): {type(e).__name__}: {e}" )
        return ( mol_idx, k, None, None )
    diag = {
        "init_force": float( history[0]["max_force_rms"] ) * KJ_MOL_TO_EV,   # eV/Å
        "n_iter":     int( history[-1]["step"] ),
        "converged":  bool( history[-1]["max_force_rms"] < tol_kJ ),
    }
    return ( mol_idx, k, x_relaxed.astype( np.float32 ), diag )


def _cluster_representatives( samples: np.ndarray, Z, thresh: float ) -> list[int]:
    """Greedy heavy-atom Kabsch-RMSD clustering of the raw samples; return the
    representative index of each cluster (the sample that opened it).

    A sample within `thresh` A of an existing representative is absorbed (it will
    relax to the same minimum), so only representatives need relaxing. The `all(...)`
    short-circuits on the first close representative, so absorbed samples are cheap;
    cost is ~O(K + n_clusters^2), fine when the sampler produces few distinct basins.
    """
    reps: list[int] = []
    rep_geoms: list = []
    for k in range( samples.shape[0] ):
        xk = pt.tensor( samples[k], dtype=pt.float32 )
        if all( kabsch_aligned_heavy_rmsd( xk, rg, Z ) >= thresh for rg in rep_geoms ):
            reps.append( k )
            rep_geoms.append( xk )
    return reps


def relax_split( per_mol: list[dict] ) -> None:
    """Prune near-duplicate raw samples, then GFN2-xTB-relax only the cluster
    representatives (near-duplicates relax to the same minimum, so this is a large
    cost cut). Relaxed metrics are computed over the DEDUPLICATED set of relaxed
    representatives -- the distinct conformers you would actually output. Recall
    (Cov-R / AMR-R) is unchanged by dedup (dropped duplicates were not new basins);
    precision is now over the distinct conformers. Attaches metrics in place.

    NB: the raw (pre-relaxation) metrics in evaluate_run are still over ALL samples;
    only the relaxed block uses the pruned representatives.
    """
    for rec in per_mol:
        rec["rep_idx"] = _cluster_representatives( rec["samples"], rec["Z"], PRUNE_RMSD )

    tasks = [
        ( i, r, rec["Z_np"], rec["samples"][r] )
        for i, rec in enumerate( per_mol )
        for r in rec["rep_idx"]
    ]
    if len( tasks ) == 0:
        return

    n_samples_total = sum( rec["samples"].shape[0] for rec in per_mol )
    n_clusters_total = len( tasks )   # one relaxation per cluster -> total cost for this split
    print( f"    pruned {n_samples_total:,} samples -> {n_clusters_total:,} clusters "
           f"= {n_clusters_total:,} relaxations "
           f"({n_samples_total / max(n_clusters_total,1):.1f}x fewer, "
           f"{n_clusters_total / len(per_mol):.1f} clusters/mol avg, RMSD < {PRUNE_RMSD} A)" )

    relaxed: dict[tuple[int, int], np.ndarray] = {}
    diags: dict[tuple[int, int], dict] = {}
    t0 = time.time()
    ctx = mp.get_context( "spawn" )
    with ctx.Pool( processes=N_RELAX_WORKERS ) as pool:
        for done, ( mol_idx, r, x_rel, diag ) in enumerate( pool.imap_unordered( _relax_one, tasks ), start=1 ):
            if x_rel is not None:
                relaxed[( mol_idx, r )] = x_rel
                diags[( mol_idx, r )] = diag
            if done % 500 == 0:
                print( f"    relaxed {done:,}/{len(tasks):,} ({done/(time.time()-t0):.1f}/s)" )

    for i, rec in enumerate( per_mol ):
        Z = rec["Z"]
        gt = rec["gt_confs"]
        reps = rec["rep_idx"]

        keep = [ r for r in reps if relaxed.get( (i, r) ) is not None ]
        if not keep:
            rec["relaxed"] = None
            continue

        relaxed_stack = np.stack( [ relaxed[( i, r )] for r in keep ] )         # (R', N, 3)
        M_rel = rmsd_matrix( relaxed_stack, gt, Z )                             # (R', n_gt)

        # How far each representative moved under relaxation (heavy-atom aligned).
        shifts = [
            kabsch_aligned_heavy_rmsd( pt.tensor( relaxed[( i, r )], dtype=pt.float32 ),
                                       pt.tensor( rec["samples"][r], dtype=pt.float32 ), Z )
            for r in keep
        ]
        min_per_gt = M_rel.min( axis=0 )
        min_per_k = M_rel.min( axis=1 )
        mol_diags = [ diags[( i, r )] for r in keep ]
        rec["relaxed"] = {
            "relaxed_amr_p":  float( min_per_k.mean() ),
            "relaxed_amr_r":  float( min_per_gt.mean() ),
            "relax_shift":    float( np.mean( shifts ) ),
            "init_force":     float( np.mean( [ d["init_force"] for d in mol_diags ] ) ),   # eV/Å, raw-sample strain
            "n_iter":         float( np.mean( [ d["n_iter"]     for d in mol_diags ] ) ),   # L-BFGS steps to relax
            "frac_converged": float( np.mean( [ d["converged"]  for d in mol_diags ] ) ),
            "prune_factor":   float( rec["samples"].shape[0] / max( len( reps ), 1 ) ),     # samples per relaxation
            "n_samples":      int( rec["samples"].shape[0] ),
            "n_clusters":     len( reps ),
            "n_relaxed":      len( keep ),
            "n_failed":       len( reps ) - len( keep ),
        }
        # Relaxed coverage: fraction of CREST confs (R) / samples (P) within δ AFTER
        # relaxation. With the backbone/L-shift removed, relaxed Cov-R is the clean
        # measure of how many real basins diffusion+relaxation actually recovers.
        for delta in COV_DELTAS:
            rec["relaxed"][f"relaxed_cov_r_{delta}"] = float( ( min_per_gt < delta ).mean() )
            rec["relaxed"][f"relaxed_cov_p_{delta}"] = float( ( min_per_k  < delta ).mean() )


# Aggregation
def aggregate( per_mol: list[dict] ) -> dict:
    """Mean/median across molecules of every per-molecule scalar metric."""
    def stat( vals: list[float] ) -> dict:
        a = np.asarray( vals, dtype=np.float64 )
        return { "mean": float( a.mean() ), "median": float( np.median( a ) ), "n": int( a.size ) }

    geom_keys = ["amr_r", "amr_p"] + [ f"cov_{d}_{t}" for t in COV_DELTAS for d in ("r", "p") ]
    agg = { k: stat( [ rec["geom"][k] for rec in per_mol ] ) for k in geom_keys }

    relaxed = [ rec["relaxed"] for rec in per_mol if rec.get("relaxed") is not None ]
    if relaxed:
        relaxed_keys = ( [ "relaxed_amr_p", "relaxed_amr_r", "relax_shift",
                           "init_force", "n_iter", "frac_converged", "prune_factor" ]
                         + [ f"relaxed_cov_{d}_{t}" for t in COV_DELTAS for d in ("r", "p") ] )
        for k in relaxed_keys:
            agg[k] = stat( [ r[k] for r in relaxed ] )
        agg["n_relaxed_total"] = int( sum( r["n_relaxed"] for r in relaxed ) )
        agg["n_failed_total"]  = int( sum( r["n_failed"]  for r in relaxed ) )
    return agg


# Molecule selection
def select_molecules( qm9_dir: Path, smiles_list: list[str], n_want: int ) -> list[tuple[str, TorsionalDiffusionData, object]]:
    """First `n_want` molecules in the split that have >=1 rotatable bond (the
    only ones a torsion sampler is defined on). Returns (smiles, MoleculeData,
    rd_mol) triples; reports how many were skipped."""
    out = []
    n_rigid = 0
    for smi in smiles_list:
        if len( out ) >= n_want:
            break
        pickle_path = qm9_dir / f"{smi}.pickle"
        if not pickle_path.exists():
            continue
        d = load_qm9_molecule( pickle_path )
        if d.rotatable_bonds.shape[0] == 0:
            n_rigid += 1
            continue
        with open( pickle_path, "rb" ) as f:
            rd_mol = pickle.load( f )["conformers"][0]["rd_mol"]
        out.append( ( smi, d, rd_mol ) )
    print( f"  selected {len(out)} molecules ( {n_rigid} rigid / 0-rotatable-bond skipped )" )
    return out


# Per-run driver
def evaluate_run( run: dict, splits_data: dict, qm9_dir: Path ) -> dict | None:
    ckpt = Path( run["ckpt"] )
    if not ckpt.exists():
        print( f"\n[{run['label']}] checkpoint {ckpt} not found — skipping (train it, then re-run)." )
        return None

    device = pt.device( DEVICE )
    model = load_model( ckpt, device )
    print( f"\n=== run '{run['label']}'  ({run['start']} start, "
           f"backbone_mult={run['backbone_mult']}, torsion_inits={run['torsion_inits']}) ===" )

    run_results: dict = {}
    for split in SPLITS:
        print( f"\n[{run['label']} / {split}]" )
        mols = select_molecules( qm9_dir, splits_data["splits"][split], N_PER_SPLIT )

        per_mol: list[dict] = []
        n_dropped = 0
        t0 = time.time()
        for smi, d, rd_mol in mols:
            print(smi)
            starts = build_starting_geometries( d, rd_mol, run["start"], run["backbone_mult"], run["torsion_inits"] )
            if starts is None:
                n_dropped += 1
                continue
            samples = sample_conformers( model, d, starts, device )
            print( '\tConformers generated' )
            M = rmsd_matrix( samples, d.conformers, d.Z )
            print( '\tRMSD done' )
            per_mol.append({
                "smiles": smi,
                "Z": d.Z,
                "Z_np": d.Z.numpy(),
                "gt_confs": d.conformers,
                "samples": samples,
                "geom": geom_metrics( M ),
            })
        print( f"  sampled {len(per_mol)} molecules in {time.time()-t0:.0f}s"
               + ( f" ({n_dropped} dropped: RDKit embed failed)" if n_dropped else "" ) )

        print( f"  relaxing samples with {XTB_METHOD} on {N_RELAX_WORKERS} workers..." )
        relax_split( per_mol )

        agg = aggregate( per_mol )
        run_results[split] = agg

        OUT_DIR.mkdir( exist_ok=True )
        with open( OUT_DIR / f"{run['label']}_{split}.json", "w" ) as f:
            json.dump( {
                "run": run, "split": split,
                "aggregate": agg,
                "per_molecule": [ { "smiles": r["smiles"], "geom": r["geom"], "relaxed": r.get("relaxed") } for r in per_mol ],
            }, f, indent=2 )

    return run_results


# Comparison table
def print_comparison( all_results: dict ) -> None:
    labels = [ lbl for lbl in all_results if all_results[lbl] is not None ]
    if not labels:
        return
    rows = ( ["amr_r", "amr_p"] + [ f"cov_{d}_{t}" for t in COV_DELTAS for d in ("r", "p") ]
             + ["relaxed_amr_r", "relaxed_amr_p", "relax_shift"]
             + [ f"relaxed_cov_{d}_{t}" for t in COV_DELTAS for d in ("r", "p") ]
             + ["init_force", "n_iter", "frac_converged", "prune_factor"] )

    print( "\n\n=================== COMPARISON (mean over molecules) ===================" )
    for split in SPLITS:
        print( f"\n--- {split} ---" )
        print( f"{'metric':<16}" + "".join( f"{lbl:>14}" for lbl in labels ) )
        for r in rows:
            cells = ""
            for lbl in labels:
                agg = all_results[lbl].get( split, {} )
                cells += f"{agg[r]['mean']:>14.3f}" if r in agg else f"{'-':>14}"
            print( f"{r:<16}" + cells )

    # Gap vs the baseline (default 'crest', the ceiling): metric[run] − metric[baseline].
    # Sign reads per metric: AMR / init_force / n_iter -> lower is better (positive gap =
    # worse); coverage / frac_converged -> higher is better (negative gap = worse).
    others = [ lbl for lbl in labels if lbl != BASELINE_LABEL ]
    if BASELINE_LABEL in labels and others:
        print( f"\n\n============ GAP vs '{BASELINE_LABEL}'  (metric[run] − metric[{BASELINE_LABEL}]) ============" )
        for split in SPLITS:
            print( f"\n--- {split} ---" )
            print( f"{'metric':<16}" + "".join( f"{lbl:>14}" for lbl in others ) )
            base = all_results[BASELINE_LABEL].get( split, {} )
            for r in rows:
                cells = ""
                for lbl in others:
                    agg = all_results[lbl].get( split, {} )
                    cells += ( f"{agg[r]['mean'] - base[r]['mean']:>+14.3f}"
                               if r in agg and r in base else f"{'-':>14}" )
                print( f"{r:<16}" + cells )


# Markdown report (renders cleanly in an editor / GitHub, unlike terminal columns)
_REPORT_GROUPS = [
    ( "Raw (pre-relaxation)",
      ["amr_r", "amr_p"] + [ f"cov_{d}_{t}" for t in COV_DELTAS for d in ("r", "p") ] ),
    ( "Relaxed (GFN2-xTB)",
      ["relaxed_amr_r", "relaxed_amr_p", "relax_shift"]
      + [ f"relaxed_cov_{d}_{t}" for t in COV_DELTAS for d in ("r", "p") ] ),
    ( "Relaxation cost / validity",
      ["init_force", "n_iter", "frac_converged", "prune_factor"] ),
]


def _md_table( header: list[str], rows: list[list[str]] ) -> str:
    """Assemble a GitHub-flavoured Markdown table (first column is the row label)."""
    out = [ "| " + " | ".join( header ) + " |",
            "| " + " | ".join( "---" for _ in header ) + " |" ]
    for cells in rows:
        out.append( "| " + " | ".join( cells ) + " |" )
    return "\n".join( out )


def write_report( all_results: dict, path: Path ) -> None:
    """Write the comparison + gap tables as a Markdown document -- far more
    legible than the terminal when there are many metrics."""
    labels = [ lbl for lbl in all_results if all_results[lbl] is not None ]
    if not labels:
        return
    others = [ lbl for lbl in labels if lbl != BASELINE_LABEL ]
    has_gap = BASELINE_LABEL in labels and bool( others )

    def val( lbl, split, r ):
        agg = all_results[lbl].get( split, {} )
        return f"{agg[r]['mean']:.3f}" if r in agg else "-"

    def gap( lbl, split, r, base ):
        agg = all_results[lbl].get( split, {} )
        return ( f"{agg[r]['mean'] - base[r]['mean']:+.3f}"
                 if r in agg and r in base else "-" )

    L = [
        f"# Torsional-diffusion eval - {time.strftime('%Y-%m-%d %H:%M')}",
        "",
        f"- **method** {XTB_METHOD} - force tol {FORCE_TOL_EV_A} eV/A - "
        f"{N_PER_SPLIT} mols/split - K = 2*n_crest samples/mol",
        f"- **runs** {', '.join( labels )} - gap baseline **{BASELINE_LABEL}**",
        "- units: AMR / relax_shift in A, init_force in eV/A, Cov & frac_converged in [0,1], "
        "n_iter = L-BFGS steps.",
        "- gap = `run - baseline`. Lower-is-better (AMR, init_force, n_iter): **+ = worse**. "
        "Higher-is-better (Cov, frac_converged): **- = worse**.",
        "",
    ]
    for split in SPLITS:
        L += [ f"## {split}", "" ]
        for gname, grows in _REPORT_GROUPS:
            present = [ r for r in grows
                        if any( r in all_results[lbl].get( split, {} ) for lbl in labels ) ]
            if not present:
                continue
            L += [ f"### {gname}", "",
                   _md_table( ["metric"] + labels,
                              [ [r] + [ val( lbl, split, r ) for lbl in labels ] for r in present ] ),
                   "" ]
            if has_gap:
                base = all_results[BASELINE_LABEL].get( split, {} )
                L += [ f"_Gap vs {BASELINE_LABEL}:_", "",
                       _md_table( ["metric"] + others,
                                  [ [r] + [ gap( lbl, split, r, base ) for lbl in others ] for r in present ] ),
                       "" ]

    path.parent.mkdir( parents=True, exist_ok=True )
    path.write_text( "\n".join( L ), encoding="utf-8" )
    print( f"\nWrote Markdown report -> {path}" )


def main() -> None:
    load_dotenv()
    pt.manual_seed( SEED )
    np.random.seed( SEED )

    with open( "./data_config.json" ) as f:
        data_config = json.load( f )
    qm9_dir    = Path( data_config["qm9_folder"] )
    parsed_dir = qm9_dir.parent / "parsed"
    with open( parsed_dir / "splits.json" ) as f:
        splits_data = json.load( f )

    all_results = { run["label"]: evaluate_run( run, splits_data, qm9_dir ) for run in RUNS }
    print_comparison( all_results )
    write_report( all_results, OUT_DIR / "report_1000.md" )


if __name__ == "__main__":
    main()
