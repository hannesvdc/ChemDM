"""
Evaluate a trained torsional-diffusion model on the test split.

Orchestrator: spawns one `eval_worker.py` subprocess per chunk of B_MOL
molecules and aggregates their RMSD matrices into the AMR / Coverage
metrics from the torsional-diffusion paper (Table 1), plus K-dependence
curves and per-SMILES distribution percentiles.

One-subprocess-per-chunk dodges Apple's MPSGraph compilation cache, which
grows unboundedly under variable-shape ops (see feedback_mps_gotchas.md #5).

Per chunk, the worker:
    1. ETKDGv3-embeds each molecule from its QM9 pickle for L̂.
    2. Samples K conformers via the reverse-SDE sampler.
    3. Kabsch-aligns each sample to each ground-truth conformer on heavy
       atoms and writes a `(K, n_gt)` RMSD matrix per molecule.

Metrics aggregated here:
    AMR-R / AMR-P              mean min-RMSD (recall / precision)
    Coverage-R(δ) / Coverage-P fraction within δ Å (recall / precision)
                               at δ ∈ {0.5, 0.75, 1.0, 1.25} Å
    AMR / Cov vs K'            sub-sampled K' ∈ 1..K_max — shows
                               where the diminishing-returns kink lands
    Per-SMILES percentiles     {p10, p25, p50, p75, p90} of each scalar
                               per-molecule metric — exposes the tail

Run
---
    python evaluate.py [rdkit|crest]

    rdkit (default) — paper inference protocol, starts each sample from a
                      fresh ETKDGv3 embed.
    crest           — starts from the molecule's ground-truth conformer.
                      Numbers are optimistic; useful for isolating the
                      score-net from the RDKit-vs-CREST shift in L.

For post-hoc re-analysis of an existing run without re-sampling, import
`run_analysis` from this module on the cached `eval_chunks/*.pt` files.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch as pt


B_MOL = 8                       # must match eval_worker.py
COVERAGE_THRESHOLDS_ANG = (0.5, 0.75, 1.0, 1.25)
PERCENTILES = (10, 25, 50, 75, 90)

# Precomputed (D, 1) array for broadcasting against per-molecule min-RMSD
# vectors in per_molecule_metrics / aggregate. One numpy op instead of a
# 4-iter Python loop per molecule.
_DELTAS_COL = np.asarray(COVERAGE_THRESHOLDS_ANG, dtype=np.float64).reshape(-1, 1)


# Scalar summary (the paper's headline numbers)
def aggregate( rmsd_matrices: list[np.ndarray] ) -> dict:
    """
    AMR (recall, precision) and Coverage (recall, precision) at multiple δ.

    Per molecule:
        AMR-R_i  = mean over gt of min over k of rmsd[k, g]
        AMR-P_i  = mean over k  of min over g of rmsd[k, g]
        Cov-R_i(δ) = mean over gt of [min over k of rmsd[k, g] < δ]
        Cov-P_i(δ) = mean over k  of [min over g of rmsd[k, g] < δ]
    """
    amr_r, amr_p = [], []
    cov_r_rows: list[np.ndarray] = []   # each (D,)
    cov_p_rows: list[np.ndarray] = []

    for M in rmsd_matrices:
        if M.size == 0:
            continue
        min_per_gt = M.min(axis=0)
        min_per_k  = M.min(axis=1)

        amr_r.append( float(min_per_gt.mean()) )
        amr_p.append( float(min_per_k .mean()) )

        # Broadcast (D, 1) < (n_gt,) and (D, 1) < (K,) → (D, ·); mean over last axis.
        cov_r_rows.append( (min_per_gt[None, :] < _DELTAS_COL).mean(axis=1) )
        cov_p_rows.append( (min_per_k [None, :] < _DELTAS_COL).mean(axis=1) )

    D = len(COVERAGE_THRESHOLDS_ANG)
    cov_r_mat = np.stack(cov_r_rows) if cov_r_rows else np.empty((0, D))   # (n_mols, D)
    cov_p_mat = np.stack(cov_p_rows) if cov_p_rows else np.empty((0, D))

    def summary_arr(a: np.ndarray) -> dict:
        return {
            "mean":   float(a.mean()),
            "median": float(np.median(a)),
            "n":      int(a.size),
        }

    return {
        "AMR-R":      summary_arr(np.asarray(amr_r)),
        "AMR-P":      summary_arr(np.asarray(amr_p)),
        "Coverage-R": {d: summary_arr(cov_r_mat[:, i]) for i, d in enumerate(COVERAGE_THRESHOLDS_ANG)},
        "Coverage-P": {d: summary_arr(cov_p_mat[:, i]) for i, d in enumerate(COVERAGE_THRESHOLDS_ANG)},
    }


# K-dependence + per-SMILES distribution
def per_molecule_metrics( M: np.ndarray, k_sub: np.ndarray ) -> dict:
    """
    M : (K, n_gt) RMSD matrix for one molecule
    k_sub   : indices (k_prime,) into the K axis — a subset of size k_prime ≤ K

    Returns per-molecule metrics on the subset. `cov_r` / `cov_p` are (D,)
    arrays aligned with COVERAGE_THRESHOLDS_ANG; the per-δ axis stays
    vectorised through every consumer downstream.
    """
    Msub = M[k_sub, :]
    min_per_gt = Msub.min(axis=0)
    min_per_k = Msub.min(axis=1)
    return {
        "amr_r": float(min_per_gt.mean()),
        "amr_p": float(min_per_k.mean()),
        # Broadcast (D, 1) < (n_gt,) and (D, 1) < (k_prime,) → (D, ·); mean over last axis.
        "cov_r": (min_per_gt[None, :] < _DELTAS_COL).mean(axis=1),   # (D,)
        "cov_p": (min_per_k [None, :] < _DELTAS_COL).mean(axis=1),   # (D,)
    }


def k_curve( rmsd_matrices: list[np.ndarray] ) -> dict:
    """
    For each K' from 1 to K_max, compute per-molecule metrics using the first
    K' samples (samples are i.i.d. within a molecule, so the prefix is an
    unbiased estimator of the expected metric at K' samples). Aggregate
    mean/median across molecules at each K' and discard the per-molecule
    arrays before moving to the next K' — peak memory is one K's worth, not
    K_max worth.

    K_max is the smallest K across all molecules (so every molecule contributes
    to every point on the curve).
    """
    valid = [M for M in rmsd_matrices if M.size > 0]
    if not valid:
        raise ValueError("no non-empty RMSD matrices to analyze")
    K_max = int(min(M.shape[0] for M in valid))
    print(f"  K_max (common across molecules) = {K_max}")
    print(f"  molecules with data             = {len(valid):,}")

    out: dict = {
        "K_max": K_max,
        "n_molecules": len(valid),
        "K_grid": list(range(1, K_max + 1)),
        "metrics": {},
    }

    for k_prime in range(1, K_max + 1):
        print(f"K = {k_prime}")
        idx = np.arange(k_prime)

        amr_r_list: list[float]      = []
        amr_p_list: list[float]      = []
        cov_r_rows: list[np.ndarray] = []   # each (D,)
        cov_p_rows: list[np.ndarray] = []

        for M in valid:
            vals = per_molecule_metrics(M, idx)
            amr_r_list.append(vals["amr_r"])
            amr_p_list.append(vals["amr_p"])
            cov_r_rows.append(vals["cov_r"])
            cov_p_rows.append(vals["cov_p"])

        amr_r_arr = np.asarray(amr_r_list)
        amr_p_arr = np.asarray(amr_p_list)
        cov_r_mat = np.stack(cov_r_rows)      # (n_mols, D)
        cov_p_mat = np.stack(cov_p_rows)

        def stat(a: np.ndarray) -> dict:
            return {"mean": float(a.mean()), "median": float(np.median(a))}

        metrics_k = {
            "amr_r": stat(amr_r_arr),
            "amr_p": stat(amr_p_arr),
        }
        for i, d in enumerate(COVERAGE_THRESHOLDS_ANG):
            metrics_k[f"cov_r_{d}"] = stat(cov_r_mat[:, i])
            metrics_k[f"cov_p_{d}"] = stat(cov_p_mat[:, i])
        out["metrics"][k_prime] = metrics_k

    return out


def smiles_distribution_percentiles( rmsd_matrices: list[np.ndarray] ) -> dict:
    """
    For each scalar metric at full K, report percentiles of the per-molecule
    distribution. Shows the tail the mean hides. Independently recomputes
    from rmsd_matrices rather than reusing k_curve's intermediates — keeps
    peak memory bounded to one K's worth of per-molecule data at a time.
    """
    valid = [M for M in rmsd_matrices if M.size > 0]
    amr_r_list: list[float] = []
    amr_p_list: list[float] = []
    cov_r_rows: list[np.ndarray] = []   # each (D,)
    cov_p_rows: list[np.ndarray] = []
    for M in valid:
        vals = per_molecule_metrics(M, np.arange(M.shape[0]))
        amr_r_list.append(vals["amr_r"])
        amr_p_list.append(vals["amr_p"])
        cov_r_rows.append(vals["cov_r"])
        cov_p_rows.append(vals["cov_p"])

    D = len(COVERAGE_THRESHOLDS_ANG)
    cov_r_mat = np.stack(cov_r_rows) if cov_r_rows else np.empty((0, D))
    cov_p_mat = np.stack(cov_p_rows) if cov_p_rows else np.empty((0, D))

    def stat(a: np.ndarray) -> dict:
        return {
            "mean":   float(a.mean()),
            "median": float(np.median(a)),
            **{f"p{p}": float(np.percentile(a, p)) for p in PERCENTILES},
        }

    out = {
        "amr_r": stat(np.asarray(amr_r_list)),
        "amr_p": stat(np.asarray(amr_p_list)),
    }
    for i, d in enumerate(COVERAGE_THRESHOLDS_ANG):
        out[f"cov_r_{d}"] = stat(cov_r_mat[:, i])
        out[f"cov_p_{d}"] = stat(cov_p_mat[:, i])
    return out


def plot_k_curves( kc: dict, mode: str ) -> None:
    """Render the AMR + Coverage vs K' plot and show it interactively."""
    K_grid = kc["K_grid"]

    def series(key: str, stat: str = "mean") -> list[float]:
        return [kc["metrics"][k][key][stat] for k in K_grid]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel 1: AMR-R / AMR-P vs K'
    ax = axes[0]
    ax.plot(K_grid, series("amr_r"), marker="o", label="AMR-R (mean)")
    ax.plot(K_grid, series("amr_p"), marker="s", label="AMR-P (mean)")
    ax.plot(K_grid, series("amr_r", "median"), marker="o", linestyle=":", alpha=0.6, label="AMR-R (median)")
    ax.plot(K_grid, series("amr_p", "median"), marker="s", linestyle=":", alpha=0.6, label="AMR-P (median)")
    ax.set_xlabel("K' (sub-sampled diffusion samples)")
    ax.set_ylabel("AMR (Å)")
    ax.set_title(f"AMR vs K  ({mode}, n={kc['n_molecules']:,} mols)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Panel 2: Coverage-R/P vs K' at each δ
    ax = axes[1]
    cmap = plt.get_cmap("tab10")
    for i, d in enumerate(COVERAGE_THRESHOLDS_ANG):
        c = cmap(i)
        ax.plot(K_grid, [v * 100 for v in series(f"cov_r_{d}")], marker="o", color=c, label=f"Cov-R @ {d}Å")
        ax.plot(K_grid, [v * 100 for v in series(f"cov_p_{d}")], marker="s", color=c, linestyle="--", alpha=0.7, label=f"Cov-P @ {d}Å")
    ax.set_xlabel("K' (sub-sampled diffusion samples)")
    ax.set_ylabel("Coverage (%)")
    ax.set_title(f"Coverage vs K  ({mode})")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)

    fig.tight_layout()
    plt.show()


def run_analysis( rmsd_matrices: list[np.ndarray],
                  mode: str,
                  save_dir: Path | None = None,
                  ) -> dict:
    """
    Run the K-dependence and SMILES-percentile analyses on already-loaded RMSD
    matrices. If `save_dir` is given, write `{mode}_analysis.json` there (JSON
    only — no figure saving). If `show_plot` is True, also pop up the K-curve
    plot interactively. Returns the assembled results dict.
    """
    print("\n=== K-dependence ===")
    kc = k_curve( rmsd_matrices )

    print("\n=== Distribution over SMILES (full K) ===")
    perc = smiles_distribution_percentiles( rmsd_matrices )

    print(f"\n  AMR-R   mean={perc['amr_r']['mean']:.3f}   "
          f"p10={perc['amr_r']['p10']:.3f}  p25={perc['amr_r']['p25']:.3f}  "
          f"p50={perc['amr_r']['p50']:.3f}  p75={perc['amr_r']['p75']:.3f}  "
          f"p90={perc['amr_r']['p90']:.3f}")
    print(f"  AMR-P   mean={perc['amr_p']['mean']:.3f}   "
          f"p10={perc['amr_p']['p10']:.3f}  p25={perc['amr_p']['p25']:.3f}  "
          f"p50={perc['amr_p']['p50']:.3f}  p75={perc['amr_p']['p75']:.3f}  "
          f"p90={perc['amr_p']['p90']:.3f}")
    for d in COVERAGE_THRESHOLDS_ANG:
        rd = perc[f"cov_r_{d}"]
        pd = perc[f"cov_p_{d}"]
        print(
            f"  δ={d:.2f}Å  Cov-R p10..p90 = "
            f"{rd['p10']*100:5.1f}  {rd['p25']*100:5.1f}  {rd['p50']*100:5.1f}  "
            f"{rd['p75']*100:5.1f}  {rd['p90']*100:5.1f}    "
            f"Cov-P p10..p90 = "
            f"{pd['p10']*100:5.1f}  {pd['p25']*100:5.1f}  {pd['p50']*100:5.1f}  "
            f"{pd['p75']*100:5.1f}  {pd['p90']*100:5.1f}"
        )

    results = {
        "mode": mode,
        "n_molecules": kc["n_molecules"],
        "K_max": kc["K_max"],
        "k_curve": kc,
        "smiles_percentiles": perc,
    }

    if save_dir is not None:
        out_json = save_dir / f"{mode}_analysis.json"
        with open(out_json, "w") as f:
            json.dump( results, f, indent=2 )
        print( f"  wrote {out_json}" )
    plot_k_curves( kc, mode )

    return results


# ============================================================
# Entry point
# ============================================================

def main():
    MODE = sys.argv[1] if len(sys.argv) > 1 else "rdkit"
    if MODE not in ("rdkit", "crest"):
        raise ValueError(f"unknown mode {MODE!r}; expected 'rdkit' or 'crest'")

    with open("./data_config.json", "r") as f:
        data_config = json.load(f)
    qm9_dir = Path( data_config["qm9_folder"] )
    parsed_dir = qm9_dir.parent / "parsed"
    out_dir = Path( "./eval_chunks" )
    worker = Path(__file__).parent / "eval_worker.py"

    data = pt.load(parsed_dir / "test.pt", weights_only=False)
    n_mols_total = len(data["mol_Z"])
    n_chunks = (n_mols_total + B_MOL - 1) // B_MOL
    out_dir.mkdir(exist_ok=True)
    for f in out_dir.glob(f"{MODE}_chunk_*.pt"):
        f.unlink()

    print(f"evaluating {n_mols_total:,} molecules across {n_chunks} subprocess chunks of {B_MOL}   (mode={MODE})")
    t0 = time.time()
    for idx in range(n_chunks):
        subprocess.run([sys.executable, str(worker), str(idx), MODE], check=True)
        done = min((idx + 1) * B_MOL, n_mols_total)
        dt   = time.time() - t0
        print(f"  chunk {idx+1}/{n_chunks}  ({done/max(dt,1e-9):.2f} mol/s)")

    rmsd_matrices: list[np.ndarray] = []
    for idx in range(n_chunks):
        rmsd_matrices.extend(pt.load(out_dir / f"{MODE}_chunk_{idx}.pt", weights_only=False))

    print()
    summary = aggregate(rmsd_matrices)
    print("=" * 60)
    print(f"results on {summary['AMR-R']['n']:,} molecules   (mode={MODE}):")
    print("=" * 60)
    print(f"  AMR-R   mean={summary['AMR-R']['mean']:.3f} Å   median={summary['AMR-R']['median']:.3f} Å")
    print(f"  AMR-P   mean={summary['AMR-P']['mean']:.3f} Å   median={summary['AMR-P']['median']:.3f} Å")
    print()
    for d in COVERAGE_THRESHOLDS_ANG:
        r = summary["Coverage-R"][d]
        p = summary["Coverage-P"][d]
        print(
            f"  δ = {d:.2f} Å    "
            f"Coverage-R mean={r['mean']*100:5.1f}%  median={r['median']*100:5.1f}%    "
            f"Coverage-P mean={p['mean']*100:5.1f}%  median={p['median']*100:5.1f}%"
        )

    # K-dependence curves + SMILES-distribution percentiles. JSON gets written
    # to eval_chunks/{mode}_analysis.json; the K-curve plot pops up at the very
    # end so the blocking plt.show() doesn't gate anything else.
    run_analysis(rmsd_matrices, MODE, save_dir=out_dir )


if __name__ == "__main__":
    main()
