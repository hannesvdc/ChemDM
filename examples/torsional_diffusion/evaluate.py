"""
Evaluate a trained torsional-diffusion model on the test split.

Orchestrator: spawns one `eval_worker.py` subprocess per chunk of B_MOL
molecules and aggregates their RMSD matrices into the AMR / Coverage
metrics from the torsional-diffusion paper (Table 1).

One-subprocess-per-chunk dodges Apple's MPSGraph compilation cache, which
grows unboundedly under variable-shape ops (see feedback_mps_gotchas.md #5).

Per chunk, the worker:
    1. ETKDGv3-embeds each molecule from its QM9 pickle for L̂.
    2. Samples K conformers via the reverse-SDE sampler.
    3. Kabsch-aligns each sample to each ground-truth conformer on heavy
       atoms and writes a `(K, n_gt)` RMSD matrix per molecule.

Metrics (aggregated here):
    AMR-R / AMR-P              mean min-RMSD (recall / precision)
    Coverage-R(δ) / Coverage-P fraction within δ Å (recall / precision)
    at δ ∈ {0.5, 0.75, 1.0, 1.25} Å.

Run
---
    python evaluate.py [rdkit|crest]

    rdkit (default) — paper inference protocol, starts each sample from a
                      fresh ETKDGv3 embed.
    crest           — starts from the molecule's ground-truth conformer.
                      Numbers are optimistic; useful for isolating the
                      score-net from the RDKit-vs-CREST shift in L.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch as pt


B_MOL = 8                       # must match eval_worker.py
COVERAGE_THRESHOLDS_ANG = (0.5, 0.75, 1.0, 1.25)


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
    cov_r = {d: [] for d in COVERAGE_THRESHOLDS_ANG}
    cov_p = {d: [] for d in COVERAGE_THRESHOLDS_ANG}

    for M in rmsd_matrices:
        if M.size == 0:
            continue
        min_per_gt = M.min(axis=0)
        min_per_k  = M.min(axis=1)

        amr_r.append( float(min_per_gt.mean()) )
        amr_p.append( float(min_per_k .mean()) )

        for d in COVERAGE_THRESHOLDS_ANG:
            cov_r[d].append( float((min_per_gt < d).mean()) )
            cov_p[d].append( float((min_per_k  < d).mean()) )

    def summary(vs: list[float]) -> dict:
        a = np.asarray(vs)
        return {
            "mean":   float(a.mean()),
            "median": float(np.median(a)),
            "n":      len(vs),
        }

    return {
        "AMR-R":      summary(amr_r),
        "AMR-P":      summary(amr_p),
        "Coverage-R": {d: summary(cov_r[d]) for d in COVERAGE_THRESHOLDS_ANG},
        "Coverage-P": {d: summary(cov_p[d]) for d in COVERAGE_THRESHOLDS_ANG},
    }


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


if __name__ == "__main__":
    main()
