"""
Does post-ML path smoothing speed up xTB-NEB, and what is the best alpha?

For each reaction in the transition1x dataset we:
  1. build the attention model's initial-guess path (once),
  2. run xTB-NEB on it unsmoothed  -> baseline,
  3. for every alpha in ALPHA_GRID, smooth the same guess with
     chemdm.path_smoothing.smooth_path_penalized_least_squares and run xTB-NEB,

and record, per run, the initial perpendicular NEB force and the number of NEB
iterations to converge. We then plot, as a function of alpha, the reduction in
initial perpendicular force and the reduction factor in iteration count
(baseline / smoothed). A reduction factor > 1 means smoothing helps.

Beyond the two requested curves we also track (see HEADER below for why):
  - convergence rate vs alpha     -- smoothing must not *break* convergence;
                                     iteration reductions are meaningless on
                                     runs that never reach the tolerance.
  - converged-path agreement      -- RMSD between the smoothed and baseline
    and barrier difference vs alpha  converged paths, to catch over-smoothing
                                     that sends NEB to a different / worse MEP.
The recommended alpha maximizes the iteration reduction subject to not hurting
the convergence rate and not distorting the converged barrier.

NEB settings match the production defaults in chemdm.commands.transition_path
(n_images=20, force_tol=0.1 kJ/mol/A, max 2500 steps), and the ML guess goes
through the same kabsch-align + methyl-cleanup pipeline.

Cost: one NEB per (reaction, alpha) plus a baseline, i.e. (1 + len(ALPHA_GRID))
xTB-NEB optimizations per reaction, each up to 2500 steps x 20 images.
N_REACTIONS caps how many reactions per split are evaluated. Everything is
recomputed on every run (no caching); the per-split PNG figure is the saved
artifact.
"""

import sys
import json
import os
import contextlib
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

EXAMPLES = Path(__file__).resolve().parent.parent
sys.path.append(str(EXAMPLES))

from chemdm.TransitionPathDataset import TransitionPathDataset
from chemdm.path_smoothing import smooth_path_penalized_least_squares
from chemdm.nebXtbDirect import run_neb_xtb
from chemdm.geometry import kabsch_align_numpy
from chemdm.util import formula_from_Z
from chemdm.Constants import KJ_MOL_PER_EV
from chemdm.commands.transition_path import load_attention_model, _ml_initial_guess, cleanupPath


SPLITS = ["train", "val", "test"]   # each evaluated and plotted separately
N_REACTIONS = 100                   # cap per split (None = whole split)
ALPHA_GRID = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]

# NEB settings (match chemdm.commands.transition_path.run defaults)
N_IMAGES = 20      # images along the path, incl. endpoints (production default)
FORCE_TOL = 5.0    # kJ/mol/A (~0.05 eV/A), meaningful NEB convergence criterion
N_STEPS = 2500     # hard cap on NEB iterations (production max_iterations)
LR = 1e-2
K_SPRING = 1.0 * KJ_MOL_PER_EV # kJ/mol/A^2  (= 1 eV/A^2)
MAX_STEP_A = 0.02
MAX_WORKERS = 12   # parallelize image xTB evals across processes


class _Recorder:
    """Captures NEB callbacks: callback(iter, maxF). Step 0 = initial perp force."""
    def __init__(self):
        self.calls = []
    def __call__(self, it, maxF):
        self.calls.append((int(it), float(maxF)))
    @property
    def init_force(self):
        return self.calls[0][1] if self.calls else float("nan")
    @property
    def n_iter(self):
        return self.calls[-1][0] if self.calls else -1


@contextlib.contextmanager
def _silence_output():
    """Suppress stdout+stderr at the OS fd level for the duration of an xTB-NEB
    run. fd-level (not contextlib.redirect_*) so spawned xTB workers, which
    inherit fds 1/2, are silenced too. Our own progress prints happen outside
    this block, so they are unaffected."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = [os.dup(1), os.dup(2)]
    sys.stdout.flush(); sys.stderr.flush()   # push prior buffered output to the real fds
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        # Flush while fds still point at devnull, so buffered library print()s
        # are discarded rather than escaping to stdout after the restore.
        sys.stdout.flush(); sys.stderr.flush()
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        os.close(devnull)
        os.close(saved[0])
        os.close(saved[1])


def run_neb( Z_np, path0 ):
    """One xTB-NEB run. Returns dict with init_force, n_iter, converged, barrier, path."""
    rec = _Recorder()
    with _silence_output():
        path_opt, E_opt, best_force = run_neb_xtb(
            Z_np, path0, N_STEPS, LR, K_SPRING, MAX_STEP_A, FORCE_TOL,
            max_workers=MAX_WORKERS, callback=rec,
        )
    return {
        "init_force": rec.init_force,
        "n_iter":     rec.n_iter,
        "converged":  bool(best_force <= FORCE_TOL),
        "barrier":    float((E_opt - E_opt[0]).max()),
        "path":       path_opt,
    }


def ml_guess( model, traj ):
    """Attention initial-guess path for one reaction, endpoints kabsch-aligned.

    Mirrors the production pipeline: kabsch-align endpoints, run the network, then
    methyl-cleanup for molecules with more than 6 heavy atoms.
    """
    Z = traj.Z.cpu().numpy().astype(np.int64)
    xA = traj.xA.cpu().numpy().astype(float)
    xB = traj.xB.cpu().numpy().astype(float)
    GA = traj.GA.cpu().numpy()
    GB = traj.GB.cpu().numpy()
    xB = kabsch_align_numpy( xB, xA, Z )
    path0, s0 = _ml_initial_guess( model, Z, xA, xB, GA, GB, N_IMAGES )
    if np.sum(Z != 1) > 6:
        path0 = cleanupPath( Z, path0, s0, GA, GB )
    return Z, path0


def evaluate(split, model):
    with open( EXAMPLES / "transition1x_attention" / "data_config.json" ) as f:
        data_dir = json.load(f)["data_folder"]
    dataset = TransitionPathDataset( split, data_dir )
    n = len(dataset) if N_REACTIONS is None else min(N_REACTIONS, len(dataset))
    na = len(ALPHA_GRID)
    print( f"{split}: evaluating {n} reactions x (1 baseline + {na} alphas) NEB runs" )

    # (reaction, alpha) for smoothed runs; (reaction,) for baseline.
    A = lambda: np.full((n, na), np.nan)
    R = {
        "alpha": np.array(ALPHA_GRID),
        "base_init_force": np.full(n, np.nan), "base_n_iter": np.full(n, np.nan),
        "base_converged": np.zeros(n, bool),   "base_barrier": np.full(n, np.nan),
        "init_force": A(), "n_iter": A(), "converged": np.zeros((n, na), bool),
        "barrier": A(), "path_rmsd_to_base": A(),
    }

    for i in range(n):
        traj = dataset[i][-1]
        formula = formula_from_Z(traj.Z)
        try:
            Z_np, path0 = ml_guess(model, traj)
            base = run_neb(Z_np, path0)
        except Exception as e:
            print(f"  [{i}/{n}] {formula}: baseline failed ({type(e).__name__}: {e}); skipping")
            continue

        R["base_init_force"][i] = base["init_force"]
        R["base_n_iter"][i] = base["n_iter"]
        R["base_converged"][i] = base["converged"]
        R["base_barrier"][i] = base["barrier"]

        for j, alpha in enumerate(ALPHA_GRID):
            try:
                sm = run_neb( Z_np, smooth_path_penalized_least_squares(path0, alpha) )
            except Exception:
                continue   # leaves NaN for this alpha; reflected in the conv count below
            R["init_force"][i, j] = sm["init_force"]
            R["n_iter"][i, j] = sm["n_iter"]
            R["converged"][i, j] = sm["converged"]
            R["barrier"][i, j] = sm["barrier"]
            if base["converged"] and sm["converged"]:
                R["path_rmsd_to_base"][i, j] = float( np.sqrt(((sm["path"] - base["path"]) ** 2).sum(-1).mean()) )

        # One concise line per reaction: baseline + how the alpha sweep did.
        n_aconv = int(R["converged"][i].sum())
        best = "n/a"
        if base["converged"]:
            m = R["converged"][i] & np.isfinite(R["n_iter"][i])
            if m.any():
                best = f"{float(np.max(base['n_iter'] / R['n_iter'][i][m])):.2f}x"
        print(f"  [{i+1}/{n}] {formula:<10} base iters={base['n_iter']:>4} conv={int(base['converged'])} | "
              f"alpha conv={n_aconv}/{na} best_iter_red={best}", flush=True)

    return R


def _geomean(x):
    x = x[np.isfinite(x) & (x > 0)]
    return float(np.exp(np.mean(np.log(x)))) if len(x) else float("nan")


def report_and_plot(R, split):
    alpha = R["alpha"]; na = len(alpha)
    base_F, base_it = R["base_init_force"], R["base_n_iter"]
    base_conv = R["base_converged"]

    # Per-alpha aggregates.
    f_geo, f_lo, f_hi = (np.full(na, np.nan) for _ in range(3))
    it_geo, it_lo, it_hi = (np.full(na, np.nan) for _ in range(3))
    conv_rate = np.full(na, np.nan)
    barrier_err = np.full(na, np.nan)
    path_rmsd = np.full(na, np.nan)
    it_win = np.full(na, np.nan)
    it_pval = np.full(na, np.nan)

    print("\n=== Smoothing sweep summary ===")
    print(f"baseline convergence rate: {100*base_conv.mean():.1f}%  "
          f"(n={len(base_conv)})   median baseline iters (converged): "
          f"{np.nanmedian(base_it[base_conv]):.0f}")
    hdr = (f"  {'alpha':>6} {'F0_red(geo)':>12} {'iter_red(geo)':>13} "
           f"{'conv%':>7} {'iter_win%':>10} {'barrErr':>9} {'pathRMSD':>9}")
    print(hdr); print("  " + "-" * (len(hdr) - 2))

    for j in range(na):
        # Initial-force reduction: defined for every reaction (paired ratio).
        rF = base_F / R["init_force"][:, j]
        f_geo[j] = _geomean(rF)
        good = np.isfinite(rF) & (rF > 0)
        if good.sum():
            f_lo[j], f_hi[j] = np.percentile(rF[good], [25, 75])

        # Iteration reduction: only reactions where BOTH converged.
        both = base_conv & R["converged"][:, j] & np.isfinite(R["n_iter"][:, j])
        rIt = (base_it[both] / R["n_iter"][both, j]) if both.sum() else np.array([])
        it_geo[j] = _geomean(rIt)
        if len(rIt):
            it_lo[j], it_hi[j] = np.percentile(rIt[np.isfinite(rIt) & (rIt > 0)], [25, 75])
            it_win[j] = 100 * np.mean(rIt > 1)
            if both.sum() >= 6:
                try:
                    _, it_pval[j] = wilcoxon(base_it[both], R["n_iter"][both, j])
                except ValueError:
                    pass

        conv_rate[j] = 100 * R["converged"][:, j].mean()
        barrier_err[j] = np.nanmedian(np.abs(R["barrier"][:, j] - R["base_barrier"]))
        path_rmsd[j] = np.nanmedian(R["path_rmsd_to_base"][:, j])

        print(f"  {alpha[j]:>6} {f_geo[j]:>12.2f} {it_geo[j]:>13.2f} "
              f"{conv_rate[j]:>6.0f}% {it_win[j]:>9.0f}% {barrier_err[j]:>9.2f} {path_rmsd[j]:>9.3f}")

    # Recommended alpha: best iteration reduction that doesn't hurt convergence
    # (>= baseline) and keeps the barrier within 5 kJ/mol of baseline.
    ok = (conv_rate >= 100 * base_conv.mean() - 1e-9) & (np.nan_to_num(barrier_err, nan=1e9) < 5.0)
    cand = np.where(ok & np.isfinite(it_geo), it_geo, -np.inf)
    if np.any(np.isfinite(cand) & (cand > -np.inf)):
        best = int(np.argmax(cand))
        print(f"\nRecommended alpha = {alpha[best]} "
              f"(iter reduction {it_geo[best]:.2f}x, conv {conv_rate[best]:.0f}%, "
              f"barrier err {barrier_err[best]:.2f} kJ/mol)")

    # --- Plots ---
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))
    xlim = (alpha.min() * 0.8, alpha.max() * 1.25)   # anchors log axis even if a series is all-NaN

    # geomean as the line, IQR as a shaded band. (Not errorbar arms: the geomean
    # can fall outside [p25, p75] for skewed ratios, which would make an arm
    # negative and errorbar reject it.)
    a = ax[0, 0]
    a.fill_between(alpha, f_lo, f_hi, color="tab:blue", alpha=0.2, label="IQR")
    a.plot(alpha, f_geo, marker="o", color="tab:blue", label="geomean")
    a.axhline(1.0, color="k", lw=1, ls="--")
    a.set_xscale("log"); a.set_xlim(*xlim); a.set_xlabel("alpha"); a.set_ylabel("F0 reduction")
    a.set_title("Initial perpendicular-force reduction (baseline / smoothed)")
    a.legend(); a.grid(alpha=0.3)

    a = ax[0, 1]
    a.fill_between(alpha, it_lo, it_hi, color="tab:green", alpha=0.2, label="IQR")
    a.plot(alpha, it_geo, marker="o", color="tab:green", label="geomean")
    a.axhline(1.0, color="k", lw=1, ls="--")
    a.set_xscale("log"); a.set_xlim(*xlim); a.set_xlabel("alpha"); a.set_ylabel("iteration reduction")
    a.set_title("NEB iteration reduction (baseline / smoothed)")
    a.legend(); a.grid(alpha=0.3)

    a = ax[1, 0]
    a.plot(alpha, conv_rate, marker="o", label="smoothed")
    a.axhline(100 * base_conv.mean(), color="k", lw=1, ls="--", label="baseline")
    a.set_xscale("log"); a.set_xlim(*xlim); a.set_xlabel("alpha"); a.set_ylabel("convergence rate [%]")
    a.set_title("Convergence rate (must not drop below baseline)")
    a.legend(); a.grid(alpha=0.3)

    a = ax[1, 1]
    a.plot(alpha, path_rmsd, marker="o", color="tab:red", label="path RMSD vs baseline [A]")
    a.set_xscale("log"); a.set_xlim(*xlim); a.set_xlabel("alpha"); a.set_ylabel("converged path RMSD [A]")
    a2 = a.twinx()
    a2.plot(alpha, barrier_err, marker="s", color="tab:purple", label="barrier err [kJ/mol]")
    a2.set_ylabel("barrier abs error [kJ/mol]")
    a.set_title("Over-smoothing guardrail (converged result distortion)")
    a.grid(alpha=0.3)

    fig.suptitle(f"Post-ML smoothing alpha sweep — {split} (n={len(base_conv)})")
    fig.tight_layout()


def main():
    model = load_attention_model()
    for split in SPLITS:
        print(f"\n################  {split}  ################")
        R = evaluate(split, model)
        report_and_plot(R, split)
    plt.show()


if __name__ == "__main__":
    main()
