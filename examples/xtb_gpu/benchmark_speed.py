"""
Speed testing script: dxtb-batched (GPU) vs xtb-python multiprocess, swept over
BOTH molecule size and batch size, for GFN2-xTB (libcint).

The crossover is a surface over two axes, not a single curve:
    * molecule size  -- bigger molecules have a larger SCF (the GPU-batched part),
                        so GPU has more room to beat the CPU baseline.
    * batch size B    -- more molecules per call amortize GPU launch / transfer cost.
This script sweeps MOLECULES x BATCH_SIZES and reports, per molecule, the smallest
B at which dxtb-GPU-batched beats the xtb-python multiprocess baseline.

Modes:
    * xtb-mp: xtb-python across spawned worker processes -- the baseline ChemDM
                 runs now (ProcessPoolExecutor + spawn, one XTBPotential/worker).
    * dxtb(<dev>) : dxtb, all B packed into one batched call, per device.
    * (opt-in) dxtb-mp : dxtb NON-batched across CPU worker processes.
    * (opt-in) xtb-seq : single-process xtb loop, per-core reference.

GFN2 requires libcint (dipole/quadrupole integrals); dxtb uses it by default and
we do NOT override the integral driver. NB from the dxtb source: libcint runs on
CPU and is looped per-molecule, so on CUDA the integral construction is CPU-bound
with device transfers while only the SCF batches on GPU -- this is exactly the
CPU<->GPU overhead the crossover has to overcome. Timing captures it end-to-end
(force_cpu_for_libcint transfers + cuda.synchronize()).

Timing excludes calculator construction, pool startup, and a warm-up call, so we
measure steady-state throughput (pools are built once per molecule and reused).

No CLI flags: edit the constants below. Plots are shown interactively (not saved).

How this file is organized (top to bottom):
    * Configuration constants -- what to sweep, which devices, how many repeats.
    * _timeit / make_batch    -- timing helper and batch-of-geometries builder.
    * The evaluation modes, each a small function:
        _mp_* + xtb_multiprocess       -- xtb across worker processes   (baseline)
        _dxtb_mp_* + dxtb_multiprocess -- dxtb non-batched across procs  (opt-in)
        xtb_sequential                 -- single-process xtb loop        (opt-in)
        dxtb_batched                   -- dxtb, one batched call/device   (contender)
    * pooled_sweep   -- time one mode at each batch size, one B at a time.
    * sweep_molecule -- run all enabled modes for one molecule.
    * main           -- loop molecules; print tables + crossover summary + plot.

Call flow: main -> (per molecule) sweep_molecule -> pooled_sweep / dxtb_batched -> _timeit.
"""

from __future__ import annotations

import os
import gc
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch as pt
from tad_mctc.batch import pack
from ase.units import Bohr

import dxtb
from molecules import build_geometry, SUITE
from chemdm.xtbSetup import XTBPotential

from typing import Callable

dxtb.OutputHandler.verbosity = 0

# Configuration
METHOD = "GFN2-xTB"                     # the production method (needs libcint)
# Molecule-size axis: names from molecules.SUITE, small -> large.
MOLECULES = ["thiophene", "benzene", "caffeine", "cholesterol", "C60-alkane"]
# Batch-size axis. Push high enough to find the crossover; watch GPU memory.
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256]
MAX_WORKERS = os.cpu_count()           # spawned xtb worker processes (ChemDM uses cpu_count)
# Batched dxtb device(s) with dtype. On the cluster CUDA is the point.
DXTB_DEVICES = [("cuda:p", pt.float64), ("cpu", pt.float64)]
RUN_XTB_SEQ = False                    # single-process xtb reference (context)
RUN_DXTB_MP = False                    # non-batched dxtb across CPU procs (context)
N_REPEAT = 3                           # take the min wall-time over repeats
SHOW_PLOT = True

_PARAM = { "GFN1-xTB": dxtb.GFN1_XTB, "GFN2-xTB": dxtb.GFN2_XTB }


def _free_resources():
    """Release memory between measurements so each starts from a clean state.

    Runs Python garbage collection and returns any cached GPU blocks to the driver,
    so one measurement doesn't clog resources for the next (which would skew the
    comparison). Worker pools are already shut down by pooled_sweep's `with` block
    before this is called, so no extra process cleanup is needed here.
    """
    gc.collect()
    if pt.cuda.is_available():
        pt.cuda.synchronize()
        pt.cuda.empty_cache()


def _timeit( fn : Callable,
             n_repeat : int = N_REPEAT ):
    """Wall-time of calling `fn`, taken as the MIN over `n_repeat` runs.

    The first call is a throw-away warm-up (JIT/compile, memory allocation, cache
    fill, worker spin-up) and is not timed. We take the minimum, not the mean,
    because it is the least-noisy estimate of the true cost: background load can
    only ever make a run slower, never faster.
    """
    fn( )  # warm-up (compilation, allocation, caches, worker init)
    best = float("inf")
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        fn( )
        best = min(best, time.perf_counter() - t0)
    return best


def make_batch( x0 : np.ndarray, B : int, seed : int = 0 ):
    """B distinct geometries (batch size B): x0 + small rattle. Distinct positions
    are required so the xtb-python baselines can't return ASE-cached results."""
    rng = np.random.default_rng( seed )
    return [ x0 + rng.normal(scale=0.02, size=x0.shape) for _ in range(B) ]


# --- Mode: xtb across spawned worker processes (what ChemDM runs now) -------
# ProcessPoolExecutor pattern: each worker process builds ONE XTBPotential in an
# initializer (_mp_init) and stashes it in a module-global (_MP_POT). Later calls
# reuse it, so we don't rebuild the (expensive) calculator every evaluation. We use
# the `spawn` start method because xtb has global state that is unsafe to `fork`.
# Geometries are then handed to workers with pool.map.
_MP_POT = None
def _mp_init( Z : np.ndarray,
              method : str ):
    """Runs once inside each spawned worker; builds that worker's XTBPotential."""
    global _MP_POT
    from chemdm.xtbSetup import XTBPotential
    _MP_POT = XTBPotential( Z=np.asarray(Z, dtype=int), method=method )


def _mp_eval( x ):
    _MP_POT.atoms.calc.reset()          # clear ASE cache for fair repeat timing
    return _MP_POT.energy_forces(x)


def xtb_multiprocess( pool, xs ):
    """Time evaluating all geometries `xs` by farming them across worker `pool`."""
    def run():
        list(pool.map(_mp_eval, xs))   # distribute the B geometries over the workers
    return _timeit(run)


# --- Mode: dxtb NON-batched, one molecule per call, across CPU processes -----
# Same worker-pool idea as above, but each worker runs dxtb on a single molecule.
# This isolates "parallelism" from "batching": comparing it to dxtb_batched shows
# whether packing molecules together helps beyond just using many cores.
_DXTB_POT = None
def _dxtb_mp_init( Z : np.ndarray,
                   method : str ):
    """Runs once per worker; builds a CPU DxtbPotential pinned to 1 BLAS thread.

    We want P worker processes x 1 thread each = P cores. If each worker's linear
    algebra also spawned its own thread pool, P workers would oversubscribe the
    cores and thrash (~150x slower), so we cap threads to 1 per worker.
    """
    global _DXTB_POT
    import torch as pt
    pt.set_num_threads(1)
    from dxtb_potential import DxtbPotential
    _DXTB_POT = DxtbPotential( Z=np.asarray(Z, dtype=int), method=method,
                              device="cpu", dtype=pt.float64 )


def _dxtb_mp_eval( x ):
    return _DXTB_POT.energy_forces( x )


def dxtb_multiprocess( pool, xs ):
    """Time evaluating `xs` with one-dxtb-call-per-molecule across the worker pool."""
    def run():
        list(pool.map(_dxtb_mp_eval, xs))
    return _timeit(run)


# Mode: single-process xtb-python loop (per-core reference)
def xtb_python_sequential( Z : np.ndarray,
                           xs : list ):
    """Time evaluating `xs` one at a time in a single process (no parallelism).

    Reference point: xtb_multiprocess divided by this gives the parallel speedup.
    """
    pot = XTBPotential( Z=Z, method=METHOD )

    def run():
        for x in xs:
            pot.atoms.calc.reset()   # clear ASE cache so identical repeats recompute
            pot.energy_forces(x)
    return _timeit(run)


# Mode: dxtb batched on one device (the main contender for ChemDM)
def dxtb_batched( Z : np.ndarray,
                  xs : np.ndarray,
                  device : pt.device,
                  dtype : pt.dtype ):
    """Time one batched dxtb energy+forces call over all B geometries on `device`.

    All B molecules share the same Z (homogeneous batch), so they pack with no
    padding. `pack` stacks the per-molecule tensors into one batched tensor of
    shape (B, ...); dxtb then runs a single batched SCF and we get forces via
    autograd (one backward pass over the whole batch).
    """
    dev = pt.device( device )
    # Batched inputs: numbers (B, n_atoms), positions (B, n_atoms, 3) in Bohr.
    # dxtb works in atomic units, so convert the Angstrom geometries with / Bohr.
    numbers = pack( [pt.tensor(Z, dtype=pt.long, device=dev) ] * len(xs))
    positions = pack( [pt.tensor(np.asarray(x) / Bohr, dtype=dtype, device=dev ) for x in xs])

    # batch_mode=1 tells dxtb the leading dimension is a batch of independent systems.
    calc = dxtb.Calculator( numbers, _PARAM[METHOD],
                            opts={"batch_mode": 1, "verbosity": 0},
                            device=dev, dtype=dtype )
    chrg = pt.zeros( len(xs), 1, dtype=dtype, device=dev )  # per-molecule charge; (B,1) works for all B

    def run():
        calc.reset()                                  # drop tensors cached from the previous call
        pos = positions.clone().requires_grad_(True)  # track grad so autograd can produce forces
        e = calc.get_energy( pos, chrg=chrg )           # (B,) energies from one batched SCF
        (g,) = pt.autograd.grad( e.sum(), pos )         # dE/dpos (= -forces) for all B at once
        # GPU kernels are launched asynchronously, so wait for them to actually
        # finish before we stop the clock -- otherwise we'd time only the launch.
        if device == "cuda:0":
            pt.cuda.synchronize()
        elif device == "mps":
            pt.mps.synchronize()
        return g
    return _timeit( run )


# One full batch-size sweep behind a spawned worker pool
_THREAD_KEYS = ( "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS" )


def pooled_sweep( Z, x0, batches, n_workers, ctx, initializer, warm_eval, time_fn,
                 single_thread=False ):
    """Time one mode at each batch size, ONE BATCH SIZE AT A TIME; return [time per B].

    Batch sizes are stepped through sequentially (never concurrently). For each B,
    this measures how long the mode takes to evaluate a single batch of B molecules.
    For the xtb-mp mode, "evaluate a batch of B" means distributing those B molecules
    across the worker pool -- i.e. multi-process xtb-python on CPU. That is the CPU
    side of the core question: for a given B, is it faster to batch the B molecules
    into one dxtb-GPU call, or to multi-process them across CPU cores with xtb?

    A fresh worker pool is built (via `initializer`), warmed once (untimed) so the
    workers finish importing/constructing, then reused for every B; it is torn down
    on exit of the `with` block (which reaps the workers). `single_thread=True` pins
    BLAS to 1 thread/worker (for the dxtb-mp mode; see below).
    """
    # Cap BLAS/LAPACK to 1 thread/worker for dxtb workers (P procs x 1 thread);
    # env vars must be set before torch imports in the spawned child, else each
    # worker's Accelerate/LAPACK oversubscribes.
    saved = { k: os.environ.get(k) for k in _THREAD_KEYS } if single_thread else {}
    if single_thread:
        os.environ.update( {k: "1" for k in _THREAD_KEYS} )
    try:
        with ProcessPoolExecutor( max_workers=n_workers, mp_context=ctx,
                                 initializer=initializer, initargs=(Z, METHOD) ) as pool:
            list(pool.map(warm_eval, make_batch(x0, n_workers)))  # init/warm workers
            return [time_fn(pool, batches[B]) for B in BATCH_SIZES]
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


def sweep_molecule(name, Z, x0, mp_col, dxtb_mp_col, dxtb_cols, devices,
                   n_workers, ctx):
    """Run every enabled mode for one molecule over all batch sizes.

    Returns {column_name: [wall-time for each B in BATCH_SIZES]}. Always includes the
    xtb-mp baseline and the batched dxtb device columns; the dxtb-mp and xtb-seq
    context columns are added only if RUN_DXTB_MP / RUN_XTB_SEQ are set. A device that
    errors (e.g. CUDA out-of-memory at large B) is recorded as NaN so the sweep keeps
    going instead of crashing.
    """
    batches = {B: make_batch(x0, B) for B in BATCH_SIZES}   # geometries, reused by all modes
    res = {}

    # Modes run STRICTLY ONE AT A TIME, freeing resources in between, so each is
    # timed on an otherwise-idle machine (no leftover worker processes or cached GPU
    # memory skewing the comparison). pooled_sweep tears down its pool on exit.
    res[mp_col] = pooled_sweep(Z, x0, batches, n_workers, ctx,
                               _mp_init, _mp_eval, xtb_multiprocess)
    _free_resources()

    if RUN_DXTB_MP:
        res[dxtb_mp_col] = pooled_sweep(Z, x0, batches, n_workers, ctx,
                                        _dxtb_mp_init, _dxtb_mp_eval,
                                        dxtb_multiprocess, single_thread=True)
        _free_resources()

    if RUN_XTB_SEQ:
        res["xtb-seq(1p)"] = [xtb_python_sequential(Z, batches[B]) for B in BATCH_SIZES]
        _free_resources()

    # Batched dxtb per device: time each batch size individually and free GPU memory
    # after every B. Doing them one-by-one (not one big comprehension) means an
    # out-of-memory at large B only NaNs that single point, leaving smaller-B results.
    for (d, t), c in zip(devices, dxtb_cols):
        times = []
        for B in BATCH_SIZES:
            try:
                times.append(dxtb_batched(Z, batches[B], d, t))
            except Exception as exc:  # noqa: BLE001  (e.g. CUDA OOM at large B)
                print(f"  !! {name} {c} B={B}: {type(exc).__name__}: {exc}")
                times.append(float("nan"))
            _free_resources()   # return memory before the next (larger) batch
        res[c] = times
    return res


def main():
    """Sweep every molecule x batch size, then print tables + crossover summary + plot."""
    n_workers = MAX_WORKERS or 1

    # Keep only the requested dxtb devices that actually exist on this machine
    # (e.g. drop CUDA on a laptop). Each DXTB_DEVICES entry is (device_string, dtype).
    devices = []
    for name, dtype in DXTB_DEVICES:
        ok = (name == "cpu"
              or (name == "cuda" and pt.cuda.is_available())
              or (name == "mps" and pt.backends.mps.is_available()))
        (devices.append((name, dtype)) if ok
         else print(f"# (skipping dxtb device {name!r}: unavailable)"))

    # Column labels for the output tables. `cols` is the ordered set of columns
    # actually shown: baseline first, then any opt-in context modes, then dxtb devices.
    mp_col = f"xtb-mp({n_workers}p)"
    dxtb_mp_col = f"dxtb-mp({n_workers}p)"
    dxtb_cols = [f"dxtb({d},{str(t).split('.')[-1]})" for d, t in devices]
    cols = [mp_col]
    if RUN_DXTB_MP:
        cols.append(dxtb_mp_col)
    if RUN_XTB_SEQ:
        cols.append("xtb-seq(1p)")
    cols += dxtb_cols

    print(f"# 2-D speed sweep: {METHOD}, energy+forces, min over {N_REPEAT} reps\n"
          f"# baseline = {mp_col}; batched dxtb devices = {[d for d,_ in devices]}\n"
          f"# molecule-size x batch-size; construction/startup/warm-up excluded\n")

    ctx = mp.get_context("spawn")   # xtb needs spawn (fork is unsafe with its global state)
    geoms = {name: build_geometry(dict(SUITE)[name]) for name in MOLECULES}  # SMILES -> (Z, coords)
    all_results = {}                # all_results[molecule][column] = [time per B]

    # ---- run the sweep + print one timing table per molecule ------------
    for name in MOLECULES:
        Z, x0 = geoms[name]
        res = sweep_molecule(name, Z, x0, mp_col, dxtb_mp_col, dxtb_cols,
                             devices, n_workers, ctx)
        all_results[name] = res
        print(f"## {name}  (n_atoms={len(Z)})")
        hdr = f"{'B':>5s}  " + "  ".join(f"{c:>20s}" for c in cols)
        print(hdr); print("-" * len(hdr))
        for i, B in enumerate(BATCH_SIZES):
            print(f"{B:5d}  " + "  ".join(f"{res[c][i]*1e3:17.1f} ms" for c in cols))
        print()
        _free_resources()   # clean slate before the next molecule

    # ---- crossover summary: per molecule, per dxtb device --------------
    print(f"# Crossover vs {mp_col}: smallest B where dxtb-batched wins, and peak speedup")
    hdr = f"{'molecule':16s} {'n':>4s}  " + "  ".join(
        f"{c + ' Bwin/peak':>28s}" for c in dxtb_cols)
    print(hdr); print("-" * len(hdr))
    for name in MOLECULES:
        Z, _ = geoms[name]
        base = np.array(all_results[name][mp_col])      # baseline (xtb-mp) time per B
        cells = []
        for c in dxtb_cols:
            r = base / np.array(all_results[name][c])   # speedup ratio; >1 => dxtb faster
            wins = [BATCH_SIZES[i] for i in range(len(BATCH_SIZES)) if r[i] > 1]
            peak = np.nanmax(r) if np.isfinite(r).any() else float("nan")   # best speedup seen
            bwin = f"B={wins[0]}" if wins else "never"   # smallest batch that beats the baseline
            cells.append(f"{bwin:>10s} / {peak:5.2f}x")
        print(f"{name:16s} {len(Z):4d}  " + "  ".join(f"{x:>28s}" for x in cells))

    # ---- plot: speedup vs batch size, one line per molecule ------------
    if SHOW_PLOT and dxtb_cols:
        try:
            import matplotlib.pyplot as plt
            primary = dxtb_cols[0]   # plot the first dxtb device (cuda if present)
            for name in MOLECULES:
                base = np.array(all_results[name][mp_col])
                r = base / np.array(all_results[name][primary])
                plt.plot(BATCH_SIZES, r, "o-", label=f"{name} ({len(geoms[name][0])})")
            plt.axhline(1.0, color="k", ls="--", lw=1, label="parity (xtb-mp)")
            plt.xlabel("batch size B"); plt.ylabel(f"speedup {primary} vs {mp_col}")
            plt.xscale("log", base=2); plt.yscale("log")
            plt.title(f"{METHOD}: dxtb-batched speedup over xtb multiprocess")
            plt.legend(); plt.grid(True, which="both", alpha=0.3); plt.show()
        except Exception as exc:  # noqa: BLE001
            print(f"(plot skipped: {exc})")


if __name__ == "__main__":
    main()
