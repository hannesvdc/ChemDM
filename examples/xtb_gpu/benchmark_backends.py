"""
Per-call CPU timing: xtb-python vs tblite, single-point energy + forces.

Both backends run behind the same `energy_forces` interface; production evaluates
them in a spawn ProcessPool of single-threaded workers, so the decision-relevant
number is per-call latency at one thread -- which is what this measures. Each
backend runs in its OWN subprocess (OpenMP isolation, see subproc_worker.py) with
OMP/BLAS pinned to 1 thread. Reports best-of-REPS wall time per molecule, warm-up
excluded.
"""

import os
import pickle
import subprocess
import sys
import tempfile

import numpy as np

from molecules import get_molecules

METHOD = "GFN2-xTB"        # production default; also try "GFN1-xTB"
REPS = 10
ETEMP = 300.0
MAXITER = 250
ACCURACY = 1.0

_WORKER = os.path.join( os.path.dirname(os.path.abspath(__file__)), "time_worker.py" )
_ONE_THREAD = { k: "1" for k in ( "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                  "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                                  "NUMEXPR_NUM_THREADS" ) }


def _time_backend( backend, kwargs, geometries ):
    env = dict(os.environ); env.update(_ONE_THREAD)
    with tempfile.TemporaryDirectory() as td:
        in_path = os.path.join(td, "in.pkl")
        out_path = os.path.join(td, "out.pkl")
        with open(in_path, "wb") as fh:
            pickle.dump( { "backend": backend, "kwargs": kwargs, "reps": REPS,
                           "geometries": [ (np.asarray(Z), np.asarray(x)) for Z, x in geometries ] }, fh )
        subprocess.run( [sys.executable, _WORKER, in_path, out_path], check=True, env=env )
        with open(out_path, "rb") as fh:
            return pickle.load(fh)


def main():
    kwargs = dict( charge=0, uhf=0, method=METHOD, accuracy=ACCURACY,
                   electronic_temperature=ETEMP, max_iterations=MAXITER )
    mols = get_molecules()
    geometries = [ (Z, x) for _, Z, x in mols ]
    meta = [ (name, len(np.asarray(Z))) for name, Z, _ in mols ]

    print(f"# xtb-python vs tblite per-call timing | method={METHOD} | "
          f"single-threaded | best of {REPS} (warm-up excluded)\n")

    t_xtb = _time_backend( "xtb", kwargs, geometries )
    t_tbl = _time_backend( "tblite", kwargs, geometries )

    hdr = f"{'molecule':17s} {'n':>4s} {'xtb-python':>12s} {'tblite':>12s} {'tblite/xtb':>11s}"
    print(hdr)
    print("-" * len(hdr))
    ratios = []
    for (name, n), tx, tt in zip(meta, t_xtb, t_tbl):
        r = tt / tx if tx > 0 else float("nan")
        ratios.append(r)
        print(f"{name:17s} {n:4d} {tx*1e3:10.2f}ms {tt*1e3:10.2f}ms {r:10.2f}x")

    gm = float(np.exp(np.mean(np.log(ratios))))
    print(f"\n# geometric-mean tblite/xtb ratio: {gm:.2f}x  (<1 => tblite faster)")


if __name__ == "__main__":
    main()
