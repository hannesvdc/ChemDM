"""Single-backend timing worker for benchmark_backends.py (invoked via
subprocess.run). Imports exactly one backend (OpenMP isolation), warms up, then
records the best-of-REPS wall time of energy_forces per geometry."""

import pickle
import sys
import time

import numpy as np


def _build( backend, kwargs, Z ):
    if backend == "xtb":
        from chemdm.xtbSetup import XTBPotential
        return XTBPotential( Z=Z, **kwargs )
    if backend == "tblite":
        from tblite_potential import TBLitePotential
        return TBLitePotential( Z=Z, **kwargs )
    raise ValueError( f"unknown backend {backend!r}" )


def main():
    with open(sys.argv[1], "rb") as fh:
        job = pickle.load(fh)

    backend = job["backend"]
    kwargs = job["kwargs"]
    reps = job["reps"]

    out = []
    for Z, x in job["geometries"]:
        Z = np.asarray(Z, dtype=int)
        x = np.asarray(x, dtype=float)
        pot = _build( backend, kwargs, Z )

        # Each timed rep must use a DISTINCT geometry: ASE caches results when the
        # positions are unchanged, so re-evaluating a geometry times a cache hit,
        # not an SCF. A tiny deterministic displacement per rep forces a real
        # recompute -- which is also what production does (every call is a new
        # geometry). Warm up on the UNPERTURBED base so it never coincides with a
        # timed geometry (otherwise best-of-min would pick that one cache hit).
        geoms = [ x + 1.0e-3 * np.sin(k + np.arange(x.size)).reshape(x.shape)
                  for k in range(reps) ]

        pot.energy_forces( x )                       # warm-up (allocation / first SCF)
        best = float("inf")
        for xk in geoms:
            t0 = time.perf_counter()
            pot.energy_forces( xk )
            best = min( best, time.perf_counter() - t0 )
        out.append(best)

    with open(sys.argv[2], "wb") as fh:
        pickle.dump(out, fh)


if __name__ == "__main__":
    main()
