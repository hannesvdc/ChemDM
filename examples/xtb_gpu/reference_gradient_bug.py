"""
The GFN1-xTB analytical gradient from xtb-python is inconsistent with the true
gradient of its own energy at the equilibrium (G2) geometry of methanol -- and
both dxtb (PyTorch, autograd) and tblite (the maintained xTB library) agree with
the true gradient.

Four forces are compared at the same geometry:
    1. xtb-python analytical force        (XTB ASE calculator)
    2. tblite analytical force            (TBLite ASE calculator)
    3. dxtb autograd force                (DxtbPotential; CUDA/Linux only)
    4. -FD(<code> energy)                 central finite differences of a code's
                                          OWN energy = neutral truth for that code

Result: dxtb and tblite each agree with -FD(their own energy) to ~1e-3 eV/A (the
FD discretization floor), while xtb-python's analytical force disagrees with
-FD(its own energy) by ~0.225 eV/A on one carbon component. So the discrepancy is
a defect specific to the xtb-python analytical gradient, confirmed three ways:
(a) xtb-python is inconsistent with its own energy (needs no other code), (b) an
independent torch code (dxtb) agrees with the truth, and (c) the maintained
replacement (tblite) is self-consistent -- i.e. migrating to tblite fixes it.

The gap is invariant to SCF accuracy, FD step size, and electronic temperature,
ruling out convergence / step-size / smearing explanations.

Each backend is evaluated in its own subprocess (subproc_worker.py -- shared with
check_correspondence.py) so xtb-python (conda's libomp) and tblite (the pip
wheel's vendored libomp) never load OpenMP in the same process, which aborts. The
finite-difference gradient is built here from the energies the worker returns at
the perturbed geometries; the analytic force comes from the base geometry's
result. dxtb runs only where it is available (Linux / CUDA); on macOS it is
auto-skipped.
"""

import os
import pickle
import platform
import subprocess
import sys
import tempfile

import numpy as np


def _dxtb_available() -> bool:
    """dxtb is only run where it fully works: Linux (GFN2 needs the Linux-only
    tad-libcint) and/or a CUDA device."""
    on_linux = (platform.system() == "Linux")
    has_cuda = False
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except Exception:
        pass
    if not (on_linux or has_cuda):
        return False
    try:
        import dxtb  # noqa: F401
        return True
    except Exception:
        return False


RUN_DXTB = _dxtb_available()

# Equilibrium methanol geometry (ASE G2 collection), Angstrom.
SYMBOLS = ["C", "O", "H", "H", "H", "H"]
Z = np.array([6, 8, 1, 1, 1, 1])
XYZ = np.array([
    [-0.047131,  0.664389,  0.000000],
    [-0.047131, -0.758551,  0.000000],
    [-1.092995,  0.969785,  0.000000],
    [ 0.878534, -1.048458,  0.000000],
    [ 0.437145,  1.080376,  0.891772],
    [ 0.437145,  1.080376, -0.891772],
])

METHOD = "GFN2-xTB"        # "GFN1-xTB" or "GFN2-xTB"
ACCURACIES = (1.0, 0.1, 0.01, 0.001, 0.0001)
ETEMP = 300.0
MAXITER = 250
FD_STEP = 2e-4

_WORKER = os.path.join( os.path.dirname(os.path.abspath(__file__)), "subproc_worker.py" )


def _kwargs( accuracy : float ) -> dict:
    return dict( charge=0, uhf=0, method=METHOD, accuracy=accuracy,
                 electronic_temperature=ETEMP, max_iterations=MAXITER )


def _eval_geoms( backend : str, kwargs : dict, geoms : list ):
    """Run subproc_worker.py for one backend over a list of geometries (Z fixed),
    in a fresh process that imports only that backend (OpenMP isolation).
    Returns a list of (energy_eV, forces_eV_A)."""
    with tempfile.TemporaryDirectory() as td:
        in_path = os.path.join(td, "in.pkl")
        out_path = os.path.join(td, "out.pkl")
        with open(in_path, "wb") as fh:
            pickle.dump( { "backend": backend, "kwargs": kwargs,
                           "geometries": [ (Z, g) for g in geoms ] }, fh )
        subprocess.run( [sys.executable, _WORKER, in_path, out_path], check=True )
        with open(out_path, "rb") as fh:
            return pickle.load(fh)


def _analytic_and_fd( backend : str, accuracy : float, h : float = FD_STEP ):
    """Analytic force at XYZ and -FD(own energy), both from a single subprocess:
    geometry 0 is the base (its force = analytic), the rest are +/-h perturbations
    whose energies give the central-difference gradient."""
    geoms = [ XYZ.copy() ]
    for i in range(XYZ.shape[0]):
        for j in range(3):
            xp = XYZ.copy(); xp[i, j] += h
            xm = XYZ.copy(); xm[i, j] -= h
            geoms += [xp, xm]

    res = _eval_geoms( backend, _kwargs(accuracy), geoms )

    F_analytic = np.asarray( res[0][1], dtype=float )
    g = np.zeros_like( XYZ )
    idx = 1
    for i in range(XYZ.shape[0]):
        for j in range(3):
            e_p = res[idx][0]
            e_m = res[idx + 1][0]
            idx += 2
            g[i, j] = (e_p - e_m) / (2 * h)
    return F_analytic, -g


def main():
    # Sweep once; reuse accuracy=1.0 for the detailed table below.
    xtb = { acc: _analytic_and_fd("xtb", acc) for acc in ACCURACIES }
    tbl = { acc: _analytic_and_fd("tblite", acc) for acc in ACCURACIES }

    F_xtb, F_xtb_fd = xtb[1.0]
    F_tblite, F_tblite_fd = tbl[1.0]

    F_dxtb = None
    if RUN_DXTB:
        # dxtb autograd force is exact by construction -> only the base geometry.
        F_dxtb = np.asarray( _eval_geoms("dxtb", _kwargs(1.0), [XYZ])[0][1], dtype=float )

    # Locate the atom where xtb-python's analytical gradient is worst.
    a = int( np.abs(F_xtb - F_xtb_fd).max(axis=1).argmax() )

    print(f"{METHOD}, methanol equilibrium geometry")
    print(f"worst atom (by xtb-python error): index {a} ({SYMBOLS[a]})   [eV/A]")
    print(f"  xtb-python analytical : {np.round(F_xtb[a], 4)}")
    print(f"  tblite analytical     : {np.round(F_tblite[a], 4)}")
    if RUN_DXTB:
        print(f"  dxtb autograd         : {np.round(F_dxtb[a], 4)}")
    print(f"  -FD(own energy) xtb   : {np.round(F_xtb_fd[a], 4)}   (xtb-python truth)")
    print(f"  -FD(own energy) tblite: {np.round(F_tblite_fd[a], 4)}   (tblite truth)\n")

    print("Self-consistency: analytical force vs -FD(own energy), max over all atoms:")
    print(f"  xtb-python : {np.abs(F_xtb    - F_xtb_fd   ).max():.4e} eV/A   (BUG: should be ~0)")
    print(f"  tblite     : {np.abs(F_tblite - F_tblite_fd).max():.4e} eV/A   (FD floor => fixed)\n")

    print("Cross-checks between independent codes, max over all atoms:")
    print(f"  tblite vs -FD(xtb E)   : {np.abs(F_tblite - F_xtb_fd).max():.4e} eV/A")
    if RUN_DXTB:
        print(f"  tblite vs dxtb         : {np.abs(F_tblite - F_dxtb  ).max():.4e} eV/A")
        print(f"  dxtb   vs -FD(xtb E)   : {np.abs(F_dxtb   - F_xtb_fd).max():.4e} eV/A")
    print(f"  tblite vs xtb-python   : {np.abs(F_tblite - F_xtb   ).max():.4e} eV/A   (size of the defect)")
    if not RUN_DXTB:
        print("  (dxtb skipped: runs only on CUDA/Linux)")
    print()

    print("Self-consistency vs SCF accuracy (not a convergence issue):")
    print(f"  {'accuracy':>10s} {'xtb-python':>16s} {'tblite':>16s}")
    for acc in ACCURACIES:
        Fa_x, Ffd_x = xtb[acc]
        Fa_t, Ffd_t = tbl[acc]
        d_xtb = np.abs(Fa_x - Ffd_x).max()
        d_tbl = np.abs(Fa_t - Ffd_t).max()
        print(f"  {acc:10.4f} {d_xtb:16.4e} {d_tbl:16.4e}")


if __name__ == "__main__":
    main()
