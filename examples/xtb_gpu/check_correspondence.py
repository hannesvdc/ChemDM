"""
Correspondence check: tblite and dxtb vs the xtb-python reference (XTBPotential).

Verifies that the candidate backends reproduce the energies and forces of the
production XTBPotential, to within SCF/backend noise. Both contenders are
compared against the same xtb-python reference:
    - tblite  (the maintained xTB library; the migration target) -- always run
    - dxtb    (PyTorch autograd) -- run only where it is available (CUDA/Linux),
              since dxtb GFN2 needs the Linux-only tad-libcint.

This script is method- and device-agnostic (see the constants below):
    - macOS: METHOD="GFN1-xTB" or "GFN2-xTB" (both work via tblite 0.7.0);
      dxtb is auto-skipped.
    - cluster (Linux): either method, DEVICE="cpu"/"cuda"; full three-way.

For each geometry, per contender, we report vs the xtb-python reference:
    - absolute energy difference
    - forces: max abs error, RMSE, cosine alignment
Note: where xtb-python carries its analytical-gradient defect (some symmetric
geometries; see reference_gradient_bug.py), tblite/dxtb will *disagree* with the
reference's forces on purpose -- that is a fix, not a regression.
"""

from __future__ import annotations

import os
import pickle
import platform
import subprocess
import sys
import tempfile

import numpy as np

from molecules import get_molecules


def _dxtb_available() -> bool:
    """dxtb runs only where it fully works: Linux (GFN2 needs Linux-only
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
if RUN_DXTB:
    import torch as pt          # for DTYPE below; potentials are built in workers

METHOD = "GFN1-xTB"        # "GFN1-xTB" or "GFN2-xTB" (both work on macOS via tblite)
DEVICE = "cpu"             # "cpu", "cuda", "mps" (mps forces f32); dxtb only
DTYPE = pt.float64 if RUN_DXTB else None   # double precision; dxtb only
CHARGE = 0
UHF = 0
ETEMP = 300.0
MAXITER = 250
ACCURACY = 1.0

EV_TO_HARTREE = 1.0 / 27.211386024367243


_WORKER = os.path.join( os.path.dirname(os.path.abspath(__file__)), "subproc_worker.py" )


def _eval_backend( backend : str, kwargs : dict, geometries : list ):
    """Evaluate ALL geometries for ONE backend in a single fresh top-level
    process (subprocess.run of subproc_worker.py -- NOT multiprocessing, which
    deadlocks tblite's vendored libomp). The worker imports only this one
    backend, so xtb (conda libomp) and tblite (vendored libomp) never share a
    process. Returns a list of (energy_eV, forces_eV_A)."""
    with tempfile.TemporaryDirectory() as td:
        in_path = os.path.join(td, "in.pkl")
        out_path = os.path.join(td, "out.pkl")
        with open(in_path, "wb") as fh:
            pickle.dump( { "backend": backend, "kwargs": kwargs,
                           "geometries": [ (np.asarray(Z), np.asarray(x)) for Z, x in geometries ] }, fh )
        subprocess.run( [sys.executable, _WORKER, in_path, out_path], check=True )
        with open(out_path, "rb") as fh:
            return pickle.load(fh)


def _stats( e_ref, f_ref, e_c, f_c, name, n ):
    de = e_c - e_ref
    df = f_c - f_ref
    fmax = float(np.abs(df).max())
    frmse = float(np.sqrt(np.mean(df**2)))
    denom = np.linalg.norm(f_c) * np.linalg.norm(f_ref)
    cos = float(np.dot(f_c.ravel(), f_ref.ravel()) / denom) if denom > 0 else 1.0
    return {
        "name": name,
        "n": n,
        "de_eV": de,
        "de_Ha": de * EV_TO_HARTREE,
        "f_max_err": fmax,
        "f_rmse": frmse,
        "f_cos": cos,
    }


def main():
    codes = "tblite" + (", dxtb" if RUN_DXTB else " (dxtb skipped: CUDA/Linux only)")
    print(f"# Correspondence vs xtb-python reference | method={METHOD} "
          f"device={DEVICE} dtype={DTYPE}")
    print("# each evaluation runs in its own spawned process (OpenMP isolation)")
    print(f"# contenders: {codes}\n")

    ref_kwargs = dict( charge=CHARGE, uhf=UHF, method=METHOD,
                      accuracy=ACCURACY, electronic_temperature=ETEMP,
                      max_iterations=MAXITER )
    dx_kwargs = dict(**ref_kwargs, device=DEVICE, dtype=DTYPE)

    mols = get_molecules()
    geometries = [ (Z, x_A) for _, Z, x_A in mols ]
    meta = [ (name, len(np.asarray(Z))) for name, Z, _ in mols ]

    # One subprocess per backend (each runs all geometries) -> 2-3 processes total.
    xtb_res = _eval_backend( "xtb", ref_kwargs, geometries )
    res_by_code = { "tblite": _eval_backend( "tblite", ref_kwargs, geometries ) }
    if RUN_DXTB:
        res_by_code["dxtb"] = _eval_backend( "dxtb", dx_kwargs, geometries )

    labels = ["tblite"] + (["dxtb"] if RUN_DXTB else [])
    rows = []
    for i, (name, n) in enumerate(meta):
        e_ref, f_ref = xtb_res[i]
        for label in labels:
            e_c, f_c = res_by_code[label][i]
            r = _stats( e_ref, f_ref, e_c, f_c, name, n )
            r["code"] = label
            rows.append(r)

    # ---- per-molecule table ---------------------------------------------
    hdr = (f"{'molecule':17s} {'code':7s} {'n':>3s}  {'dE [eV]':>11s} {'dE [Ha]':>11s}  "
           f"{'Fmax_err':>10s} {'F_rmse':>10s} {'F_cos':>12s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['name']:17s} {r['code']:7s} {r['n']:3d}  {r['de_eV']:11.2e} {r['de_Ha']:11.2e}  "
              f"{r['f_max_err']:10.2e} {r['f_rmse']:10.2e} {r['f_cos']:12.9f}")

    # ---- aggregate per contender ----------------------------------------
    print("\n# Aggregate vs xtb-python reference")
    for label in ["tblite"] + (["dxtb"] if RUN_DXTB else []):
        sub = [r for r in rows if r["code"] == label]
        if not sub:
            continue
        de_eV = np.array([r["de_eV"] for r in sub])
        fmax = np.array([r["f_max_err"] for r in sub])
        cos = np.array([r["f_cos"] for r in sub])
        print(f"  [{label}] over {len(sub)} geometries")
        print(f"    |dE| eV   : mean {np.abs(de_eV).mean():.2e}  max {np.abs(de_eV).max():.2e}")
        print(f"    |dE| Ha   : mean {np.abs(de_eV*EV_TO_HARTREE).mean():.2e}  "
              f"max {np.abs(de_eV*EV_TO_HARTREE).max():.2e}")
        print(f"    F max err : mean {fmax.mean():.2e}  max {fmax.max():.2e}  eV/A")
        print(f"    worst cos : {cos.min():.9f}")


if __name__ == "__main__":
    main()
