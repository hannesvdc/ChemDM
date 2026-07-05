"""
Correspondence check: dxtb (PyTorch) vs xtb-python (tblite) reference.

Priority #1 of the xtb-on-GPU effort: verify that DxtbPotential reproduces the
energies and forces of the production XTBPotential, to within SCF/backend noise.

This script is method- and device-agnostic (see the constants below):
    - Stage 1 (local, macOS): METHOD="GFN1-xTB", DEVICE="cpu", f64
    - Stage 2 (cluster, Linux): METHOD="GFN2-xTB", DEVICE="cpu"/"cuda", f64

For each geometry we compare:
    - absolute energy            (dxtb vs ref)
    - forces: max abs error, RMSE, cosine alignment
As an independent self-consistency check (no dxtb involved), the reference's
analytical force can be compared against a finite-difference gradient of the
reference's own energy: F_analytic should equal -dE/dx.
"""

from __future__ import annotations

import numpy as np
import torch as pt

from dxtb_potential import DxtbPotential
from molecules import get_molecules
from chemdm.xtbSetup import XTBPotential
from chemdm.opt import EnergyForceEvaluator

METHOD = "GFN1-xTB"        # "GFN1-xTB" (macOS ok) or "GFN2-xTB" (Linux only)
DEVICE = "cpu"             # "cpu", "cuda", "mps" (mps forces f32)
DTYPE = pt.float64         # double precision for a meaningful correspondence test
CHARGE = 0
UHF = 0
ETEMP = 300.0
MAXITER = 250
ACCURACY = 1.0

EV_TO_HARTREE = 1.0 / 27.211386024367243


def _stats( dxtb_pot : EnergyForceEvaluator, 
            ref_pot : EnergyForceEvaluator, 
            name : str, 
            Z : np.ndarray, 
            x_A : np.ndarray ):
    e_ref, f_ref = ref_pot.energy_forces( x_A )
    e_dx, f_dx = dxtb_pot.energy_forces( x_A )

    de = e_dx - e_ref
    df = f_dx - f_ref
    fmax = float(np.abs(df).max())
    frmse = float(np.sqrt(np.mean(df**2)))
    denom = np.linalg.norm(f_dx) * np.linalg.norm(f_ref)
    cos = float(np.dot(f_dx.ravel(), f_ref.ravel()) / denom) if denom > 0 else 1.0
    return {
        "name": name,
        "n": len(Z),
        "e_ref": e_ref,
        "e_dx": e_dx,
        "de_eV": de,
        "de_Ha": de * EV_TO_HARTREE,
        "f_max_err": fmax,
        "f_rmse": frmse,
        "f_cos": cos,
    }


def main():
    print(f"# Correspondence: dxtb vs xtb-python | method={METHOD} "
          f"device={DEVICE} dtype={DTYPE}\n")

    ref_kwargs = dict( charge=CHARGE, uhf=UHF, method=METHOD,
                      accuracy=ACCURACY, electronic_temperature=ETEMP,
                      max_iterations=MAXITER )
    dx_kwargs = dict(**ref_kwargs, device=DEVICE, dtype=DTYPE)

    mols = get_molecules()

    rows = []
    for name, Z, x_A in mols:
        ref = XTBPotential( Z=Z, **ref_kwargs )
        dx = DxtbPotential( Z=Z, **dx_kwargs )
        try:
            r = _stats(dx, ref, name, Z, x_A)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"  !! {name}: {type(exc).__name__}: {exc}")
            continue
        rows.append(r)

    # ---- per-molecule table ---------------------------------------------
    hdr = (f"{'molecule':17s} {'n':>3s}  {'dE [eV]':>11s} {'dE [Ha]':>11s}  "
           f"{'Fmax_err':>10s} {'F_rmse':>10s} {'F_cos':>12s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['name']:17s} {r['n']:3d}  {r['de_eV']:11.2e} {r['de_Ha']:11.2e}  "
              f"{r['f_max_err']:10.2e} {r['f_rmse']:10.2e} {r['f_cos']:12.9f}")

    # ---- aggregate -------------------------------------------------------
    de_eV = np.array([r["de_eV"] for r in rows])
    fmax = np.array([r["f_max_err"] for r in rows])
    cos = np.array([r["f_cos"] for r in rows])
    print("\n# Aggregate over", len(rows), "geometries")
    print(f"  |dE| eV      : mean {np.abs(de_eV).mean():.2e}  max {np.abs(de_eV).max():.2e}")
    print(f"  |dE| Ha      : mean {np.abs(de_eV*EV_TO_HARTREE).mean():.2e}  "
          f"max {np.abs(de_eV*EV_TO_HARTREE).max():.2e}")
    print(f"  F max err    : mean {fmax.mean():.2e}  max {fmax.max():.2e}  eV/A")
    print(f"  worst F_cos  : {cos.min():.9f}")


if __name__ == "__main__":
    main()
