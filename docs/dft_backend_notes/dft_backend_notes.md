# DFT Backend Notes — pyscf ↔ psi4 Numerical Agreement

**Date:** 2026-08-01 · **Code:** `src/chemdm/potentials/{PySCFPotential,dispersion}.py`,
`tests/test_chg_d2.py`

A reference for *why the two DFT engines don't agree to the last digit*, so the
loose force tolerance in the tests isn't mistaken for a bug — and so nobody
re-investigates the ~0.03 eV/Å force gap from scratch.

## TL;DR

- **Production engine is pyscf / gpu4pyscf.** psi4 is no longer in the runtime
  path (`make_potential` routes only to tblite/pyscf); `Psi4Potential.py` is kept
  **only as a validation oracle for the tests**.
- Our **CHG dispersion** (`chg_d2_dispersion`, the ωB97X-D "-D" term, which pyscf
  cannot produce — it only has D3/D4) matches psi4's `EmpiricalDispersion` to
  **~1e-8 eV/Å in forces and <1e-6 eV in energy** across all parameterized
  elements. It is exact.
- The **ωB97X functional** computed by pyscf vs psi4 agrees on **energy to
  ~2 meV** but on **forces only to ~0.03 eV/Å** for heavier atoms (e.g. Cl).
  This is a DF-auxiliary-basis + integration-grid difference between two
  independent codes — **not a bug**, and it does **not** shrink with grid.

## The measurement (chloromethane, ωB97X-D / def2-SVP)

Decomposing the full-potential force difference into functional + dispersion:

```
dispersion  (ours vs psi4)  : max|dF| = 1.1e-08 eV/A   (exact; the term itself is ~0.014 eV/A)
FUNCTIONAL  (pyscf vs psi4) : max|dF| = 0.0323 eV/A    ← the entire discrepancy
```

So the gap is **100% the functional**, 0% the dispersion. And it is not grid
noise: bumping pyscf's grid from level 3 → 5 moved it only 0.0323 → 0.0280 eV/Å
(whereas for a light-atom molecule like ethanol the same change drops pyscf's
*own* FD grid-noise ~20×, 1.2e-2 → 5.7e-4). The residual is the genuine
cross-implementation difference (different RI/DF aux basis, different DFT grid).

## Why energy agrees but forces don't

Energy is variational and smooth, so grid/DF discretization error is small and
largely cancels. Forces are derivatives and amplify that discretization — heavier
atoms (more diffuse density, steeper core) make it worse. Two codes converge to
the same functional in the basis-set/grid limit, but at production settings
(def2 basis, DF, a finite grid) they legitimately differ at the ~0.03 eV/Å level.

## Consequences

- **This gap is a *validation* artifact only.** In production, both CPU and GPU
  run pyscf/gpu4pyscf — the *same* functional implementation, DF, and grid — so
  CPU and GPU results agree with each other; the cross-engine gap never appears
  in a real run. It shows up solely because the tests use psi4 as an external
  oracle.
- **Test design (`tests/test_chg_d2.py`):** the tight check is on **energy**
  (which carries the ~50 meV dispersion signal, so it catches a missing/incorrect
  D2), while cross-engine **forces** are a loose ballpark sanity (`< 0.05 eV/Å`).
  Precise force correctness is covered separately: the dispersion forces by
  finite difference + live psi4 (1e-6), and the SCF forces by pyscf's own
  analytic gradient.
- **For NEB / optimization**, use `grid_level` 4–5 (default 3 gives ~1e-2 eV/Å of
  pyscf's *own* force noise, near typical NEB tolerances). That is a separate
  concern from the cross-engine gap above and *does* shrink with grid.

## Reproduce

`tests/test_chg_d2.py::test_full_wb97x_d_matches_psi4` exercises the full
potential across O/N/F/S/Cl; the decomposition above is a few lines building
`PySCFPotential`, `Psi4Potential`, and `psi4.driver.EmpiricalDispersion` on the
same geometry and differencing their forces.
