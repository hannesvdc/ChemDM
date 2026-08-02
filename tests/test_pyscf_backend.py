"""Backend-mechanics tests for PySCFPotential (CPU path).

Numerical correctness of ωB97X-D (functional + CHG dispersion, checked against
psi4) lives in test_chg_d2.py. Here we only exercise the plumbing: energy/force
units and shapes, the device selector and its GPU->CPU fallback, and that
make_potential routes each force field to the right backend (with `device`
never leaking into the CPU-only tblite constructor). Skips if pyscf is absent.
"""
import numpy as np
import pytest

pytest.importorskip("pyscf")

from chemdm.potentials.PySCFPotential import PySCFPotential
from chemdm.potentialInterface import make_potential

Z = np.array([8, 1, 1])
X0 = np.array([[0.0, 0.0, 0.117], [0.0, 0.757, -0.469], [0.0, -0.757, -0.469]])


def test_energy_forces_units_and_shapes():
    e, f = PySCFPotential(Z, functional="wb97xd", basis="def2-svp", device="cpu").energy_forces(X0)
    assert np.isfinite(e)
    assert -2200 < e < -1950          # water ωB97X ~ -2075 eV -> confirms eV (not Hartree)
    assert f.shape == (3, 3) and np.all(np.isfinite(f))


def test_device_cpu_selects_cpu():
    assert PySCFPotential(Z, basis="def2-svp", device="cpu")._gpu is False


def test_gpu_falls_back_to_cpu_when_unavailable():
    # No gpu4pyscf in this env -> device="gpu" must degrade to CPU, not crash.
    assert PySCFPotential(Z, basis="def2-svp", device="gpu")._gpu is False


def test_make_potential_routes_wb97x_d_to_pyscf():
    pot = make_potential("wb97x-d", Z, basis="def2-svp", device="cpu")
    assert isinstance(pot, PySCFPotential)
    assert pot.disp == "chg-d2"        # ωB97X-D carries the CHG dispersion flag


def test_make_potential_device_is_noop_for_tblite():
    # `device` must not leak into TBLitePotential (which has no such argument).
    from chemdm.potentials.TBLitePotential import TBLitePotential
    assert isinstance(make_potential("gfn2-xtb", Z, device="gpu"), TBLitePotential)
