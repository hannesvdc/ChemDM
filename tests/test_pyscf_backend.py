"""CPU-path tests for the PySCF / gpu4pyscf DFT backend.

Validates units, the gradient sign, and the CHEMDM_DFT_ENGINE routing on CPU
pyscf -- the same code path gpu4pyscf runs on a GPU (only the imported dft
module differs). Skips if pyscf is not installed.
"""
import numpy as np
import pytest

pytest.importorskip("pyscf")

from chemdm.potentials.PySCFPotential import PySCFPotential

Z = np.array([8, 1, 1])
X0 = np.array([[0.10, 0.05, 0.00],
               [0.75, 0.60, 0.30],
               [-0.60, 0.55, -0.20]])


def test_energy_units_and_numerical_gradient():
    pot = PySCFPotential(Z, functional="wb97xd", basis="def2-svp", device="cpu")
    e, f = pot.energy_forces(X0)
    assert np.isfinite(e)
    assert f.shape == (3, 3) and np.all(np.isfinite(f))
    # water wb97x-d ~ -76 Ha ~ -2075 eV
    assert -2200 < e < -1950
    # forces == -dE/dx : central difference must match returned forces
    h = 1e-3
    fd = np.zeros_like(X0)
    for i in range(3):
        for k in range(3):
            xp = X0.copy(); xp[i, k] += h
            xm = X0.copy(); xm[i, k] -= h
            fd[i, k] = -(pot.energy_forces(xp)[0] - pot.energy_forces(xm)[0]) / (2 * h)
    assert np.allclose(f, fd, atol=2e-2), np.abs(f - fd).max()


def test_engine_selection_via_env(monkeypatch):
    from chemdm.potentialInterface import make_potential
    monkeypatch.setenv("CHEMDM_DFT_ENGINE", "pyscf")
    pot = make_potential("wb97x-d", Z, basis="def2-svp", device="cpu")
    assert isinstance(pot, PySCFPotential)
    e, _ = pot.energy_forces(X0)
    assert np.isfinite(e)
