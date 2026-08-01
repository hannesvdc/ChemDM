import importlib.util
from typing import Optional

import numpy as np
from ase.data import chemical_symbols
from ase.units import Hartree, Bohr


def _to_host(a):
    """Bring a possibly-cupy array/scalar (gpu4pyscf) back to numpy/host."""
    try:
        import cupy
        if isinstance(a, cupy.ndarray):
            return cupy.asnumpy(a)
    except ImportError:
        pass
    return np.asarray(a)


class PySCFPotential:
    """DFT backend on the PySCF engine, GPU-accelerated via gpu4pyscf when present.

    Satisfies the EnergyForceEvaluator protocol (potentials.py): energy_forces(x_A)
    takes positions in Angstrom and returns (energy_eV, forces_eV_per_A), on the
    SAME functional/basis as the Psi4 backend -- just on the pyscf/gpu4pyscf engine
    so it can run on an NVIDIA GPU for a large speedup.

    Engine selection: uses gpu4pyscf (CUDA GPU) if it is importable and
    device != "cpu", otherwise plain CPU pyscf. `functional` is the pyscf xc
    spelling (e.g. "wb97xd"); `disp` (e.g. "d3bj") adds an empirical dispersion
    correction and requires the `pyscf-dispersion` package.

    NOTE: pyscf keeps the input orientation (symmetry off, no reorientation), so
    the returned gradient is already in the same Cartesian frame as x_A -- no
    no_reorient bookkeeping is needed as it is for psi4.
    """

    def __init__(self, Z, charge: int = 0, uhf: int = 0, *,
                 functional: str = "wb97xd", basis: str = "def2-tzvp",
                 disp: Optional[str] = None, density_fit: bool = True,
                 grid_level: int = 3,
                 device: Optional[str] = None, num_threads: Optional[int] = None):
        self.Z = np.asarray(Z, dtype=int)
        self.symbols = [chemical_symbols[z] for z in self.Z]
        self.charge = int(charge)
        self.spin = int(uhf)          # pyscf spin = # unpaired electrons (2S) = uhf
        self.functional = functional
        self.basis = basis
        self.disp = disp
        self.density_fit = bool(density_fit)
        self.grid_level = int(grid_level)

        self._gpu = (device != "cpu") and importlib.util.find_spec("gpu4pyscf") is not None
        if self._gpu:
            import gpu4pyscf.dft as _dft
        else:
            import pyscf.dft as _dft
            if num_threads is not None:
                import pyscf.lib
                pyscf.lib.num_threads(int(num_threads))
        self._dft = _dft

        import pyscf
        self._pyscf = pyscf

    def energy_forces(self, x_A):
        x_A = np.asarray(x_A, dtype=float)
        atoms = [[sym, (float(px), float(py), float(pz))]
                 for sym, (px, py, pz) in zip(self.symbols, x_A)]
        mol = self._pyscf.M(atom=atoms, basis=self.basis, charge=self.charge,
                            spin=self.spin, unit="Angstrom", verbose=0)

        make = self._dft.UKS if self.spin > 0 else self._dft.RKS
        mf = make(mol, xc=self.functional)
        if self.density_fit:
            mf = mf.density_fit()   # RI-JK: ~6x faster for hybrids, matches psi4's default
        mf.grids.level = self.grid_level
        if self.disp is not None:
            mf.disp = self.disp

        energy_hartree = float(_to_host(mf.kernel()))
        grad = _to_host(mf.nuc_grad_method().kernel())      # Hartree/Bohr, (n_atoms, 3)

        energy_eV = energy_hartree * Hartree
        forces_eV_A = -np.asarray(grad, dtype=float) * (Hartree / Bohr)   # forces = -gradient
        return energy_eV, forces_eV_A
