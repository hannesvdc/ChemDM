import importlib.util
from typing import Optional

import numpy as np
from ase.data import chemical_symbols
from ase.units import Hartree, Bohr

from chemdm.potentials.dispersion import chg_d2_dispersion

from typing import Any

def _to_host( a : Any ):
    """Bring a possibly-cupy array/scalar (gpu4pyscf) back to numpy/host."""
    try:
        import cupy
        if isinstance(a, cupy.ndarray):
            return cupy.asnumpy(a)
    except ImportError:
        pass
    return np.asarray(a)


class PySCFPotential:
    """DFT backend on the PySCF engine. Can be accelerated on GPU via `gpu4pyscf`
    when asked.

    This implementation satisfies the `EnergyForceEvaluator` protocol 
    (potentialInterface.py): energy_forces(x_A) takes positions in Angstrom 
    and returns (energy_eV, forces_eV_per_A) in eV and Angstrom units.
     
    Importantl, the DFT functional/basis are the same as the Psi4 backend. This 
    not only improves testability and backward compatiblity, but is also just
    the correct backend from first principles.

    Engine selection: uses `gpu4pyscf` (CUDA GPU) when the user asks for GPU
    calculations and the module is importable. Otherwise plain CPU pyscf. 
    
    Arguments
    ---------
    functional: str
      The pyscf xc spelling (e.g. "wb97xd"). `
    disp : str, optional 
        Selects an added dispersion correction. "chg-d2" adds chemdm's own ωB97X-D CHG dispersion 
        (pyscf has no D2), any other value (e.g. "d3bj") is a pyscf-native term needing `pyscf-dispersion`.

    NOTE: pyscf keeps the input orientation (symmetry off, no reorientation), so
    the returned gradient is already in the same Cartesian frame as x_A -- no
    no_reorient bookkeeping is needed as it is for psi4.
    """

    def __init__(self, Z : np.ndarray, 
                       charge: int = 0, 
                       uhf: int = 0, *,
                       functional: str = "wb97xd", 
                       basis: str = "def2-tzvp",
                       disp: Optional[str] = None, 
                       density_fit: bool = True,
                       grid_level: int = 3,
                       device: str = "cpu", 
                       num_threads: Optional[int] = None):
        self.Z = np.asarray(Z, dtype=int)
        self.symbols = [chemical_symbols[z] for z in self.Z]
        self.charge = int(charge)
        self.spin = int(uhf)          # pyscf spin = # unpaired electrons (2S) = uhf
        self.functional = functional
        self.basis = basis
        self.disp = disp
        self.density_fit = bool(density_fit)
        self.grid_level = int(grid_level)

        # Discern the device.
        device = device.lower()
        if device == "gpu" and importlib.util.find_spec("gpu4pyscf") is None:
            print( "GPU device requested but gpu4pyscf is not available, falling back to `cpu`")
            self._gpu = False
        elif device == "gpu":
            self._gpu = True
        elif device == "cpu":
            self._gpu = False
        else:
            self._gpu = False
            print( f"Device {device} is unknown. Falling back to `cpu`.")
        
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

    def energy_forces( self, x_A : np.ndarray ) -> tuple[float, np.ndarray]:
        x_A = np.asarray(x_A, dtype=float)
        atoms = [[sym, (float(px), float(py), float(pz))]
                 for sym, (px, py, pz) in zip(self.symbols, x_A)]
        mol = self._pyscf.M(atom=atoms, basis=self.basis, charge=self.charge,
                            spin=self.spin, unit="Angstrom", verbose=0)

        make = self._dft.UKS if self.spin > 0 else self._dft.RKS
        mf = make( mol, xc=self.functional )
        if self.density_fit:
            mf = mf.density_fit( )   # RI-JK: ~6x faster for hybrids, matches psi4's default
        mf.grids.level = self.grid_level

        # "chg-d2" is our own ωB97X-D dispersion, added post-SCF below; any other
        # value is a pyscf-native correction (needs pyscf-dispersion).
        if self.disp is not None and self.disp != "chg-d2":
            mf.disp = self.disp

        # Compute the energy and gradient.
        energy_hartree = float(_to_host(mf.kernel()))
        grad = _to_host(mf.nuc_grad_method().kernel())      # Hartree/Bohr, (n_atoms, 3)
        energy_eV = energy_hartree * Hartree
        forces_eV_A = -np.asarray(grad, dtype=float) * (Hartree / Bohr)   # forces = -gradient

        # Add our own D2 dispersion if required.
        if self.disp == "chg-d2":   # ωB97X-D: pyscf has no D2, so add our own (host, cheap)
            e_d, f_d = chg_d2_dispersion(self.Z, x_A)
            energy_eV += e_d
            forces_eV_A = forces_eV_A + f_d

        return energy_eV, forces_eV_A
