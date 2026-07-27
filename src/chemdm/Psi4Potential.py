import os
import numpy as np
from ase.data import chemical_symbols
from ase.units import Hartree, Bohr

from typing import Optional

class Psi4Potential:
    """High-accuracy Kohn-Sham DFT backend potential.

    Satisfies the EnergyForceEvaluator protocol (opt.py): energy_forces(x_A)
    takes positions in Angstrom and returns (energy_eV, forces_eV_per_A).

    Important note: Psi4 uses global process state and is NOT thread-safe. Must use one
    instance per process and never share a process across threads.
    """
    def __init__(self, Z : np.ndarray, 
                       charge: int=0, 
                       uhf: int=0,
                       *,
                       functional: str="wb97x-d", 
                       basis: Optional[str]="def2-tzvp",
                       reference: Optional[str]=None,
                       options: Optional[dict]=None,
                       memory: str="4 GB", 
                       num_threads: Optional[int]=1, 
                       scratch_dir=None):
        import psi4
        self._psi4 = psi4
  
        self.Z = np.asarray( Z, dtype=int )
        self.symbols = [ chemical_symbols[z] for z in self.Z ]
        self.charge = int(charge)
        self.multiplicity = int(uhf) + 1
        self.method = functional if basis is None else f"{functional}/{basis}"

        psi4.core.be_quiet()
        psi4.set_memory( memory )
        if num_threads is not None:
            psi4.set_num_threads( int(num_threads) )
        if scratch_dir is not None:
            os.makedirs( scratch_dir, exist_ok=True )
            psi4.core.IOManager.shared_object().set_default_path( scratch_dir )

        # DFT backends default to a Kohn-Sham reference (RKS closed-shell, UKS
        # open-shell -- UKS is also what makes DFT usable through the
        # multireference TS region on transition paths). Wavefunction methods
        # (HF / MP2 / CCSD(T)) must pass reference="RHF"/"UHF" instead.
        if reference is None:
            reference = "UKS" if self.multiplicity > 1 else "RKS"
        scf_options = { "reference": reference, "maxiter": 200 }
        if options is not None:
            scf_options.update( options )
        psi4.set_options( scf_options )          # bump SCF iterations for stretched geoms

    def _geometry( self, x_A : np.ndarray ):
        x_A = np.asarray(x_A, dtype=float)
        lines = [f"{self.charge} {self.multiplicity}"]
        for sym, (x, y, z) in zip(self.symbols, x_A):
            lines.append(f"{sym} {x:.10f} {y:.10f} {z:.10f}")

        # units + no_com/no_reorient/symmetry c1 are CRITICAL: they keep the
        # gradient in the INPUT Cartesian frame. Without them Psi4 recenters
        # and reorients, and the returned forces come back in a rotated frame.
        lines += ["units angstrom", "no_com", "no_reorient", "symmetry c1"]
        return self._psi4.geometry("\n".join(lines))

    def energy_forces( self, x_A : np.ndarray ) -> tuple[float, np.ndarray]:
        psi4 = self._psi4
        psi4.core.clean()                      # free prior scratch/state
        mol = self._geometry( x_A )
        grad, wfn = psi4.gradient( self.method, molecule=mol, return_wfn=True ) # type: ignore

        energy_eV = float( wfn.energy() ) * Hartree # type: ignore
        grad = np.asarray( grad )                # Hartree/Bohr, (n_atoms, 3)
        forces_eV_A = -grad * (Hartree / Bohr) # forces = -gradient, -> eV/A
        return energy_eV, forces_eV_A