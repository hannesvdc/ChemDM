"""
TBLitePotential
===============

The production xTB potential, backed by `tblite` (grimme-lab), the actively-
maintained library reimplementation of the xTB Hamiltonians. It replaces the
deprecated ``xtb-python`` backend (see ``chemdm.xtbSetup.XTBPotential``), which is
unmaintained and has an analytic-gradient defect at symmetric geometries.

Public contract (identical to the old XTBPotential):

    pot = TBLitePotential(Z, method="GFN1-xTB")
    energy_eV, forces_eV_A = pot.energy_forces(x_A)      # x_A in Angstrom

Energies are returned in eV, forces in eV/Angstrom; input positions are assumed
to be in Angstrom (no conversion at this level).

This is deliberately **standalone** -- it does NOT subclass ``XTBPotential``. The
old class imports ``xtb.ase`` (the deprecated ``xtb-python`` package) at module
load, so subclassing would force ``xtb-python`` to remain a hard dependency of
every process that touches the potential, defeating the migration. Standalone,
``TBLitePotential`` depends only on ``tblite`` + ``ase``; it satisfies the
``chemdm.opt.EnergyForceEvaluator`` contract structurally (duck-typed), no
inheritance required.

Notes
-----
* ``method`` accepts "GFN1-xTB" or "GFN2-xTB".
* ``uhf`` (number of unpaired electrons, 2S) maps onto tblite's ``multiplicity``
  keyword as ``multiplicity = uhf + 1`` (2S+1).
* Implicit solvation is not wired up yet (tblite's ``solvation`` spec differs
  from xtb-python's ``solvent`` string); passing ``solvent`` raises.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms
from tblite.ase import TBLite


class TBLitePotential:
    """xTB potential backed by tblite; energies in eV, forces in eV/Angstrom."""

    def __init__( self,
                  Z: np.ndarray,
                  charge: int = 0,
                  uhf: int = 0,
                  method: str = "GFN2-xTB",
                  accuracy: float = 1.0,
                  electronic_temperature: float = 300.0,
                  max_iterations: int = 250,
                  solvent: str | None = None, ):
        if solvent is not None:
            raise NotImplementedError( "Implicit solvation is not wired up in TBLitePotential yet." )

        self.Z = np.asarray(Z, dtype=int)
        self.charge = charge
        self.uhf = uhf
        self.method = method
        self.accuracy = accuracy
        self.electronic_temperature = electronic_temperature
        self.max_iterations = max_iterations
        self.solvent = solvent

        # tblite takes the total charge and the spin multiplicity (2S+1); the
        # reference's `uhf` is the number of unpaired electrons (2S).
        kwargs = dict( method=method, charge=charge, multiplicity=uhf + 1,
                       accuracy=accuracy, electronic_temperature=electronic_temperature,
                       max_iterations=max_iterations, verbosity=0, )

        self.atoms = Atoms(numbers=self.Z, positions=np.zeros((len(self.Z), 3)))
        self.atoms.calc = TBLite( self.atoms, **kwargs )

    def energy_forces( self, x_A: np.ndarray ) -> tuple[float, np.ndarray]:
        """
        x_A: positions in Angstrom, shape (n_atoms, 3)

        returns:
            energy_eV: float
            forces_eV_per_A: shape (n_atoms, 3)
        """
        self.atoms.positions = np.asarray(x_A, dtype=float)

        # ASE/tblite returns eV and eV/Angstrom
        energy = float(self.atoms.get_potential_energy())
        forces = np.asarray(self.atoms.get_forces(), dtype=float)

        return energy, forces
