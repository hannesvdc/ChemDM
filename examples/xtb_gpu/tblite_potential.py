"""
TBLitePotential
===============

A drop-in analogue of ``chemdm.xtbSetup.XTBPotential`` backed by `tblite`
(grimme-lab), the actively-maintained library reimplementation of the xTB
Hamiltonians. It replaces the deprecated ``xtb-python`` backend used by
``XTBPotential`` while keeping the exact same public contract:

    pot = TBLitePotential(Z, method="GFN1-xTB")
    energy_eV, forces_eV_A = pot.energy_forces(x_A)      # x_A in Angstrom

Because ``tblite.ase.TBLite`` is an ASE calculator just like ``xtb-python``'s
``XTB``, this is a subclass of ``XTBPotential`` that overrides only the
calculator construction; ``energy_forces`` is inherited unchanged (it merely
drives ``self.atoms``). Both codes speak ASE's eV / eV·Angstrom^-1 convention, so
a correspondence test measures only algorithmic differences between the two
implementations of GFN-xTB.

Motivation
----------
The production ``XTBPotential`` uses ``xtb-python``, which is deprecated and has a
confirmed analytical-gradient defect at some symmetric-equilibrium geometries
(the reference's analytical force disagrees with the finite-difference gradient
of its own energy; see ``reference_gradient_bug.py``). tblite is the grimme-lab-
recommended replacement and is expected to be free of that defect.

Notes
-----
* ``method`` accepts the same strings as ``XTBPotential`` ("GFN1-xTB",
  "GFN2-xTB"). Unlike dxtb, tblite provides both GFN1 and GFN2 on macOS.
* ``uhf`` (number of unpaired electrons) maps onto tblite's ``multiplicity``
  keyword as ``multiplicity = uhf + 1`` (2S+1).
* Implicit solvation is not wired up yet (tblite's ``solvation`` spec differs
  from xtb-python's ``solvent`` string); passing ``solvent`` raises.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms
from tblite.ase import TBLite

from chemdm.xtbSetup import XTBPotential


class TBLitePotential(XTBPotential):
    """xTB potential backed by tblite; mirrors XTBPotential.energy_forces."""

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
