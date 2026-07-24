from __future__ import annotations

from typing import Protocol

import numpy as np

# Evaluator interface
class EnergyForceEvaluator(Protocol):
    """
    Protocol for duck-typing energy and force evaluators.

    The only convention is that positions must be passed in Angstrom,
    and the returned energy has units of eV. The forces have 
    units eV / A.
    """
    # `x` is declared positional-only (the `/`) so that implementers may name
    # the parameter whatever they like (e.g. XTBPotential uses `x_A`).
    def energy_forces(self, x: np.ndarray, /) -> tuple[float, np.ndarray]:
        ...


_XTB_METHODS = {"xtb1": "GFN1-xTB", "xtb2": "GFN2-xTB", 
                "gfn1": "GFN1-xTB", "gfn2": "GFN2-xTB",
                "gfn1-xtb": "GFN1-xTB", "gfn2-xtb": "GFN2-xTB"}
def make_potential( force_field: str,
                    Z : np.ndarray,
                    charge: int=0,
                    uhf: int=0,
                    **kw ) -> EnergyForceEvaluator:
    """
    Factory function to create the required force field. Lazy-imports the necessary
    dependencies during runtime to avoid clashes.
    """
    ff = force_field.lower()
    Z = np.asarray( Z, dtype=int )

    if ff in _XTB_METHODS:
        from chemdm.TBLitePotential import TBLitePotential
        return TBLitePotential( Z, charge=charge, uhf=uhf, method=_XTB_METHODS[ff], **kw )
    if force_field == "dft":
        from chemdm.Psi4Potential import Psi4Potential
        return Psi4Potential( Z, charge=charge, uhf=uhf, **kw )
    raise ValueError( f"Unknown force_field {force_field!r}." )