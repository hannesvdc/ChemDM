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
                "gfn1-xtb": "GFN1-xTB", "gfn2-xtb": "GFN2-xTB",
                "xtb": "GFN2-xTB"}

# Psi4 backends. Each entry: Psi4 method string, default basis (None lets a
# composite method load its own built-in basis), SCF reference family
# ("ks" = Kohn-Sham DFT, "hf" = wavefunction), optional extra Psi4 options,
# and whether the method needs the external dispersion/gCP programs
# (simple-dftd3 + gcp-correction) installed in the environment.
_PSI4_METHODS = {
    # DFT hybrids -- dispersion handled internally by Psi4
    "wb97x-d":    dict(method="wb97x-d",    basis="def2-tzvp", reference="ks"),
    "wb97m-v":    dict(method="wb97m-v",    basis="def2-tzvp", reference="ks"),
    "m06-2x":     dict(method="m06-2x",     basis="def2-tzvp", reference="ks"),
    # DFT needing external D3/gCP (simple-dftd3 + gcp-correction)
    "b3lyp-d3bj": dict(method="b3lyp-d3bj", basis="def2-tzvp", reference="ks", needs_dispersion=True),
    "pbe0-d3bj":  dict(method="pbe0-d3bj",  basis="def2-tzvp", reference="ks", needs_dispersion=True),
    "pbeh-3c":    dict(method="pbeh3c",     basis=None,        reference="ks", needs_dispersion=True),
    "b97-3c":     dict(method="b973c",      basis=None,        reference="ks", needs_dispersion=True),
    # Wavefunction methods -- HF reference
    "hf":         dict(method="hf",         basis="def2-tzvp", reference="hf"),
    "mp2":        dict(method="mp2",        basis="def2-tzvp", reference="hf"),
    "scs-mp2":    dict(method="scs-mp2",    basis="def2-tzvp", reference="hf"),
    "ccsd(t)":    dict(method="ccsd(t)",    basis="cc-pvdz",   reference="hf",
                       options={"scf_type": "pk", "cc_type": "conv",
                                "mp2_type": "conv", "freeze_core": True}),
}
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

    if ff in _PSI4_METHODS:
        from chemdm.Psi4Potential import Psi4Potential
        spec = _PSI4_METHODS[ff]
        open_shell = uhf > 0
        reference = ( "UKS" if open_shell else "RKS" ) if spec["reference"] == "ks" \
                    else ( "UHF" if open_shell else "RHF" )
        kwargs = dict( functional=spec["method"], basis=spec.get("basis"),
                       reference=reference, options=spec.get("options") )
        kwargs.update( kw )   # caller config (e.g. basis, num_threads) overrides defaults
        return Psi4Potential( Z, charge=charge, uhf=uhf, **kwargs )

    raise ValueError(
        f"Unknown force_field {force_field!r}. "
        f"Choose from xTB {sorted(_XTB_METHODS)} or Psi4 {sorted(_PSI4_METHODS)}." )