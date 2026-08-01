from __future__ import annotations

import importlib.util
import os
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


# ---------------------------------------------------------------------------
# The force-field registry: the single source of truth for every selectable
# backend. Everything else (the factory, the RS catalog, name resolution) is
# derived from this list. Order == display order in the RS UI.
#
# Each entry:
#   id           canonical id (what commands / make_potential resolve to)
#   label        display label shown in RS
#   category     UI grouping
#   backend      "tblite" | "psi4"
#   supports_tp  may drive a transition-path (NEB) run. xTB and DFT are true
#                PESs that handle bond breaking; HF/MP2/CCSD(T) are relaxation /
#                single-point only (MP2 diverges near a TS, HF dissociates
#                incorrectly, CCSD(T) is too costly for a full band).
#   aliases      alternate accepted names (case-insensitive), e.g. GFN spellings
#   build        kwargs handed to the backend potential constructor. For psi4,
#                `reference` is a family ("ks" = Kohn-Sham DFT, "hf" =
#                wavefunction) resolved to RKS/UKS or RHF/UHF by spin, and
#                `basis=None` lets a composite method load its own built-in basis.
#   needs_dispersion  (psi4 only) requires simple-dftd3 + dftd3-python + gcp.
# ---------------------------------------------------------------------------
_FORCE_FIELDS = [
    dict( id="gfn2-xtb", label="GFN2-xTB", category="Semiempirical (xTB)",
          backend="tblite", supports_tp=True, aliases=["xtb2", "gfn2", "xtb"],
          build=dict(method="GFN2-xTB") ),
    dict( id="gfn1-xtb", label="GFN1-xTB", category="Semiempirical (xTB)",
          backend="tblite", supports_tp=True, aliases=["xtb1", "gfn1"],
          build=dict(method="GFN1-xTB") ),

    dict( id="wb97x-d", label="ωB97X-D / def2-TZVP", category="DFT",
          backend="psi4", supports_tp=True,
          build=dict(functional="wb97x-d", basis="def2-tzvp", reference="ks") ),
    dict( id="wb97m-v", label="ωB97M-V / def2-TZVP", category="DFT",
          backend="psi4", supports_tp=True,
          build=dict(functional="wb97m-v", basis="def2-tzvp", reference="ks") ),
    dict( id="m06-2x", label="M06-2X / def2-TZVP", category="DFT",
          backend="psi4", supports_tp=True,
          build=dict(functional="m06-2x", basis="def2-tzvp", reference="ks") ),
    dict( id="b3lyp-d3bj", label="B3LYP-D3(BJ) / def2-TZVP", category="DFT",
          backend="psi4", supports_tp=True, needs_dispersion=True,
          build=dict(functional="b3lyp-d3bj", basis="def2-tzvp", reference="ks") ),
    dict( id="pbe0-d3bj", label="PBE0-D3(BJ) / def2-TZVP", category="DFT",
          backend="psi4", supports_tp=True, needs_dispersion=True,
          build=dict(functional="pbe0-d3bj", basis="def2-tzvp", reference="ks") ),
    dict( id="pbeh-3c", label="PBEh-3c", category="DFT (composite)",
          backend="psi4", supports_tp=True, needs_dispersion=True,
          build=dict(functional="pbeh3c", basis=None, reference="ks") ),
    dict( id="b97-3c", label="B97-3c", category="DFT (composite)",
          backend="psi4", supports_tp=True, needs_dispersion=True,
          build=dict(functional="b973c", basis=None, reference="ks") ),

    dict( id="mp2", label="MP2 / def2-TZVP", category="Wavefunction",
          backend="psi4", supports_tp=False,
          build=dict(functional="mp2", basis="def2-tzvp", reference="hf") ),
    dict( id="scs-mp2", label="SCS-MP2 / def2-TZVP", category="Wavefunction",
          backend="psi4", supports_tp=False,
          build=dict(functional="scs-mp2", basis="def2-tzvp", reference="hf") ),
    dict( id="hf", label="Hartree-Fock / def2-TZVP", category="Wavefunction",
          backend="psi4", supports_tp=False,
          build=dict(functional="hf", basis="def2-tzvp", reference="hf") ),
    dict( id="ccsd(t)", label="CCSD(T) / cc-pVDZ", category="Wavefunction",
          backend="psi4", supports_tp=False,
          build=dict(functional="ccsd(t)", basis="cc-pvdz", reference="hf",
                     options={"scf_type": "pk", "cc_type": "conv",
                              "mp2_type": "conv", "freeze_core": True}) ),
]

DEFAULT_FORCE_FIELD = "gfn2-xtb"

# Lower-cased name (id | label | alias) -> registry entry. One index feeds both
# name resolution and the factory, so "what RS shows" and the internal name can
# never disagree.
_BY_NAME = {
    name.lower(): ff
    for ff in _FORCE_FIELDS
    for name in [ff["id"], ff["label"], *ff.get("aliases", [])]
}


def _lookup( force_field: str ) -> dict:
    """Return the registry entry for an id / label / alias, else raise."""
    ff = _BY_NAME.get( force_field.strip().lower() ) # a dict --- element of _FORCE_FIELDS
    if ff is None:
        valid = [ f["id"] for f in _FORCE_FIELDS ]
        raise ValueError( f"Unknown force field {force_field!r}. Valid options: {valid}" )
    return ff


def resolve_force_field( name: str ) -> str:
    """Normalize an RS-provided force field (id, display label, or alias, all
    case-insensitive) to its canonical id. The single map shared between the RS
    catalog (list-force-fields) and every command."""
    return _lookup( name )["id"]


# psi4 DFT force-field id -> pyscf/gpu4pyscf functional spelling (+ dispersion).
# Only these DFT methods are wired for the pyscf engine. The wb97* / m06-2x
# entries were verified to match the psi4 numbers to ~1e-4 Ha; the -d3bj ones
# additionally need `pip install pyscf-dispersion` on the GPU node.
_PYSCF_XC = {
    "wb97x-d":    dict( xc="wb97xd" ),
    "wb97m-v":    dict( xc="wb97mv" ),
    "m06-2x":     dict( xc="m06-2x" ),
    "b3lyp-d3bj": dict( xc="b3lyp", disp="d3bj" ),
    "pbe0-d3bj":  dict( xc="pbe0",  disp="d3bj" ),
}


def make_potential( force_field: str,
                    Z: np.ndarray,
                    charge: int = 0,
                    uhf: int = 0,
                    **kw ) -> EnergyForceEvaluator:
    """Build the requested force field. Accepts any id / label / alias. Lazy-
    imports the backend so the xTB path never pulls in psi4 and vice versa.
    Extra kwargs (e.g. basis, num_threads) override the registry defaults."""
    spec = _lookup( force_field )
    Z = np.asarray( Z, dtype=int )
    build = spec["build"]

    if spec["backend"] == "tblite":
        from chemdm.potentials.TBLitePotential import TBLitePotential
        return TBLitePotential( Z, charge=charge, uhf=uhf, method=build["method"], **kw )

    # DFT may run on the pyscf / gpu4pyscf engine instead (set
    # CHEMDM_DFT_ENGINE=pyscf, e.g. on a GPU node) for the same functional at
    # GPU speed. Wavefunction methods (hf/mp2/ccsd(t)) always run on psi4.
    engine = os.environ.get( "CHEMDM_DFT_ENGINE", "psi4" ).strip().lower()
    if build["reference"] == "ks" and engine in ( "pyscf", "gpu4pyscf", "gpu" ):
        xc = _PYSCF_XC.get( spec["id"] )
        if xc is None:
            raise ValueError(
                f"Force field {spec['id']!r} is not available on the pyscf/gpu4pyscf "
                f"engine yet; unset CHEMDM_DFT_ENGINE to run it on psi4." )
        from chemdm.potentials.PySCFPotential import PySCFPotential
        kwargs = dict( functional=xc["xc"], basis=build.get("basis"), disp=xc.get("disp") )
        kwargs.update( kw )
        return PySCFPotential( Z, charge=charge, uhf=uhf, **kwargs )

    # psi4: resolve the reference family to a concrete reference by spin state.
    from chemdm.potentials.Psi4Potential import Psi4Potential
    open_shell = (uhf > 0)
    reference = ( "UKS" if open_shell else "RKS" ) if build["reference"] == "ks" \
                else ( "UHF" if open_shell else "RHF" )
    kwargs = dict( functional=build["functional"], basis=build.get("basis"),
                   reference=reference, options=build.get("options") )
    kwargs.update( kw )
    return Psi4Potential( Z, charge=charge, uhf=uhf, **kwargs )


def _module_available( name: str ) -> bool:
    try:
        return importlib.util.find_spec( name ) is not None
    except (ImportError, ValueError):
        return False


def available_force_fields() -> list[dict]:
    """
    The force-field catalog annotated with runtime availability, so the RS
    front end only offers what is actually installed in this deployment.

    Each entry: id, label, category, supports_tp, available, reason.
    """
    have_tblite = _module_available( "tblite" )
    have_psi4 = _module_available( "psi4" )
    have_dftd3 = _module_available( "dftd3" )  # D3 Python API for -d3bj / 3c methods

    out = []
    for ff in _FORCE_FIELDS:
        if ff["backend"] == "tblite":
            ok, reason = have_tblite, ( None if have_tblite else "tblite not installed" )
        elif not have_psi4:
            ok, reason = False, "psi4 not installed"
        elif ff.get("needs_dispersion") and not have_dftd3:
            ok, reason = False, "dftd3-python not installed"
        else:
            ok, reason = True, None
        out.append( { "id": ff["id"], "label": ff["label"], "category": ff["category"],
                      "supports_tp": ff["supports_tp"], "available": ok, "reason": reason } )
    return out
