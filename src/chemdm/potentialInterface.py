from __future__ import annotations

import importlib.util
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
# The force-field registry for every selectable backend. Everything else 
# (the factory, the RS catalog, name resolution) is derived from this list. 
# 
# Order == display order in the RS UI.
#
# Each entry:
#   id           canonical id (what commands / make_potential resolve to)
#   label        display label shown in RS
#   category     UI grouping
#   backend      "tblite" | "pyscf"
#   supports_tp  Can drive a transition-path (NEB) run. xTB and DFT are true
#                PESs that handle bond breaking; Other force fields may not be 
#                valid out of equilibrium or may be too slow.
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
          build=dict(method="GFN2-xTB"), supports_gpu=False ),
    dict( id="gfn1-xtb", label="GFN1-xTB", category="Semiempirical (xTB)",
          backend="tblite", supports_tp=True, aliases=["xtb1", "gfn1"],
          build=dict(method="GFN1-xTB"), supports_gpu=False ),

    dict( id="wb97x-d", label="ωB97X-D / def2-TZVP", category="DFT",
          backend="pyscf", supports_tp=True,
          build=dict(functional="wb97x-d", basis="def2-tzvp", reference="ks"),
          supports_gpu=True ),
    dict( id="wb97m-v", label="ωB97M-V / def2-TZVP", category="DFT",
          backend="pyscf", supports_tp=True,
          build=dict(functional="wb97m-v", basis="def2-tzvp", reference="ks"),
          supports_gpu=True ),
]
DEFAULT_FORCE_FIELD = "gfn2-xtb"

# Lower-cased name (id | label | alias) -> registry entry.
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


# Force-field id -> pyscf functional spelling and add-on dispersion. "chg-d2"
# is chemdm's own ωB97X-D CHG dispersion (added post-SCF in PySCFPotential,
# since pyscf has no D2); VV10 functionals (wb97m-v) need no add-on.
_PYSCF_XC = {
    "wb97x-d":  dict( xc="wb97xd", disp="chg-d2" ),
    "wb97m-v":  dict( xc="wb97mv" ),
}


def make_potential( force_field: str,
                    Z: np.ndarray,
                    charge: int = 0,
                    uhf: int = 0,
                    *,
                    device: str = "cpu",
                    **kw ) -> EnergyForceEvaluator:
    """Build the requested force field. Accepts any id / label / alias. Lazy-
    imports the backend so the xTB path never pulls in psi4 and vice versa.
    Extra kwargs (e.g. basis, num_threads) override the registry defaults."""
    spec = _lookup( force_field )
    Z = np.asarray( Z, dtype=int )
    build = spec["build"]

    if spec["backend"] == "tblite":
        # tblite is always CPU (for now).
        from chemdm.potentials.TBLitePotential import TBLitePotential
        return TBLitePotential( Z, charge=charge, uhf=uhf, method=build["method"], **kw )

    # DFT runs on pyscf (CPU) or gpu4pyscf (GPU); the `device` kwarg picks the
    # engine inside PySCFPotential. _PYSCF_XC maps the id to the pyscf functional
    # spelling and any add-on dispersion (e.g. "chg-d2" for ωB97X-D).
    if spec["backend"] == "pyscf":
        from chemdm.potentials.PySCFPotential import PySCFPotential
        xc = _PYSCF_XC[ spec["id"] ]
        kwargs = dict( functional=xc["xc"], basis=build.get("basis"), disp=xc.get("disp") )
        kwargs.update( kw )        # num_threads, basis override
        return PySCFPotential( Z, charge=charge, uhf=uhf, device=device, **kwargs )

    raise ValueError( f"Unknown backend {spec['backend']!r} for force field {spec['id']!r}" )


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
    have_pyscf = _module_available( "pyscf" )

    out = []
    for ff in _FORCE_FIELDS:
        if ff["backend"] == "tblite":
            ok, reason = have_tblite, ( None if have_tblite else "tblite not installed" )
        elif ff["backend"] == "pyscf":
            ok, reason = have_pyscf, ( None if have_pyscf else "PySCF not installed" )
        else:
            ok, reason = True, None
        out.append( { "id": ff["id"], "label": ff["label"], "category": ff["category"],
                      "supports_tp": ff["supports_tp"], "available": ok, "reason": reason,
                      "supports_gpu": ff["supports_gpu"] } )
    return out
