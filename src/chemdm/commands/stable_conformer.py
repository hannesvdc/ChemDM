"""
Minimize the energy of a 3D Molecule: Forcefield refinement.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_XTB_DIR = _REPO_ROOT / "examples" / "xtb"
if str(_XTB_DIR) not in sys.path:
    sys.path.insert(0, str(_XTB_DIR))

import numpy as np

from chemdm.xtbSetup import XTBPotential
from chemdm.relaxMolecule import relaxMolecule

def run(input_data: dict) -> dict:
    """
    Empty implementation for now.
    """
    molecule = input_data["molecule"]
    theory = input_data.get( "theory", "xtb" )
    force_tol = input_data.get( "force_tolerance", 5.0 ) #kJ/mol/A
    max_optimizer_steps = input_data.get( "max_optimizer_steps", 2500 )

    # Fetch the molecule.
    Z = np.asarray( molecule["Z"], dtype=np.long )
    x0 = np.asarray( molecule["x"] )
    bonds = np.asarray( molecule["bonds"] )

    # Construct the XTB force field
    if theory.lower() == "xtb":
        xtb = XTBPotential(Z)

    # Do Adam minimization first, then fine-tune
    relaxMolecule( xtb, x0, minimizer="Adam", verbose=True, returnOptimizationHistory=True )
    return input_data