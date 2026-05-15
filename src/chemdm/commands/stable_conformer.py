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
from chemdm.relaxMolecule import minimize_with_adam
from chemdm.progress import ProgressCallback

def run(input_data: dict, 
        on_progress : ProgressCallback) -> dict:
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

    # Construct the XTB force field
    if theory.lower() == "xtb":
        xtb = XTBPotential(Z)

    # Do Adam minimization first, then fine-tune
    lr0 = 1e-3
    x_min, history = minimize_with_adam( xtb, x0, lr0=lr0, force_tolerance_kJ_mol_A=force_tol, max_steps=max_optimizer_steps, verbose=True )
    energies = np.array([ row["energy_kJ_mol"] for row in history])
    rmsds = np.array([ row["rmsd"] for row in history])
    converged = (history[-1]["max_force_rms"] < force_tol)

    # Build the output dictionary
    output_data = { "Z" : Z, 
                    "x" : x_min, 
                    "energies" : energies, 
                    "rmsds" : rmsds, 
                    "final_force_max" : history[-1]["max_force_rms"],
                    "converged" : converged}
    return output_data