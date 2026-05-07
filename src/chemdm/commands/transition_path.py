"""Transition-path experiment: ML initial guess + xTB-NEB refinement.

Bridges to the science code in ``examples/xtb/`` (xtbSetup.py, neb.py, etc.)
via sys.path injection. Once those modules move into the package proper, the
sys.path block can go.
"""
from __future__ import annotations

import copy
import os
import sys
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[3]
_XTB_DIR = _REPO_ROOT / "examples" / "xtb"
if str(_XTB_DIR) not in sys.path:
    sys.path.insert(0, str(_XTB_DIR))

import numpy as np
import torch as pt

from chemdm.NewtonE3NN import NewtonE3NN
from chemdm.MolecularEmbeddingNetwork import MolecularEmbeddingGNN
from chemdm.MoleculeGraph import MoleculeGraph, batchMolecules
from chemdm.xtbSetup import create_xtb_context
from chemdm.nebXtb import run_neb_xtb, normalized_arclengths
from chemdm.relaxMolecule import relaxMolecule
from chemdm.geometry import kabsch_align_numpy
from chemdm.progress import ProgressCallback

def load_transition_path_model() -> NewtonE3NN:
    d_cutoff = 5.0
    n_rbf = 10

    xA_embedding = MolecularEmbeddingGNN(64, 64, 5, d_cutoff)
    xB_embedding = MolecularEmbeddingGNN(64, 64, 5, d_cutoff)

    n_refinement_steps = 7
    model = NewtonE3NN( xA_embedding_network=xA_embedding,
                        xB_embedding_network=xB_embedding,
                        irreps_node_str="48x0e + 16x1o + 16x1e + 8x2e",
                        n_refinement_steps=n_refinement_steps,
                        d_cutoff=d_cutoff,
                        n_freq=8,
                        n_rbf=n_rbf,
                        )

    model_path = os.environ.get( "CHEMDM_TRANSITION_PATH_MODEL", str(_REPO_ROOT / "models" / "newton_reaction_trajectory_model.pth"), )
    state_dict = pt.load(model_path, map_location=pt.device("cpu"), weights_only=True)
    model.load_state_dict(state_dict)
    model.to(dtype=pt.float32)
    model.eval()

    return model

def run( input_data: dict, 
         on_progress : ProgressCallback, 
         tp_network : Optional[NewtonE3NN] ) -> dict:
    """Compute a NEB-refined transition path.

    Arguments:
    ---------
    input_data: dict with keys
        trajectory:
            Z, xA, xB, GA, GB
        n_images:
            Number of images along the path, including endpoints.
        theory:
            Currently only "xTB".
        relax_endpoints:
            Whether to relax endpoints before path generation.
    on_progress : ProgressCallback
        Feed computational progress info back to the caller.
    tp_network : chemdm.NewtonE3NN
        The main Newton model used to generate initial guesses for the transitoin path.
    
    Returns:
    --------
    output_data: dict with keys:
        x:
            Optimized path, shape (n_images, n_atoms, 3)
        s:
            Normalized arclength coordinates.
        E_opt_eV:
            Energies along optimized path.
        best_force:
            Best/worst force metric returned by NEB optimizer.
        diagnostics:
            Algorithm settings used for this run.
    """
    print(f"[runner] keys={list(input_data.keys())}", flush=True, file=sys.stderr) 
    print(f"[runner] {input_data['n_images']}", flush=True, file=sys.stderr) 

    n_images = int( input_data.get( "n_images", 10 ) )
    theory = input_data.get( "theory", "xTB" )
    relax_endpoints = bool( input_data.get( "relax_endpoints", False) )
    
    trajectory = input_data["trajectory"]
    Z = np.asarray(trajectory["Z"])
    xA = np.asarray(trajectory["xA"])
    xB = np.asarray(trajectory["xB"])
    GA = np.asarray(trajectory["GA"])
    GB = np.asarray(trajectory["GB"])

    # Construct the OpenMM force field
    if theory.lower() == "xtb":
        context = create_xtb_context(Z)

    # Align the end points for stability. Relax endpoints if desired.
    if relax_endpoints:
        on_progress("relax", "Relaxinging reactants and products", fraction=0.02)
        xA = relaxMolecule( context, xA, minimizer="Adam" )
        xB = relaxMolecule( context, xB, minimizer="Adam" )

    on_progress("align", "Aligning endpoints", fraction=0.10)
    xB = kabsch_align_numpy( xB, xA )

    # Evaluate the Newton model for a good initial guess.
    on_progress("generate_path", "Generating initial guess for the path", fraction=0.15)
    if tp_network is None:
        tp_network = load_transition_path_model( )
    path0 = _ml_initial_guess( tp_network, Z, xA, xB, GA, GB, n_images )

    n_steps = 1000
    lr = 1e-3
    k = 1.0           # eV / A^2
    max_step_A = 0.02
    force_tol = 0.03  # eV / A
    on_progress( "fine_tune_path", "Fine-tuning", fraction=0.50 )
    path_opt, E_opt_eV, best_force = run_neb_xtb( context, path0, n_steps, lr, k, max_step_A, force_tol )
    s = normalized_arclengths(path_opt)

    # Send back to the server as a dict.
    on_progress( "path_done", "Calculations Finished", fraction=1.0 )
    output = copy.deepcopy(input_data)
    output["x"] = path_opt
    output["s"] = s
    output["E_opt_eV"] = E_opt_eV
    output["best_force"] = best_force
    return output


def _ml_initial_guess( tp_network : NewtonE3NN,
                       Z : np.ndarray, 
                      xA : np.ndarray, 
                      xB : np.ndarray, 
                      GA : np.ndarray, 
                      GB : np.ndarray,
                      n_images : int ):
    assert n_images >= 3, "The number of images along the transition path - including endpoints - must be larger than 3."
    mol_size = len(Z)

    model_path = os.environ.get( "CHEMDM_TRANSITION_PATH_MODEL", str(_REPO_ROOT / "models" / "newton_reaction_trajectory_model.pth") )
    state_dict = pt.load( model_path, map_location=pt.device("cpu"), weights_only=True )
    tp_network.load_state_dict( state_dict )
    tp_network.to( dtype=pt.float32 )

    # Evaluate the network at equidistant points along the trajectory.
    s_t = pt.linspace(0.0, 1.0, n_images)
    xa_batched, xb_batched, s_values = [], [], []
    for n in range(n_images):
        xa_batched.append(MoleculeGraph(pt.tensor(Z, dtype=pt.int), pt.tensor(xA, dtype=pt.float32), pt.tensor(GA)))
        xb_batched.append(MoleculeGraph(pt.tensor(Z, dtype=pt.int), pt.tensor(xB, dtype=pt.float32), pt.tensor(GB)))
        s_values.append(s_t[n] * pt.ones(mol_size, dtype=pt.float32))
    xa_mol = batchMolecules(xa_batched)
    xb_mol = batchMolecules(xb_batched)
    s_cat = pt.cat(s_values)

    molecule_path, _ = tp_network(xa_mol, xb_mol, s_cat)
    x = molecule_path.x.detach().numpy().reshape(n_images, mol_size, 3)
    return x
