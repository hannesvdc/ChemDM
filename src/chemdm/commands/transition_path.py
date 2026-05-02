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

def run(input_data: dict) -> dict:
    """Compute a NEB-refined transition path.

    Input keys: Z, xA, xB, GA, GB. Optional: x, s (ignored — ML provides initial guess).
    Output keys: input keys plus x, s, E_opt_eV, best_force.
    """

    Z = np.asarray(input_data["Z"])
    xA = np.asarray(input_data["xA"])
    xB = np.asarray(input_data["xB"])
    GA = np.asarray(input_data["GA"])
    GB = np.asarray(input_data["GB"])

    context = create_xtb_context(Z)
    path0 = _ml_initial_guess(Z, xA, xB, GA, GB)

    n_steps = 1000
    lr = 1e-3
    k = 1.0           # eV / A^2
    max_step_A = 0.02
    force_tol = 0.03  # eV / A
    path_opt, E_opt_eV, best_force = run_neb_xtb(
        context, path0, n_steps, lr, k, max_step_A, force_tol,
    )
    s = normalized_arclengths(path_opt)

    output = copy.deepcopy(input_data)
    output["x"] = path_opt
    output["s"] = s
    output["E_opt_eV"] = E_opt_eV
    output["best_force"] = best_force
    return output


def _ml_initial_guess(Z, xA, xB, GA, GB):

    mol_size = len(Z)
    d_cutoff = 5.0
    n_rbf = 10

    xA_embedding = MolecularEmbeddingGNN(64, 64, 5, d_cutoff)
    xB_embedding = MolecularEmbeddingGNN(64, 64, 5, d_cutoff)

    tp_network = NewtonE3NN(
        xA_embedding_network=xA_embedding,
        xB_embedding_network=xB_embedding,
        irreps_node_str="48x0e + 16x1o + 16x1e + 8x2e",
        n_refinement_steps=7,
        d_cutoff=d_cutoff,
        n_freq=8,
        n_rbf=n_rbf,
    )

    model_path = os.environ.get(
        "CHEMDM_TRANSITION_PATH_MODEL",
        str(_REPO_ROOT / "models" / "newton_reaction_trajectory_model.pth"),
    )
    state_dict = pt.load(model_path, map_location=pt.device("cpu"), weights_only=True)
    tp_network.load_state_dict(state_dict)
    tp_network.to( dtype=pt.float32 )

    n_images = 10
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
