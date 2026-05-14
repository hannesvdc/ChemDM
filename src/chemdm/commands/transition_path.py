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

from chemdm.Constants import *
from chemdm.NewtonE3NN import NewtonE3NN
from chemdm.MolecularEmbeddingNetwork import MolecularEmbeddingGNN
from chemdm.MoleculeGraph import MoleculeGraph, batchMolecules
from chemdm.xtbSetup import XTBPotential
from chemdm.nebXtbDirect import run_neb_xtb, normalized_arclengths
from chemdm.relaxMolecule import relaxMolecule
from chemdm.geometry import kabsch_align_numpy, normalize, perpendicular_basis, perpendicular_basis_continuous
from chemdm.progress import ProgressCallback
from chemdm.graph.algorithms import neighbors_from_graph

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
        force_tolerance:
            Main tolerance used for the maximum perpendicular force along the path.
        max_optimizer_steps:
            Hard bound on the total number of Adam optimization steps. L-BFGS typically does not do better.
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
    force_tol = float( input_data.get( "force_tolerance", 0.1 )) # kJ / mol / A
    n_steps = int( input_data.get( "max_optimizer_steps", 2500) )
    
    trajectory = input_data["trajectory"]
    Z = np.asarray(trajectory["Z"], dtype=np.long)
    xA = np.asarray(trajectory["xA"])
    xB = np.asarray(trajectory["xB"])
    GA = np.asarray(trajectory["GA"])
    GB = np.asarray(trajectory["GB"])

    # Construct the XTB force field
    if theory.lower() == "xtb":
        xtb = XTBPotential(Z)

    # Align the end points for stability. Relax endpoints if desired.
    if relax_endpoints:
        on_progress("relax", "Relaxinging reactants and products", fraction=0.02)
        xA = relaxMolecule( xtb, xA, minimizer="Adam", returnOptimizationHistory=False )
        xB = relaxMolecule( xtb, xB, minimizer="Adam", returnOptimizationHistory=False )

    on_progress("align", "Aligning endpoints", fraction=0.10)
    xB = kabsch_align_numpy( xB, xA, Z ) # type: ignore

    # Evaluate the Newton model for a good initial guess.
    on_progress("generate_path", "Generating initial guess for the path", fraction=0.15)
    if tp_network is None:
        tp_network = load_transition_path_model( )
    path0, s0 = _ml_initial_guess( tp_network, Z, xA, xB, GA, GB, n_images ) # type: ignore

    # Make the path chemically feasible. Only do it if the moleulce is large enough.
    # For example, butane is fully relient on methyl rotation.
    if np.sum( (Z != 1) ) > 6:
        path0 = cleanupPath( Z, path0, s0, GA, GB )

    lr = 1e-2
    max_step_A = 0.02
    k = 1.0 * KJ_MOL_PER_EV   # kJ/mol/Å², equivalent to 1 eV/Å²
    max_workers = n_images
    def callback( iter : int, maxF : float ) -> None:
        on_progress( "fine_tune_path", f"(Step {iter}/{n_steps}) Max. Perpendicular Force: {maxF:.2f} [kJ / (mol A)]", 
                     fraction = progress_so_far + (1.0 - progress_so_far) * iter / n_steps )
    on_progress( "fine_tune_path", "Fine-tuning", fraction=0.50 )
    progress_so_far = on_progress.getTotalProgress()
    path_opt, E_opt_eV, best_force = run_neb_xtb( Z, path0, n_steps, lr, k, max_step_A, force_tol, lbfgs_maxiter=n_steps, callback=callback, max_workers=max_workers)
    s = normalized_arclengths(path_opt)
    E_opt_eV -= E_opt_eV[0]

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
                      n_images : int ) -> tuple[np.ndarray, np.ndarray]:
    assert n_images >= 3, "The number of images along the transition path - including endpoints - must be larger than 3."
    mol_size = len(Z)

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
    return x, s_t.detach().cpu().numpy()

def cleanupPath( Z : np.ndarray,
                 path : np.ndarray,
                 s : np.ndarray,
                 GA : np.ndarray,
                 GB : np.ndarray,
                ) -> np.ndarray:
    """
    This function is meant to make the output of the chemical transition-path network 
    chemically viable. Sometimes the hydrogens in a methyl group don't stay in their native
    sp3 configuration, which should happen as long as those hydrogens don't partake in
    the chemical reaction. This function stabilizes methyl groups.

    This function places each methyl H at distance 1.09 Å from the methyl carbon
    """
    Z = Z.astype( dtype=int )
    n_images = path.shape[0]

    # Find the bonded neighbors in the reactants and products.
    carbon_indices = np.where(Z == 6)[0]
    for carbon in carbon_indices:
        neighbors_A = neighbors_from_graph( GA, carbon )
        neighbors_B = neighbors_from_graph( GB, carbon )

        # If the bonded neighbor set changes, this carbon participates in chemistry.
        if set(neighbors_A.tolist()) != set(neighbors_B.tolist()):
            continue
        H_neighbors = neighbors_A[Z[neighbors_A] == 1]
        nonH_neighbors = neighbors_A[Z[neighbors_A] != 1]

        # A methyl carbon should have exactly three H neighbors and one non-H neighbor.
        if len(H_neighbors) != 3:
            continue
        if len(nonH_neighbors) != 1:
            continue
        nonH_neighbor = nonH_neighbors[0]

        # Heavy-atom geometry along the whole path.
        A_path = path[:, nonH_neighbor, :]  # (M, 3)
        C_path = path[:, carbon, :]         # (M, 3)

        # Endpoint geometries.
        A0 = path[0, nonH_neighbor, :]
        C0 = path[0, carbon, :]
        A1 = path[-1, nonH_neighbor, :]
        C1 = path[-1, carbon, :]

        # Use the first H as the reference phase.
        # This assumes endpoint atom ordering is consistent, which is usually true.
        h_ref = int(H_neighbors[0])
        H0 = path[0, h_ref, :]
        H1 = path[-1, h_ref, :]

        phi0 = _methyl_phase_from_H(A0, C0, H0)
        phi1 = _methyl_phase_from_H(A1, C1, H1)

        # Methyl symmetry: rotations by 2π/3 are equivalent.
        # Choose the shortest chemically meaningful phase difference.
        period = 2.0 * np.pi / 3.0
        dphi = _wrap_to_interval(phi1 - phi0, period=period)

        phi_path = phi0 + s * dphi  # (M,)
        #phi_path = np.zeros(n_images)
        H_new = _build_methyl_hydrogens( A=A_path, C=C_path, phi=phi_path )  # (M, 3, 3)
        path[:, H_neighbors, :] = H_new

    return path

def _methyl_phase_from_H(A : np.ndarray, C : np.ndarray, H : np.ndarray ) -> np.ndarray:
    """
    Compute the methyl phase phi of one reference hydrogen.

    A, C, H: (..., 3)
        A = heavy-atom neighbor
        C = methyl carbon
        H = one methyl hydrogen

    Returns
    -------
    phi: (...,)
    """
    axis = normalize(A - C, axis=-1)
    u, v = perpendicular_basis(axis)
    CH = H - C

    # Project C-H direction onto the plane perpendicular to the methyl axis.
    CH_perp = CH - np.sum(CH * axis, axis=-1, keepdims=True) * axis
    CH_perp = normalize(CH_perp, axis=-1)

    x = np.sum(CH_perp * u, axis=-1)
    y = np.sum(CH_perp * v, axis=-1)

    return np.arctan2(y, x)

def _wrap_to_interval(x : np.ndarray, period : float):
    """
    Wrap x to (-period/2, period/2].
    """
    return (x + 0.5 * period) % period - 0.5 * period

def _build_methyl_hydrogens(A, C, phi, r_CH=1.09, ideal_angle_deg=109.5):
    """
    Vectorized methyl hydrogen builder.
    A:   (M, 3)
    C:   (M, 3)
    phi: (M,)

    Returns
    -------
    H: (M, 3, 3)
        Three hydrogen positions for every image.
    """
    axis = normalize(A - C, axis=1)
    u, v = perpendicular_basis_continuous(axis) # use continuous u,v along the path

    ideal_angle = np.deg2rad(ideal_angle_deg)
    axial = np.cos(ideal_angle)
    radial = np.sin(ideal_angle)

    offsets = np.array([0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0])
    angles = phi[:, None] + offsets[None, :]  # (M, 3)
    radial_dirs = np.cos(angles)[:, :, None] * u[:, None, :] + np.sin(angles)[:, :, None] * v[:, None, :]

    dirs = axial * axis[:, None, :] + radial * radial_dirs
    dirs = normalize( dirs, axis=2 )

    return C[:, None, :] + r_CH * dirs