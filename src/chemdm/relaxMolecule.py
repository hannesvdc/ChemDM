import sys

import numpy as np
import torch as pt

from chemdm.Constants import *

from chemdm.potentialInterface import EnergyForceEvaluator, make_potential
from chemdm.diagnostics import *


def evaluateEnergyAndForces( potential: EnergyForceEvaluator, 
                             x: np.ndarray):
    """
    Evaluate the energy and force field in the given positions (in Angstrom). 
    Returns energy in kJ/mol and forces in kJ / mol / A.

    x: (n_atoms, 3), Angstrom

    returns:
        energy_kJ_mol: float
        forces_eV_A: (n_atoms, 3), eV / Angstrom
    """
    E_eV, F_eV_A = potential.energy_forces( x )

    E_kj_mol = E_eV / KJ_MOL_TO_EV
    F_kJ_mol_A = F_eV_A / KJ_MOL_TO_EV

    return E_kj_mol, F_kJ_mol_A


def relaxMolecule( potential: EnergyForceEvaluator | str,
                   x0 : np.ndarray, 
                   *,
                   Z : np.ndarray | None = None,
                   minimizer : str = "Adam",
                   force_tol : float = 0.02 / KJ_MOL_TO_EV,
                   max_steps : int = 1000,
                   verbose : bool = False,
                   returnOptimizationHistory : bool = False ) -> np.ndarray | tuple[np.ndarray,list]:
    """ General entry point for all relaxation codes"""
    if isinstance( potential, str ):
        if Z is None:
            raise ValueError( "Z is required when the first argument is a force-field string." )
        potential = make_potential( potential, Z )

    if minimizer.lower() == "adam":
        x_opt, info = minimize_with_adam( potential, x0, force_tol, max_steps, verbose=verbose )
    elif minimizer.lower() == "lbfgs":
        x_opt, info = minimize_with_lbfgs( potential, x0, force_tol, max_steps, verbose=verbose )
    else:
        raise ValueError( f"Minimizer of type {minimizer} is not supported." )

    if returnOptimizationHistory:
        return x_opt, info
    return x_opt

def minimize_with_adam( potential : EnergyForceEvaluator,
                        positions_A: np.ndarray,
                        force_tolerance_kJ_mol_A: float,
                        max_steps: int = 10_000,
                        lr0: float = 1e-2,  # Angstrom-scale learning rate
                        max_step_A: float = 0.02,   # cap largest atom displacement per step
                        lr_min : float = 1e-7,
                        verbose : bool = False, ) -> tuple[np.ndarray, list]:
    """
    Adam minimizer using xTB/OpenMM forces.

    Internal Torch coordinate units:
        positions: Angstrom
        gradients: eV / Angstrom
        energy printed in kJ/mol and eV

    Returns
    -------
    R_final_A: ndarray
        Optimized atomic coordinates in Angstrom
    info : dict
        Dictionary with optimization run and convergence information.
    """

    R = pt.nn.Parameter( pt.tensor(positions_A, dtype=pt.float64) )
    optimizer = pt.optim.Adam([R], lr=lr0)
    def set_optimizer_lr( lr: float):
        for group in optimizer.param_groups:
            group["lr"] = lr
    lr = lr0

    step_count = 0
    previous_R = R.detach().clone()
    history = []
    lr_history = []
    print("step, energy_kJ_mol, energy_eV, max_force_eV_A, step_A")
    while lr >= lr_min:
        optimizer.zero_grad(set_to_none=True)

        R_np = R.detach().cpu().numpy()
        
        try:
            energy_kJ_mol, forces_kJ_mol_A = evaluateEnergyAndForces( potential, R_np )
        except Exception as exc:
            print(f"xTB failed at step {step_count}: {exc}")
            with pt.no_grad():
                R.copy_(previous_R)
            break

        max_force_kJ_mol_A = float( np.linalg.norm(forces_kJ_mol_A, axis=1).max() )
        mean_force_kJ_mol_A = float( np.linalg.norm(forces_kJ_mol_A, axis=1).mean() )

        # Torch minimizes using grad = dE/dR.
        # xTB gives force = -dE/dR.
        grad_kJ_mol_A = -forces_kJ_mol_A
        grad = pt.tensor( grad_kJ_mol_A, dtype=R.dtype )

        # Clip per-atom gradient norm so one atom can't dominate the step.
        max_grad_kJ_mol_A = 1.0                                     # per-atom cap, kJ/mol/Å
        norms = pt.linalg.norm( grad, dim=1, keepdim=True )          # (n_atoms, 1)
        grad = grad * ( max_grad_kJ_mol_A / norms ).clamp( max=1.0 ) # scale down only over-cap atoms
        R.grad = grad

        old_R = R.detach().clone()
        optimizer.step()

        # Cap the maximum coordinate displacement per optimizer step.
        with pt.no_grad():
            displacement = R - old_R
            disp_norms = pt.linalg.norm(displacement, dim=1)
            max_disp = pt.max(disp_norms).item()
            rmsd = np.sqrt( float( np.mean( np.sum( (R.cpu().numpy() - positions_A)**2, axis=1 ) ) ) )

            if max_disp > max_step_A:
                displacement *= max_step_A / max_disp
                R.copy_(old_R + displacement)
                max_disp = max_step_A

        row = { "step": step_count,
                "max_force_rms": max_force_kJ_mol_A,
                "mean_force_rms": mean_force_kJ_mol_A,
                "energy_kJ_mol": energy_kJ_mol,
                "rmsd": rmsd,
                "max_step_A": max_disp,
        }
        history.append( row )
        lr_history.append( row )

        if verbose:
            print( f"{step_count:5d}, ", f"{energy_kJ_mol: .10f} [kJ/mol], ", f"{max_force_kJ_mol_A: .8f} [kJ/(mol A)], ", f"{max_disp: .6f} [A]", file=sys.stderr )
        if max_force_kJ_mol_A < force_tolerance_kJ_mol_A:
            print("Converged.")
            break

        if has_started_increasing( lr_history, window=6, rel_increase=0.02, ):
            print( 'Adam started to increase. Reducing lr. ', file=sys.stderr )
            lr = 0.5*lr
            set_optimizer_lr( lr )
            lr_history.clear()
        elif has_plateaued( lr_history, window=6, rel_tol=0.02 ):
            print( 'Adam Plateau Reached. Reducing lr. ', file=sys.stderr )
            lr = 0.5*lr
            set_optimizer_lr( lr )
            lr_history.clear()

        previous_R = old_R
        step_count += 1
        if step_count > max_steps:
            print( 'Adam Reached the maximum number of iterations. Returning ', file=sys.stderr )
            break

    # Re-evaluate final geometry, because the last logged energy/force was
    # before the final Adam coordinate update.
    R_final_A = R.detach().cpu().numpy()
    return R_final_A, history


def minimize_with_lbfgs( potential : EnergyForceEvaluator,
                         positions_A: np.ndarray,
                         force_tolerance_kJ_mol_A: float,
                         max_steps: int = 1000,
                         history_size: int = 20,
                         verbose : bool = False, ) -> tuple[np.ndarray, list]:
    """
    L-BFGS minimizer with a strong-Wolfe line search, using xTB forces.

    Unlike Adam, the line search rejects any step that does not decrease the
    energy, so there is no initial energy jump, and the step shrinks to zero as
    the forces vanish, so the geometry settles at the minimum instead of
    drifting. The line search is the trust mechanism, so there is no learning
    rate schedule, gradient clip, or per-step displacement cap.

    Internal Torch coordinate units:
        positions: Angstrom
        gradients: kJ/mol/Angstrom
        energy: kJ/mol

    Returns
    -------
    R_final_A : (n_atoms, 3) optimized coordinates, Angstrom.
    history : list of per-step dicts (same fields as minimize_with_adam).
    """
    R = pt.nn.Parameter( pt.tensor(positions_A, dtype=pt.float64) )
    optimizer = pt.optim.LBFGS( [R],
                                lr=1.0,
                                max_iter=1,            # one L-BFGS iteration per outer step, so we can log and test convergence between steps
                                max_eval=30,           # Make sure line search does not default to one step.
                                history_size=history_size,
                                line_search_fn="strong_wolfe", )

    # The line search calls the closure many times; cache the latest energy and
    # forces it computed. strong_wolfe leaves the parameters exactly at the point
    # of its last closure evaluation, so this cache holds the values at the
    # current geometry R -- logging reuses it instead of re-evaluating xTB.
    last = {}
    def closure():
        optimizer.zero_grad( set_to_none=True )
        E, F = evaluateEnergyAndForces( potential, R.detach().cpu().numpy() )
        last["E"], last["F"] = E, F
        R.grad = pt.tensor( -F, dtype=R.dtype )    # LBFGS reads R.grad; grad = dE/dR = -F
        return pt.tensor( E, dtype=R.dtype ) # Always return the energy ('loss')

    history = []
    prev_R = R.detach().clone()

    def log_step( step_count : int ) -> float:
        """Append a history row from the cached energy/forces, return max force."""
        E, F = last["E"], last["F"]
        R_np = R.detach().cpu().numpy()
        force_norms = np.linalg.norm( F, axis=1 )
        max_force = float( force_norms.max() )
        max_step = float( np.linalg.norm( R_np - prev_R.cpu().numpy(), axis=1 ).max() )
        rmsd = np.sqrt( float( np.mean( np.sum( (R_np - positions_A)**2, axis=1 ) ) ) )

        history.append( { "step": step_count,
                          "max_force_rms": max_force,
                          "mean_force_rms": float( force_norms.mean() ),
                          "energy_kJ_mol": E,
                          "rmsd": rmsd,
                          "max_step_A": max_step, } )
        if verbose:
            print( f"{step_count:5d},  {E: .10f} [kJ/mol],  {max_force: .8f} [kJ/(mol A)],  {max_step: .6f} [A]", file=sys.stderr )
        return max_force

    # Evaluate the initial geometry once to seed the cache, log step 0, and stop
    # early if it is already relaxed.
    closure()
    if log_step( 0 ) < force_tolerance_kJ_mol_A:
        return R.detach().cpu().numpy(), history

    for step_count in range( 1, max_steps + 1 ):
        try:
            optimizer.step( closure )
        except Exception as exc:
            print( f"xTB failed at step {step_count}: {exc}", file=sys.stderr )
            with pt.no_grad():
                R.copy_( prev_R )
            break

        max_force = log_step( step_count )
        prev_R = R.detach().clone()
        if max_force < force_tolerance_kJ_mol_A:
            if verbose:
                print( "Converged.", file=sys.stderr )
            break

    return R.detach().cpu().numpy(), history