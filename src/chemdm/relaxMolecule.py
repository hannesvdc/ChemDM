import sys

import numpy as np
import torch as pt

from chemdm.Constants import *
from chemdm.xtbSetup import XTBPotential
from chemdm.diagnostics import *

from typing import Optional 

def evaluateEnergyAndForces( xtb: XTBPotential, 
                             x: np.ndarray):
    """
    Evaluate the energy and force field in the given positions (in Angstrom). 
    Returns energy in kJ/mol and forces in kJ / mol / A.

    x: (n_atoms, 3), Angstrom

    returns:
        energy_kJ_mol: float
        forces_eV_A: (n_atoms, 3), eV / Angstrom
    """
    E_eV, F_eV_A = xtb.energy_forces( x )

    E_kj_mol = E_eV / KJ_MOL_TO_EV
    F_kJ_mol_A = F_eV_A / KJ_MOL_TO_EV

    return E_kj_mol, F_kJ_mol_A


def relaxMolecule( xtb : XTBPotential,
                   x0 : np.ndarray, 
                   minimizer : str = "Adam",
                   force_tol : float = 0.02 / KJ_MOL_TO_EV,
                   max_steps : int = 1000,
                   verbose : bool = False,
                   returnOptimizationHistory : bool = False ) -> np.ndarray | tuple[np.ndarray,list]:
    """ General entry point for all relaxation codes"""
    if minimizer.lower() == "adam":
        x_opt, info = minimize_with_adam( xtb, x0, force_tol, max_steps, verbose=verbose )
    else:
        raise ValueError( f"Minimizer of type {minimizer} is not supported." )

    if returnOptimizationHistory:
        return x_opt, info
    return x_opt

def minimize_with_adam( xtb : XTBPotential,
                        positions_A: np.ndarray,
                        force_tolerance_kJ_mol_A: float,
                        n_steps: int = 10_000,
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
    #for step in range(n_steps):
    while lr >= lr_min:
        optimizer.zero_grad(set_to_none=True)

        R_np = R.detach().cpu().numpy()
        
        try:
            energy_kJ_mol, forces_kJ_mol_A = evaluateEnergyAndForces( xtb, R_np )
        except Exception as exc:
            print(f"xTB/OpenMM failed at step {step_count}: {exc}")
            with pt.no_grad():
                R.copy_(previous_R)
            break

        max_force_kJ_mol_A = float( np.linalg.norm(forces_kJ_mol_A, axis=1).max() )
        mean_force_kJ_mol_A = float( np.linalg.norm(forces_kJ_mol_A, axis=1).mean() )

        # Torch minimizes using grad = dE/dR.
        # OpenMM gives force = -dE/dR.
        grad_kJ_mol_A = -forces_kJ_mol_A
        R.grad = pt.tensor( grad_kJ_mol_A, dtype=R.dtype )

        old_R = R.detach().clone()
        optimizer.step()

        # Cap the maximum coordinate displacement per optimizer step.
        with pt.no_grad():
            displacement = R - old_R
            disp_norms = pt.linalg.norm(displacement, dim=1)
            max_disp = pt.max(disp_norms).item()
            rmsd = np.sqrt( float( np.mean( (R.cpu().numpy() - positions_A)**2, axis=1 ) ) )

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

        if verbose and (step_count % 10 == 0 or step_count == n_steps - 1):
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

    # Re-evaluate final geometry, because the last logged energy/force was
    # before the final Adam coordinate update.
    R_final_A = R.detach().cpu().numpy()    
    return R_final_A, history