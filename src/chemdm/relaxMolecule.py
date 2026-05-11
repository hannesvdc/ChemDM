import numpy as np
import torch as pt

from chemdm.xtbSetup import XTBPotential

KJ_MOL_TO_EV = 0.01036427230133138 # eV / kJ
KJ_MOL_NM_TO_EV_A = KJ_MOL_TO_EV / 10.0


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
                   verbose : bool = False ) -> np.ndarray:
    """ General entry point for all relaxation codes"""
    if minimizer.lower() == "adam":
        x_opt = minimize_with_adam( xtb, x0, verbose=verbose )
    else:
        raise ValueError( f"Minimizer of type {minimizer} is not supported." )

    return x_opt

def minimize_with_adam( xtb : XTBPotential,
                        positions_A: np.ndarray,
                        n_steps: int = 10_000,
                        lr: float = 1e-3,  # Angstrom-scale learning rate
                        force_tolerance_kJ_mol_A: float = 0.02 / KJ_MOL_TO_EV,
                        max_step_A: float = 0.02,   # cap largest atom displacement per step
                        verbose : bool = False, ) -> np.ndarray:
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
    optimizer = pt.optim.Adam([R], lr=lr)
    previous_R = R.detach().clone()

    print("step, energy_kJ_mol, energy_eV, max_force_eV_A, step_A")
    for step in range(n_steps):
        optimizer.zero_grad(set_to_none=True)

        R_np = R.detach().cpu().numpy()
        
        try:
            energy_kJ_mol, forces_kJ_mol_A = evaluateEnergyAndForces( xtb, R_np )
        except Exception as exc:
            print(f"xTB/OpenMM failed at step {step}: {exc}")
            with pt.no_grad():
                R.copy_(previous_R)
            break

        max_force_kJ_mol_A = float(np.linalg.norm(forces_kJ_mol_A, axis=1).max())

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

            if max_disp > max_step_A:
                displacement *= max_step_A / max_disp
                R.copy_(old_R + displacement)
                max_disp = max_step_A

        if step % 10 == 0 or step == n_steps - 1:
            print( f"{step:5d}, ", f"{energy_kJ_mol: .10f} [kJ/mol], ", f"{max_force_kJ_mol_A: .8f} [kJ/(mol A)], ", f"{max_disp: .6f} [A]" )
        if max_force_kJ_mol_A < force_tolerance_kJ_mol_A:
            print("Converged.")
            break

        previous_R = old_R

    # Re-evaluate final geometry, because the last logged energy/force was
    # before the final Adam coordinate update.
    R_final_A = R.detach().cpu().numpy()    
    return R_final_A