import numpy as np
import torch as pt
import openmm as mm
import openmm.unit as unit

KJ_MOL_TO_EV = 0.01036427230133138 # eV / kJ
KJ_MOL_NM_TO_EV_A = KJ_MOL_TO_EV / 10.0


def evaluateEnergyAndForces( context: mm.Context, 
                             x: np.ndarray):
    """
    Evaluate the energy and force field in the given positions (in Angstrom). 
    Returns energy in eV and forces in eV / A.

    x: (n_atoms, 3), Angstrom

    returns:
        energy_eV: float
        forces_eV_A: (n_atoms, 3), eV / Angstrom
    """
    context.setPositions( x * unit.angstrom )
    state = context.getState(getEnergy=True, getForces=True)

    E_kj_mol = state.getPotentialEnergy().value_in_unit( unit.kilojoule_per_mole )
    F_kj_mol_nm = state.getForces(asNumpy=True).value_in_unit(  unit.kilojoule_per_mole / unit.nanometer ) # type: ignore

    E_eV = E_kj_mol * KJ_MOL_TO_EV
    F_eV_A = np.asarray(F_kj_mol_nm) * KJ_MOL_NM_TO_EV_A

    return E_eV, F_eV_A


def relaxMolecule( context : mm.Context,
                   x0 : np.ndarray, 
                   minimizer : str = "Adam",
                   verbose : bool = False ) -> np.ndarray:
    """ General entry point for all relaxation codes"""
    if minimizer.lower() == "adam":
        x_opt = minimize_with_adam( context, x0, verbose=verbose )
    else:
        raise ValueError( f"Minimizer of type {minimizer} is not supported." )

    return x_opt

def minimize_with_adam( context : mm.Context,
                        positions_A: np.ndarray,
                        n_steps: int = 10_000,
                        lr: float = 1e-3,  # Angstrom-scale learning rate
                        force_tolerance_ev_A: float = 0.02,
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
            energy_eV, forces_ev_A = evaluateEnergyAndForces( context, R_np )
        except Exception as exc:
            print(f"xTB/OpenMM failed at step {step}: {exc}")
            with pt.no_grad():
                R.copy_(previous_R)
            break

        max_force_ev_A = float(np.linalg.norm(forces_ev_A, axis=1).max())

        # Torch minimizes using grad = dE/dR.
        # OpenMM gives force = -dE/dR.
        grad_ev_A = -forces_ev_A
        R.grad = pt.tensor(grad_ev_A, dtype=R.dtype)

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
            print( f"{step:5d}, ", f"{energy_eV: .10f} [eV], ", f"{max_force_ev_A: .8f} [eV/A], ", f"{max_disp: .6f} [A]" )
        if max_force_ev_A < force_tolerance_ev_A:
            print("Converged.")
            break

        previous_R = old_R

    # Re-evaluate final geometry, because the last logged energy/force was
    # before the final Adam coordinate update.
    R_final_A = R.detach().cpu().numpy()    
    return R_final_A