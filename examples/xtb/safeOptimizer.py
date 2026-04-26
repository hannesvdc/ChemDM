import numpy as np
import torch as pt
import openmm as mm
import openmm.unit as unit

KJ_MOL_NM_TO_EV_A = 0.001036427230133138
KJ_MOL_TO_EV = 0.01036427230133138

def evaluate_openmm_xtb(context, positions_A: np.ndarray):
    """
    positions_A: (N, 3), Angstrom

    Returns
    -------
        energy_kj_mol: float
        forces_kj_mol_nm: (N, 3), kJ/(mol nm)
    """
    context.setPositions(positions_A * unit.angstrom)

    state = context.getState(getEnergy=True, getForces=True)
    energy_kj_mol = state.getPotentialEnergy().value_in_unit( unit.kilojoule_per_mole )
    forces_kj_mol_nm = state.getForces(asNumpy=True).value_in_unit( unit.kilojoule_per_mole / unit.nanometer ) # type: ignore

    return float(energy_kj_mol), np.asarray(forces_kj_mol_nm, dtype=np.float64)

def displacement_stats(R0_A: np.ndarray, R1_A: np.ndarray) -> dict:
    displacement = np.linalg.norm(R1_A - R0_A, axis=1)
    return {
        "rmsd_displacement_A": float( np.sqrt( np.mean(displacement**2) ) ),
        "mean_displacement_A": float( np.mean(displacement) ),
        "max_displacement_A": float( np.max(displacement) ),
    }

def minimize_with_adam( context : mm.Context,
                        positions_A: np.ndarray,
                        n_steps: int = 10_000,
                        lr: float = 1e-3,  # Angstrom-scale learning rate
                        force_tolerance_ev_A: float = 0.02,
                        max_step_A: float = 0.02,   # cap largest atom displacement per step
                        print_every : int = 10,
                        ) -> tuple[np.ndarray, dict]:

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
    R0_A = np.copy( positions_A )

    R = pt.nn.Parameter( pt.tensor(positions_A, dtype=pt.float64) )
    optimizer = pt.optim.Adam([R], lr=lr)

    previous_R = R.detach().clone()

    status = "max_steps"
    error_type = None
    error_message = None
    initial_energy_kj_mol = None
    initial_max_force_ev_A = None
    final_energy_kj_mol = None
    final_max_force_ev_A = None

    print("step, energy_kJ_mol, energy_eV, max_force_eV_A, step_A")

    for step in range(n_steps):
        optimizer.zero_grad(set_to_none=True)

        R_np = R.detach().cpu().numpy()
        
        try:
            energy_kj_mol, forces_kj_mol_nm = evaluate_openmm_xtb( context, R_np )
        except Exception as exc:
            print(f"xTB/OpenMM failed at step {step}: {exc}")
            error_type = type(exc).__name__
            error_message = str(exc)
            status = "failed"
            with pt.no_grad():
                R.copy_(previous_R)
            break

        forces_ev_A = forces_kj_mol_nm * KJ_MOL_NM_TO_EV_A
        max_force_ev_A = float(np.linalg.norm(forces_ev_A, axis=1).max())
        energy_eV = float(energy_kj_mol * KJ_MOL_TO_EV)

        if step == 0:
            initial_energy_kj_mol = float(energy_kj_mol)
            initial_max_force_ev_A = max_force_ev_A

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

        final_energy_kj_mol = float(energy_kj_mol)
        final_max_force_ev_A = max_force_ev_A
        if step % print_every == 0 or step == n_steps - 1:
            print( f"{step:5d}, ", f"{energy_kj_mol: .10f}, ", f"{energy_eV: .10f}, ", f"{max_force_ev_A: .8f}, ", f"{max_disp: .6f}" )
        if max_force_ev_A < force_tolerance_ev_A:
            print("Converged.")
            status = "converged"
            break

        previous_R = old_R

    # Re-evaluate final geometry, because the last logged energy/force was
    # before the final Adam coordinate update.
    R_final_A = R.detach().cpu().numpy()
    try:
        final_energy_kj_mol, final_forces_kj_mol_nm = evaluate_openmm_xtb( context, R_final_A, )
        final_forces_ev_A = final_forces_kj_mol_nm * KJ_MOL_NM_TO_EV_A
        final_max_force_ev_A = float(np.linalg.norm(final_forces_ev_A, axis=1).max())
    except Exception as exc:
        if status != "failed":
            status = "failed_final_eval"
            error_type = type(exc).__name__
            error_message = str(exc)

    disp_info = displacement_stats(R0_A, R_final_A)
    info = {
        "status": status,
        "lr": lr,
        "force_tolerance_ev_A": force_tolerance_ev_A,
        "max_step_A": max_step_A,
        "initial_energy_kj_mol": initial_energy_kj_mol,
        "initial_max_force_ev_A": initial_max_force_ev_A,
        "final_max_force_ev_A": final_max_force_ev_A,
        **disp_info,
        "error_type": error_type,
        "error_message": error_message,
    }
    info["final_energy_kj_mol"] = float(final_energy_kj_mol) if final_energy_kj_mol is not None else None
    info[ "delta_energy_kj_mol"] = float(final_energy_kj_mol) - float(initial_energy_kj_mol) \
                                    if final_energy_kj_mol is not None and initial_energy_kj_mol is not None else None

    return R_final_A, info