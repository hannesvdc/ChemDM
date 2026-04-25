import numpy as np
import torch as pt
import openmm as mm
import openmm.unit as unit

KJ_MOL_NM_TO_EV_A = 0.001036427230133138

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

def minimize_with_adam( context : mm.Context,
                        positions_A: np.ndarray,
                        n_steps: int = 1000,
                        lr: float = 1e-3,  # Angstrom-scale learning rate
                        force_tolerance_ev_A: float = 0.02,
                        max_step_A: float = 0.02,   # cap largest atom displacement per step
                        ):

    """
    Adam minimizer using xTB/OpenMM forces.

    Internal Torch coordinate units:
        positions: Angstrom
        gradients: eV / Angstrom
        energy printed in kJ/mol and eV
    """
    R = pt.nn.Parameter( pt.tensor(positions_A, dtype=pt.float64) )
    optimizer = pt.optim.Adam([R], lr=lr)

    previous_R = R.detach().clone()
    print("step, energy_kJ_mol, energy_eV, max_force_eV_A, step_A")

    for step in range(n_steps):
        optimizer.zero_grad(set_to_none=True)
        R_np = R.detach().cpu().numpy()
        try:
            energy_kj_mol, forces_kj_mol_nm = evaluate_openmm_xtb( context, R_np )
        except Exception as exc:
            print(f"xTB/OpenMM failed at step {step}: {exc}")
            with pt.no_grad():
                R.copy_(previous_R)
            break

        forces_ev_A = forces_kj_mol_nm * KJ_MOL_NM_TO_EV_A

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

        max_force = np.linalg.norm(forces_ev_A, axis=1).max()
        energy_eV = energy_kj_mol * 0.01036427230133138
        if step % 10 == 0 or step == n_steps - 1:
            print( f"{step:5d}, ", f"{energy_kj_mol: .10f}, ", f"{energy_eV: .10f}, ", f"{max_force: .8f}, ", f"{max_disp: .6f}" )

        if max_force < force_tolerance_ev_A:
            print("Converged.")
            break

        previous_R = old_R

    return R.detach().cpu().numpy()