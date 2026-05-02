import numpy as np

import openmm as mm
import openmm.unit as unit
from openmm import app

from chemdm.xtbSetup import create_xtb_system
from safeOptimizer import minimize_with_adam

from typing import Optional

def energy_and_forces( system: mm.System, 
                       positions_angstrom: np.ndarray,
                       integrator : Optional[mm.Integrator] = None ) -> tuple[float, np.ndarray]:
    """
    Evaluate energy and forces for a single geometry using the xTB semi-empirical force field.

    Arguments
    ---------
    system : mm.System
        The molecular system.
    positions_angstrong : ndarray
        Positions of all atoms.
    integrator: mm.Integrator
        Dummy integrator used to construct an OpenMM Context. 
        Pass to avoid creating a new integrator every evaluation. 
        Not used for any simulations.

    Returns
    -------
    energy_kj_mol:
        Potential energy in kJ/mol.
    forces_kj_mol_nm:
        Forces in kJ / (mol Å), shape (N, 3).
    """
    if integrator is None:
        integrator = mm.VerletIntegrator(1.0 * unit.femtosecond) # type: ignore
    context = mm.Context(system, integrator)

    context.setPositions(positions_angstrom * unit.angstrom)
    state = context.getState(getEnergy=True, getForces=True)

    energy_kj_mol = state.getPotentialEnergy().value_in_unit( unit.kilojoule_per_mole )
    forces_kj_mol_nm = state.getForces(asNumpy=True).value_in_unit( unit.kilojoule_per_mole / unit.nanometer ) # type: ignore

    return float(energy_kj_mol), np.asarray(forces_kj_mol_nm)

def get_energy_forces(simulation):
    state = simulation.context.getState(getEnergy=True, getForces=True)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    forces = state.getForces(asNumpy=True).value_in_unit( unit.kilojoule_per_mole / unit.nanometer ) # type: ignore

    return energy, np.asarray(forces)

def main():
    # Formaldehyde: CH2O
    #
    # Atom order:
    #   0 C
    #   1 O
    #   2 H
    #   3 H
    atomic_numbers = np.array([6, 8, 1, 1], dtype=int)
    positions_A = np.array( [
            [0.000000,  0.000000,  0.000000],   # C
            [1.210000,  0.000000,  0.000000],   # O
            [-0.605000,  0.935307,  0.000000],  # H
            [-0.605000, -0.935307,  0.000000],  # H
        ], dtype=float, )

    # OpenMM dynamics usually use nm internally.
    positions_nm = positions_A * 0.1
    system = create_xtb_system( atomic_numbers=atomic_numbers, charge=0.0, multiplicity=1, method="GFN2xTB", )

    # Small timestep for xTB molecular dynamics.
    temperature = 200 * unit.kelvin # type: ignore
    dt = 0.05 * unit.femtosecond # type: ignore
    integrator = mm.LangevinMiddleIntegrator( temperature, 10.0 / unit.picosecond, dt ) # type: ignore
    simulation = app.Simulation( topology=app.Topology(), system=system, integrator=integrator, )
    simulation.context.setPositions(positions_nm * unit.nanometer) # type: ignore

    print("Initial single-point:")
    E0, F0 = get_energy_forces(simulation)
    print(f"  Energy:      {E0:.8f} kJ/mol")
    print(f"  Max |force|: {np.linalg.norm(F0, axis=1).max():.8f} kJ/(mol nm)")
    print()

    # Assign velocities for dynamics.
    simulation.context.setVelocitiesToTemperature(temperature)

    print()
    print("Running short Verlet simulation...")
    print("step, energy_kJ_mol, max_force_kJ_mol_nm")

    n_steps = 1000
    report_every = 10
    for step in range(0, n_steps + 1, report_every):
        E, F = get_energy_forces(simulation)
        max_force = np.linalg.norm(F, axis=1).max()
        print(f"{step:5d}, {E: .8f}, {max_force: .8f}")
        if step < n_steps:
            simulation.step(report_every)

    minimized_positions_A = minimize_with_adam(
        context=simulation.context,
        positions_A=positions_A,
        n_steps=1000,
        lr=1e-3,
        force_tolerance_ev_A=0.02,
        max_step_A=0.02, )
    print( minimized_positions_A )

if __name__ == "__main__":
    main()