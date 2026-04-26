import numpy as np
import openmm as mm
import openmm.unit as unit
from openmmxtb import XtbForce

def create_xtb_system( atomic_numbers: np.ndarray, charge: float = 0.0, multiplicity: int = 1, method: str = "GFN2xTB", ) -> mm.System:
    """
    Create an OpenMM System containing only an xTB force.

    Parameters
    ----------
    atomic_numbers:
        Array of atomic numbers, shape (N,).
    charge:
        Total molecular charge.
    multiplicity:
        Spin multiplicity. Use 1 for closed-shell singlet.
    method:
        One of: "GFN1xTB", "GFN2xTB", "GFNFF".
    """
    atomic_numbers = np.asarray(atomic_numbers, dtype=int)
    n_atoms = len(atomic_numbers)
    system = mm.System()

    # Masses are not important for single-point energy/force evaluation,
    # but OpenMM requires particles to exist in the System.
    masses = { 1: 1.008, 6: 12.011, 7: 14.007, 8: 15.999, }
    for Z in atomic_numbers:
        system.addParticle( masses[int(Z)] * unit.dalton ) # type: ignore

    method_map = {
        "GFN1xTB": XtbForce.GFN1xTB,
        "GFN2xTB": XtbForce.GFN2xTB,
        "GFNFF": XtbForce.GFNFF,
    }
    if method not in method_map:
        raise ValueError(f"Unknown xTB method: {method}")

    particle_indices = list(range(n_atoms))
    atomic_numbers_list = [int(z) for z in atomic_numbers]

    xtb_force = XtbForce( method_map[method], charge,  multiplicity, False,  particle_indices, atomic_numbers_list, )
    system.addForce(xtb_force)
    return system

def create_xtb_context(atomic_numbers: np.ndarray) -> mm.Context:
    system = create_xtb_system(atomic_numbers)

    integrator = mm.VerletIntegrator(1.0 * unit.femtosecond)
    platform = mm.Platform.getPlatformByName("CPU")
    properties = {"Threads": "1"}
    context = mm.Context(system, integrator, platform, properties)

    # Keep references alive by attaching them.
    context._system_ref = system
    context._integrator_ref = integrator

    return context