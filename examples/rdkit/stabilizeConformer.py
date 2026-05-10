import openmm as mm
import openmm.unit as unit

import numpy as np
from scipy.optimize import fmin_l_bfgs_b

def energy_and_gradient( context : mm.Context, 
                         x_flat_angstrom : np.ndarray
                        ) -> tuple[float, np.ndarray]:
    positions_angstrom = x_flat_angstrom.reshape((-1, 3))
    context.setPositions( positions_angstrom * unit.angstrom )

    state = context.getState(getEnergy=True, getForces=True)
    energy = state.getPotentialEnergy().value_in_unit( unit.kilojoules_per_mole )
    forces = state.getForces(asNumpy=True).value_in_unit( unit.kilojoules_per_mole / unit.angstrom )

    gradient = -forces
    return float(energy), gradient.reshape(-1)

def stabilizeConformer( context : mm.Context,
                        x : np.ndarray,
                        force_tol : float = 1.0, # [kJ / mol / A]
                       ) -> tuple[np.ndarray, float, np.ndarray, dict]:
    assert x.ndim == 2 and x.shape[1] == 3, f"`x` must have shape (N,3) but got {x.shape}."
    x0 = x.flatten()

    assert force_tol > 0, f"The force tolerance must be strictly positive."
    x_min, E_opt, info = fmin_l_bfgs_b( lambda m : energy_and_gradient( context, m ), x0,
                              fprime=None, approx_grad=False, pgtol=force_tol )
    print( info )

    E_opt, F_opt = energy_and_gradient( context, x_min )
    return x_min.reshape( (-1, 3) ), E_opt, F_opt, info