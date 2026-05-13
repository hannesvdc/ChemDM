import numpy as np

from chemdm.xtbSetup import XTBPotential
from safeOptimizer import minimize_with_adam

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

    # xtb uses Angstrom
    system = XTBPotential( atomic_numbers )

    print("Initial single-point:")
    E0, F0 = system.energy_forces( positions_A )
    print(f"  Energy:      {E0:.8f} kJ/mol")
    print(f"  Max |force|: {np.linalg.norm(F0, axis=1).max():.8f} kJ/(mol nm)")
    print()

    minimized_positions_A = minimize_with_adam(
        xtb=system,
        positions_A=positions_A,
        n_steps=1000,
        lr=1e-3,
        force_tolerance_ev_A=0.02,
        max_step_A=0.02, )
    print( minimized_positions_A )

if __name__ == "__main__":
    main()