import numpy as np
from ase import Atoms
from xtb.ase.calculator import XTB

class XTBPotential:
    """
    Main class abstracting the interface to xTB (in ASE). After setup, the user
    can calculate the forces and energies using the `energy_forces` method.

    Unit convention: energies are returned in electron volts [eV], forces in [eV/A].
    XTBPotential will assume input positions are in Angstrom - no conversion happens at this level.
    """
    def __init__( self,
                  Z: np.ndarray,
                  charge: int = 0,
                  uhf: int = 0,
                  method: str = "GFN2-xTB",
                  accuracy: float = 1.0,
                  electronic_temperature: float = 300.0,
                  max_iterations: int = 250,
                  solvent: str | None = None, ):
        self.Z = np.asarray(Z, dtype=int)
        self.charge = charge
        self.uhf = uhf
        self.method = method
        self.accuracy = accuracy
        self.electronic_temperature = electronic_temperature
        self.max_iterations = max_iterations
        self.solvent = solvent

        kwargs = dict( method=method, accuracy=accuracy, electronic_temperature=electronic_temperature, max_iterations=max_iterations, )
        if solvent is not None:
            kwargs["solvent"] = solvent

        self.atoms = Atoms(numbers=self.Z, positions=np.zeros((len(self.Z), 3)))
        self.atoms.calc = XTB( self.atoms, **kwargs )

    def energy_forces(self, x_A: np.ndarray) -> tuple[float, np.ndarray]:
        """
        x_A: positions in Angstrom, shape (n_atoms, 3)

        returns:
            energy_eV: float
            forces_eV_per_A: shape (n_atoms, 3)
        """
        self.atoms.positions = np.asarray(x_A, dtype=float)

        # ASE/xTB returns eV and eV/Angstrom
        energy = float(self.atoms.get_potential_energy())
        forces = np.asarray(self.atoms.get_forces(), dtype=float)

        return energy, forces