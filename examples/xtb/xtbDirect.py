import numpy as np
from ase import Atoms
from xtb.ase.calculator import XTB
from rdkit import Chem

from chemdm.xtbSetup import create_xtb_context
import openmm.unit as unit

EV_TO_KJ_PER_MOLE = 96.48533212331002

class XTBPotential:
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
    

def compareOpenMMXTB():
    mol_with_h = Chem.MolFromMolFile( './../rdkit/conformers_ad/conformer_0.mol' )
    Z = np.array(  [atom.GetAtomicNum() for atom in mol_with_h.GetAtoms() ], dtype=np.int64 )
    
    # OpenMM context
    ad_openmm = create_xtb_context( Z )

    # Raw XTB Potential
    xtb = XTBPotential( Z )

    # Compute energies and forces both ways
    x = mol_with_h.GetConformer().GetPositions() # (n_atoms, 3)
    ad_openmm.setPositions( x * unit.angstrom )
    mm_state = ad_openmm.getState( getEnergy=True, getForces=True )
    mm_E = mm_state.getPotentialEnergy().value_in_unit( unit.kilojoule_per_mole )
    mm_forces = mm_state.getForces(asNumpy=True).value_in_unit( unit.kilojoule_per_mole / unit.angstrom ) # type: ignore
    xtb_E, xtb_forces = xtb.energy_forces( x )
    xtb_E = EV_TO_KJ_PER_MOLE * xtb_E
    xtb_forces = EV_TO_KJ_PER_MOLE * xtb_forces

    # Compare
    print( f'Energies {mm_E}, {xtb_E}')
    print( f'Force Difference {mm_forces - xtb_forces}')


if __name__ == '__main__':
    compareOpenMMXTB()