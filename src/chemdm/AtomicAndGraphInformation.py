import torch as pt
import torch.nn as nn

from chemdm.MoleculeGraph import Molecule

from typing import ClassVar

class MoleculeInformation( nn.Module ):
    """
    Calculate and assign all relevant chemical information to each atom in the molecule. 
    Implemented as a torch Module for generality.

    At this point, the relevant atomic and chemical information is
    -   One-hot encoding of the atomic kind (H, C, O or N)
    -   Atomic mass in amu
    -   Number of bonded atoms
    -   Number of bonded hydrogens, even if hydrogens are explicitly included in the graph structure.

    More chemical information will be included if required.
    """

    ALLOWED_ATOMIC_NUMBERS: ClassVar[tuple[int, ...]] = (1, 6, 7, 8)

    def __init__( self ) -> None:
        super().__init__()

        # Create a vectorized lookup structure for atomic masses and kinds (one-hot encoded).
        atom_type_rows = { 1 : 0, 6 : 1, 7 : 2, 8 : 3}
        atom_kind_ohe = pt.zeros( (max(atom_type_rows.keys())+1, len(atom_type_rows.keys())) )
        for k, v in atom_type_rows.items():
            atom_kind_ohe[k, v] = 1.0
        
        atom_masses = { 1 : 1.008, 6 : 12.001, 7 : 14.007, 8 : 15.999 }
        atom_mass_lookup = pt.full( (max(atom_masses.keys()) + 1,), -1.0 )
        for k, v in atom_masses.items():
            atom_mass_lookup[k] = float(v)

        allowed_lookup = pt.zeros(max(self.ALLOWED_ATOMIC_NUMBERS) + 1, dtype=pt.bool)
        allowed_lookup[list(self.ALLOWED_ATOMIC_NUMBERS)] = True

        # In case this layer moves to another device
        self.register_buffer("atom_kind_ohe", atom_kind_ohe)
        self.register_buffer("atom_mass_lookup", atom_mass_lookup)
        self.register_buffer("allowed_lookup", allowed_lookup)

    def numberOfOutputs( self ) -> int:
        """
        Returns the number of chemical features associated to each atom. 
        Useful for constructing neural network architectures.
        """
        return len(self.ALLOWED_ATOMIC_NUMBERS) + 3
    
    def forward( self, molecule : Molecule ) -> pt.Tensor:
        """
        Compute relevant chemical information per atom based on kind and bond graph.

        Arguments
        ---------
        molecule : Molecule
            Molecule for which to compute atomic information.

        Returns
        -------
        chemical_knowledge: pt.Tensor of shape ( len(molecule.Z), n_outputs )
        """
        Z = molecule.Z
        n_atoms = len(Z)
        valid = self.allowed_lookup[Z]
        assert pt.all(valid), f"Unsupported atomic numbers found: {Z[~valid].tolist()}"

        # Lookup elemental features
        atom_kinds = self.atom_kind_ohe[Z, :]   # (n_atoms, 4)
        atom_masses = (self.atom_mass_lookup[Z] )[:, None] # (n_atoms, 1)
        atom_masses = atom_masses / 16.0 # explicitly normalize.

        # Count the number of neighbors
        src = molecule.edge_index[:,0]
        degree = pt.bincount( src, minlength=n_atoms )
        degree = degree[:,None] / 4.0 # Can be more than 4, but just a sensible normalization

        # Count the number of bonded hydrogen atoms
        dest = molecule.edge_index[:,1] 
        is_hydrogen = (Z[dest] == 1).to( dtype=molecule.x.dtype )
        hydrogen_count = pt.bincount( src, weights=is_hydrogen, minlength=n_atoms )
        hydrogen_count = hydrogen_count[:,None] / 4.0

        # Store everything in one tensor
        output_tensor = pt.cat( (atom_kinds, atom_masses, degree, hydrogen_count), dim=1 )
        return output_tensor