import torch as pt
import random

from chemdm.MoleculeGraph import MoleculeGraph, BatchedMoleculeGraph

def testSimpleBatching():
    mol1 = MoleculeGraph(
        Z=pt.tensor([6, 1]),                 # 2 atoms
        x=pt.tensor([[0., 0., 0.],
                     [1., 0., 0.]]),
        bonds=pt.tensor([[0, 1], 
                         [1, 0]])
    )

    mol2 = MoleculeGraph(
        Z=pt.tensor([8, 1, 1]),              # 3 atoms
        x=pt.tensor([[0., 0., 0.],
                     [0., 1., 0.],
                     [0., 0., 1.]]),
        bonds=pt.tensor([[0, 1],
                         [0, 2]])
    )

    mol3 = MoleculeGraph(
        Z=pt.tensor([7]),                    # 1 atom
        x=pt.tensor([[2., 2., 2.]]),
        bonds=pt.empty((0, 2), dtype=pt.long)
    )

    mol4 = MoleculeGraph(
        Z=pt.tensor([7, 7]),                    # 1 atom
        x=pt.tensor([[2., 2., 2.],
                     [3., 3., 3.]]),
        bonds=pt.tensor([[0, 1],
                         [1, 0]])
    )

    # --- batch them ---
    batch = BatchedMoleculeGraph([mol1, mol2, mol3, mol4])
    print(batch)

    # --- inspect result ---
    print("Z:")
    print(batch.Z)
    print()

    print("x:")
    print(batch.x)
    print()

    print("edge_index:")
    print(batch.edge_index)
    print()

    print("molecule_id:")
    print(batch.molecule_id)
    print()

    # --- expected values ---
    expected_Z = pt.tensor([6, 1, 8, 1, 1, 7, 7, 7])

    expected_edge_index = pt.tensor([
        [0, 1],   # mol1 unchanged
        [1, 0],
        [2, 3],   # mol2 shifted by +2
        [2, 4],
        [6, 7],
        [7, 6]
    ], dtype=pt.long)

    expected_molecule_id = pt.tensor([0, 0, 1, 1, 1, 2, 3, 3], dtype=pt.long)

    # --- checks ---
    assert pt.equal(batch.Z, expected_Z)
    assert pt.equal(batch.edge_index, expected_edge_index)
    assert pt.equal(batch.molecule_id, expected_molecule_id)

    print("All simple tests tests passed.")

def performanceBatching():
    molecule_list = []
    n_molecules = 256
    for n in range( n_molecules ):
        n_atoms = random.randint(1, 11)
        Z = pt.randint(1, 9, (n_atoms,))
        x = pt.randn( (n_atoms, 3) )
        n_bonds = random.randint(1, 2*n_atoms)
        G = pt.randint(0, n_atoms, (n_bonds,2))
        molecule = MoleculeGraph( Z, x, G )
        molecule_list.append(molecule)

    print( "Merging batch" )
    batched_molecule = BatchedMoleculeGraph( molecule_list )
    print( "Done merging" )
    
    offset = 0
    bond_offset = 0
    for n in range(n_molecules):
        n_atoms = len(molecule_list[n].Z)
        assert pt.all(batched_molecule.Z[offset:offset+n_atoms] == molecule_list[n].Z)
        assert pt.all(batched_molecule.x[offset:offset+n_atoms,:] == molecule_list[n].x)
        n_bonds = molecule_list[n].edge_index.shape[0]
        assert pt.all(batched_molecule.edge_index[bond_offset:bond_offset+n_bonds,:] == molecule_list[n].edge_index + offset)
        offset += n_atoms
        bond_offset += n_bonds
    print( 'All checks passed')

if __name__ == '__main__':
    testSimpleBatching()
    performanceBatching()