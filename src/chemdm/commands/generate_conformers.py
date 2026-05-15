"""
Discover Stable Conformer from 2D Molecular Graph: RDKIT initial guess + xTB-optim refinement.
"""
from __future__ import annotations
import sys

import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem

from chemdm.xtbSetup import XTBPotential
from chemdm.geometry import kabsch_align_numpy
from chemdm.relaxMolecule import minimize_with_adam

def rdkit_positions_to_numpy(mol: Chem.Mol, conf_id: int) -> np.ndarray:
    """ RDKIT uses Angstrom internally, and so do we. """
    conf = mol.GetConformer(conf_id)

    positions = np.zeros( (mol.GetNumAtoms(), 3), dtype=float )
    for i in range(mol.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        positions[i] = [p.x, p.y, p.z]

    return positions

def numpy_positions_to_rdkit(mol: Chem.Mol, conf_id: int, positions: np.ndarray) -> None:
    conf = mol.GetConformer(conf_id)
    for i, (x, y, z) in enumerate(positions):
        conf.SetAtomPosition(i, (float(x), float(y), float(z)))

def cluster( Z : np.ndarray, 
             conformers : list[np.ndarray], 
             energies : list[float],
             rmsd_tol : float = 0.5
            ) -> tuple[list[np.ndarray], list[float]]:
    assert len(conformers) == len(energies), f"`confomers` and `energies` must contain the same number of elements."
    if len(conformers) == 0:
        return conformers, energies
    
    initial_conformer = conformers[0] - np.mean( conformers[0], axis=0, keepdims=True )
    optimal_conformers = [ initial_conformer ]
    optimal_energies = [ energies[0] ]
    for n in range(1, len(conformers)):
        print( 'Conformer', n )
        x_conf = conformers[n] - np.mean( conformers[n], axis=0, keepdims=True )

        is_new = True
        for k in range( len(optimal_conformers) ):
            reference_conformer = optimal_conformers[k]

            # Center and Kabsch align
            x_conf_aligned = kabsch_align_numpy( x_conf, reference_conformer, Z )

            # Compute the per-atom RMSD. Ignore hydrogens
            idx = (Z != 1)
            rmsd = np.sqrt(np.mean(np.sum( (x_conf_aligned[idx,:] - reference_conformer[idx,:])**2, axis=1 ) ))
            print(rmsd)
            
            if rmsd <= rmsd_tol:
                is_new = False
                break

        if is_new:
            print( "New Conformer Found!" )
            optimal_conformers.append( x_conf )
            optimal_energies.append( energies[n] )

    return optimal_conformers, energies

def rdkit_mol_to_bond_list(mol: Chem.Mol) -> np.ndarray:
    """
    Return undirected bonds as an array of shape (n_bonds, 2).
    Atom indices are RDKit atom indices, zero-based.
    """
    bonds = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bonds.append((i, j))

    return np.asarray(bonds, dtype=int)

def run( input_data: dict ) -> dict:
    """
    Empty implementation for now.
    """
    smiles = input_data["smiles"]
    n_conformers = int( input_data.get("n_conformers", 1000 ) )
    theory = input_data.get( "theory", "xtb" )
    force_tol = float( input_data.get( "force_tolerance", 5.0) )
    max_optimizer_steps = int( input_data.get( "max_optimizer_steps", 250) )

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError( "Invalid SMILES string" )
    print( Chem.MolToSmiles(mol) )

    # Add hydrogens
    mol_with_h = Chem.AddHs( mol )
    Z = np.array(  [atom.GetAtomicNum() for atom in mol_with_h.GetAtoms() ], dtype=np.int64 )
    if theory.lower() == "xtb":
        xtb = XTBPotential( Z )

    # Generate rotamers / conformers
    params = AllChem.ETKDGv3()
    params.pruneRmsThresh = 0.1  # remove very similar conformers during embedding
    print( f'\nGenerating {n_conformers} initial conformers...', file=sys.stderr )
    conf_ids = AllChem.EmbedMultipleConfs( mol_with_h, numConfs=n_conformers, params=params )
    print( f"Generated {len(conf_ids)} distinct conformers" )
    print( "ConfIds:", list(conf_ids) )

    # Stabilize all generated conformers.
    lr0 = 1e-3
    optimal_conformers = []
    energies = []
    for conf_id in conf_ids:
        print( f'\nStabilizing Conformer {conf_id}.', file=sys.stderr )
        conf_positions = rdkit_positions_to_numpy( mol_with_h, conf_id )
        conf_opt, history = minimize_with_adam( xtb, conf_positions, force_tol, max_optimizer_steps, lr0 )
        E_opt = history[-1]["energy_kJ_mol"]
        F_opt = history[-1]["max_force_rms"]
        print( f'Conformer {conf_id} stabilized to E = {E_opt} and |F| = { np.linalg.norm(F_opt)}.', file=sys.stderr )

        optimal_conformers.append( conf_opt )
        energies.append( E_opt )

    # Cluster to determine unique conformers
    print( 'Clustering', file=sys.stderr )
    optimal_conformers, energies = cluster( Z, optimal_conformers, energies, rmsd_tol=0.5 )

    output_data = { "Z" : Z,
                    "bonds" : rdkit_mol_to_bond_list(mol_with_h),
                    "conformers" : [{"x" : optimal_conformers[ii], "energy" : energies[ii]} for ii in range(len(optimal_conformers))]}
    return output_data