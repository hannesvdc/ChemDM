"""
Discover Stable Conformer from 2D Molecular Graph: RDKIT initial guess + xTB-optim refinement.
"""
from __future__ import annotations
import sys

import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem

from chemdm.xtbSetup import XTBPotential
from chemdm.relaxMolecule import minimize_with_adam
from chemdm.progress import ProgressCallback
from chemdm.Cluster import rmsd_clustering, post_relaxation_rmsd_clustering


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

def run( input_data: dict,
         on_progress : ProgressCallback ) -> dict:
    """
    Empty implementation for now.
    """
    smiles = input_data["smiles"]
    n_conformers = int( input_data.get("n_conformers", 1000 ) )
    theory = input_data.get( "theory", "xtb" )
    force_tol = float( input_data.get( "force_tolerance", 5.0) )
    max_optimizer_steps = int( input_data.get( "max_optimizer_steps", 250) )
    rmsd_tol = float( input_data.get("rmsd_tol", 1.0) )

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
    params.pruneRmsThresh = -1.0  # We do our own clustering.
    on_progress( "Generation", f"Generating {n_conformers} initial conformers", fraction=0.0 )
    conf_ids = AllChem.EmbedMultipleConfs( mol_with_h, numConfs=n_conformers, params=params )
    raw_conformers = [ np.asarray( mol_with_h.GetConformer(conf_id).GetPositions(), dtype=float, ) for conf_id in conf_ids ]
    pre_conformers, _, cluster_sizes = rmsd_clustering( Z, raw_conformers, rmsd_tol )
    on_progress( "Generation", f"Generated {len(conf_ids)} possibly distinct conformers ", fraction=0.1 )

    # Stabilize all generated conformers.
    lr0 = 1e-3
    optimal_conformers = []
    energies = []
    force_norms = []
    current_fraction = on_progress.getTotalProgress()
    remaining_fraction = (0.9 - current_fraction)
    for conf_id in range( len(pre_conformers) ):
        print( f'\nStabilizing Conformer {conf_id}.', file=sys.stderr )
        on_progress( "Stabilization", f"Stabilizing Conformation {conf_id+1}/{len(pre_conformers)}", 
                    fraction=current_fraction + (conf_id+1)/len(pre_conformers)*remaining_fraction )
        conf_opt, history = minimize_with_adam( xtb, pre_conformers[conf_id], force_tol, max_optimizer_steps, lr0, verbose=True )
        E_opt = history[-1]["energy_kJ_mol"]
        F_opt = history[-1]["max_force_rms"]
        print( f'Conformer {conf_id} stabilized to E = {E_opt} and |F| = {F_opt}.', file=sys.stderr )

        optimal_conformers.append( conf_opt )
        energies.append( E_opt )
        force_norms.append( F_opt )

    print( 'Clustering', file=sys.stderr )
    on_progress( "Clustering", f"Clustering Stable Conformers", fraction=0.9 )
    optimal_conformers, energies, force_norms, _, cluster_sizes = post_relaxation_rmsd_clustering( Z, optimal_conformers, energies, force_norms, cluster_sizes )
    print( f'Found {len(optimal_conformers)} non-trivial conformers.', file=sys.stderr )

    output_data = { "Z" : Z,
                    "bonds" : rdkit_mol_to_bond_list(mol_with_h),
                    "conformers" : [{"x" : optimal_conformers[ii], "energy" : energies[ii], "force_norm" : force_norms[ii], "cluster_size": cluster_sizes[ii]} for ii in range(len(optimal_conformers))]}
    return output_data