"""
Discover Stable Conformer from 2D Molecular Graph: RDKIT initial guess + xTB-optim refinement.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

import numpy as np
import torch as pt

from rdkit import Chem

from chemdm.xtbSetup import XTBPotential
from chemdm.relaxMolecule import minimize_with_lbfgs
from chemdm.progress import ProgressCallback
from chemdm.Cluster import rmsd_clustering, post_relaxation_rmsd_clustering
from chemdm.TorsionalScoreNetwork import TorsionalScoreNetwork

from chemdm.TorsionalDiffusionSampling import sample_conformers_from_mol

_REPO_ROOT = Path(__file__).resolve().parents[3]
_XTB_DIR = _REPO_ROOT / "examples" / "xtb"
if str(_XTB_DIR) not in sys.path:
    sys.path.insert(0, str(_XTB_DIR))

def rdkit_mol_to_bond_list( mol: Chem.Mol ) -> np.ndarray:
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

def load_torsional_diffusion_model( device : pt.device = pt.device('cpu') ) -> TorsionalScoreNetwork:
    """Instantiate the score network and load weights. Accepts both a bare
    state_dict (best.pt) and a training checkpoint dict carrying "model"."""
    model_path = os.environ.get( "CHEMDM_TORSIONAL_DIFFUSION_MODEL", str(_REPO_ROOT / "models" / "torsional_diffusion.pt"), )

    model = TorsionalScoreNetwork().to( device=device, dtype=pt.float32 )
    state = pt.load( model_path, map_location=device, weights_only=True )
    model.load_state_dict( state )
    model.eval()
    return model

def run( input_data: dict,
         on_progress : ProgressCallback,
         td_network : TorsionalScoreNetwork ) -> dict:
    print( 'running confgen', file=sys.stderr )
    smiles = input_data["smiles"]
    n_conformers = int( input_data.get("n_conformers", 10 ) )
    theory = input_data.get( "theory", "xtb" )
    force_tol = float( input_data.get( "force_tolerance", 0.1) )
    max_optimizer_steps = int( input_data.get( "max_optimizer_steps", 250) )
    rmsd_tol = float( input_data.get("rmsd_tol", 0.2) )

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
    # params = AllChem.ETKDGv3()
    # params.pruneRmsThresh = -1.0  # We do our own clustering.
    on_progress( "Generation", f"Generating {n_conformers} initial conformers", fraction=0.0 )
    # conf_ids = AllChem.EmbedMultipleConfs( mol_with_h, numConfs=n_conformers, params=params )
    # raw_conformers = [ np.asarray( mol_with_h.GetConformer(conf_id).GetPositions(), dtype=float, ) for conf_id in conf_ids ]
    
    # This output has shape (n_conformers, N, 3)
    print( 'Sampling Conformers', file=sys.stderr )
    diffusion_conformers = sample_conformers_from_mol( td_network, mol_with_h, n_conformers )
    raw_conformers = [ conf for conf in diffusion_conformers ] # unpacks the first dimension
    
    # Clustering before (expensive) XTB + L-BFGS minimization
    pre_conformers, _, cluster_sizes = rmsd_clustering( Z, raw_conformers, rmsd_tol )
    on_progress( "Generation", f"Generated {len(pre_conformers)} possibly distinct conformers ", fraction=0.1 )
    print( f"Generated {len(pre_conformers)} possibly distinct conformers ", file=sys.stderr )

    # Stabilize all generated conformers.
    optimal_conformers = []
    energies = []
    force_norms = []
    current_fraction = on_progress.getTotalProgress()
    remaining_fraction = (0.9 - current_fraction)
    for conf_id in range( len(pre_conformers) ):
        print( f'\nStabilizing Conformer {conf_id}.', file=sys.stderr )
        on_progress( "Stabilization", f"Stabilizing Conformation {conf_id+1}/{len(pre_conformers)}", 
                    fraction=current_fraction + (conf_id+1)/len(pre_conformers)*remaining_fraction )
        conf_opt, history = minimize_with_lbfgs( xtb, pre_conformers[conf_id], force_tol, max_optimizer_steps, verbose=True )
        E_opt = history[-1]["energy_kJ_mol"]
        F_opt = history[-1]["max_force_rms"]
        print( f'Conformer {conf_id} stabilized to E = {E_opt} and |F| = {F_opt}.', file=sys.stderr )

        optimal_conformers.append( conf_opt )
        energies.append( E_opt )
        force_norms.append( F_opt )

    on_progress( "Clustering", f"Clustering Stable Conformers", fraction=0.9 )
    optimal_conformers, energies, force_norms, _, cluster_sizes = post_relaxation_rmsd_clustering( Z, optimal_conformers, energies, force_norms, cluster_sizes )
    print( f'Found {len(optimal_conformers)} non-trivial conformers.', file=sys.stderr )

    output_data = { "Z" : Z,
                    "bonds" : rdkit_mol_to_bond_list(mol_with_h),
                    "conformers" : [{"x" : optimal_conformers[ii], "energy" : energies[ii], "force_norm" : force_norms[ii], "cluster_size": cluster_sizes[ii]} for ii in range(len(optimal_conformers))]}
    return output_data