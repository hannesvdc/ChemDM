import numpy as np
import torch as pt
import openmm as mm

from chemdm.xtbSetup import create_xtb_context
from chemdm.nebXtb import run_neb_xtb, evaluate_path, neb_force
from chemdm.MoleculeGraph import MoleculeGraph, batchMolecules, Molecule

from loadModels import loadNewtonModel, loadDiffusionModel
from sample_path import sample_path

from pathlib import Path
from collections import defaultdict
import re
import json

REACTION_FILE_PATTERN = re.compile( r"^(?P<split>.+?)_reaction_(?P<reaction_id>\d+)_molecule_(?P<molecule>.+)\.json$" )
REACTION_FILE_TEMPLATE = "{split}_reaction_{reaction_id}_molecule_{molecule}.json"

def build_molecule_reaction_map(data_dir: str | Path, kind : str) -> dict[str, list[int]]:
    """
    Parse Transition1x-style JSON filenames and build a molecule -> reaction IDs map.
    
    Expected filename format:
        test_reaction_0_molecule_C2H3N3O2.json
        train_reaction_123_molecule_C4H8O2.json

    Returns
    -------
    dict[str, list[int]]
        Example:
        {
            "C2H3N3O2": [0, 1, 2],
            "C4H8O2": [0, 7],
        }
    """

    data_dir = Path(data_dir)
    molecule_to_reactions = defaultdict(list)

    for path in data_dir.glob("*.json"):
        match = REACTION_FILE_PATTERN.match(path.name)
        if match is None or match.group("split") != kind:
            # Ignore unrelated JSON files.
            continue

        molecule = match.group("molecule")
        reaction_id = int(match.group("reaction_id"))
        molecule_to_reactions[molecule].append(reaction_id)

    # Sort reaction IDs for each molecule.
    return { molecule: sorted(reaction_ids) for molecule, reaction_ids in sorted(molecule_to_reactions.items()) }

def evaluateML( tp_network : pt.nn.Module,
                Z : np.ndarray, 
               xA : np.ndarray, 
               xB : np.ndarray,
               Ga : np.ndarray,
               Gb : np.ndarray) -> tuple[np.ndarray, Molecule, Molecule, pt.Tensor, Molecule]:
    mol_size = len(Z)

    # Evaluate
    n_images = 10
    s = pt.linspace(0.0, 1.0, n_images)
    xa_batched = []
    xb_batched = []
    s_values = []
    for n in range(len(s)):
        xa_batched.append( MoleculeGraph( pt.tensor(Z, dtype=pt.int), pt.tensor(xA), pt.tensor(Ga) ) )
        xb_batched.append( MoleculeGraph( pt.tensor(Z, dtype=pt.int), pt.tensor(xB), pt.tensor(Gb) ) )
        s_values.append( s[n] * pt.ones(mol_size) )
    xa_mol = batchMolecules( xa_batched )
    xb_mol = batchMolecules( xb_batched )
    s = pt.cat( s_values )

    molecule_path, _ = tp_network( xa_mol, xb_mol, s )
    x = molecule_path.x.detach().numpy() # n_images * mol_size * 3
    x = x.reshape(n_images, mol_size, 3)
    return x, xa_mol, xb_mol, s, molecule_path

def evaluateMaxForce( context : mm.Context,
                      path : np.ndarray,
                      k : float, ) -> float:
    E_np, F_np = evaluate_path( context, path )
    F_neb = neb_force( path, E_np, F_np, k )

    F_rms_i = np.sqrt( np.mean(F_neb**2, axis=(-2,-1)) )
    maxF = float( np.max(F_rms_i) )

    return maxF

def runNEB( tp_network : pt.nn.Module,
            diffusion_network : pt.nn.Module,
            context: mm.Context,
            trajectory : dict ):
    k = 1.0 # eV / A^2
    
    # Initial Guess : the Newton model
    path0_A, xa_mol, xb_mol, s, newton_path = evaluateML( tp_network, trajectory["Z"], trajectory["xA"], trajectory["xB"], trajectory["GA"], trajectory["GB"] )
    maxF = evaluateMaxForce( context, path0_A, k )
    print( f'Max Newton NEB Force {maxF} [eV / A]')

    # Generate a few samples using the diffusion model
    n_images = path0_A.shape[0]
    mol_size = path0_A.shape[1]
    n_samples = 10
    residual_scale = 0.15
    T = 100
    best_initial = path0_A
    best_F = maxF
    for count in range(n_samples):
        x_sample = sample_path( diffusion_network, xa_mol, xb_mol, s, newton_path, residual_scale, T)
        x_sample = np.reshape( x_sample.cpu().numpy(), (n_images, mol_size, 3))
        maxF = evaluateMaxForce( context, x_sample, k )
        if maxF < best_F:
            best_F = maxF
            best_initial = x_sample
        print( f'Max. NEB Force for sample {count}: {maxF} [eV / A]')

    # Finally: run NEB
    n_steps = 1000
    lr = 1e-3
    max_step_A = 0.02
    force_tol = 0.03  # eV / A
    path_opt_A, E_opt_eV, best_force = run_neb_xtb( context, best_initial, n_steps, lr, k, max_step_A, force_tol )

    maxF = evaluateMaxForce( context, path_opt_A, k )
    print( f'Max NEB Force after optimization: {maxF} [eV / A]' )
    print( E_opt_eV )

if __name__ == '__main__':
    data_dir  = Path( "/Users/hannesvdc/Open Numerics/ReactionStudio/data" )
    split = "test"
    molecule_map = build_molecule_reaction_map( data_dir, split )
    print( f"Found {len(molecule_map)} molecules: ", molecule_map.keys() )

    molecule_name = input( "Enter a molecule: " )
    reaction_ids = molecule_map[molecule_name]

    print( f"Reactions for molecule {molecule_name}: ", reaction_ids)
    reaction_id = int( input( "Select reaction number: " ) )

    filename = REACTION_FILE_TEMPLATE.format(split=split, reaction_id=reaction_id, molecule=molecule_name )
    with open( data_dir / filename, "r" ) as jsonfile:
        trajectory = json.load( jsonfile )
        print( "Reaction Loaded." )

    # Load the neural models
    device = pt.device('cpu')
    dtype = pt.float32
    tp_network = loadNewtonModel( './MLModel/', device, dtype )
    diffusion_network = loadDiffusionModel( './MLModel/', device, dtype )
    
    print(trajectory.keys())
    context = create_xtb_context( trajectory["Z"] )
    runNEB( tp_network, diffusion_network, context, trajectory )