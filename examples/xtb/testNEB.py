import numpy as np
import torch as pt
import openmm as mm

from chemdm.xtbSetup import XTBPotential
from chemdm.nebXtbDirect import run_neb_xtb, evaluate_path, neb_force
from chemdm.MoleculeGraph import MoleculeGraph, batchMolecules, Molecule

from loadModels import loadNewtonModel, loadDiffusionModel

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
               Gb : np.ndarray,
               device : pt.device) -> tuple[np.ndarray, Molecule, Molecule, pt.Tensor, Molecule]:
    mol_size = len(Z)

    # Evaluate
    n_images = 10
    s = pt.linspace(0.0, 1.0, n_images, device=device)
    xa_batched = []
    xb_batched = []
    s_values = []
    for n in range(len(s)):
        xa_batched.append( MoleculeGraph( pt.tensor(Z, dtype=pt.int, device=device), pt.tensor(xA, device=device), pt.tensor(Ga, device=device) ) )
        xb_batched.append( MoleculeGraph( pt.tensor(Z, dtype=pt.int, device=device), pt.tensor(xB, device=device), pt.tensor(Gb, device=device) ) )
        s_values.append( s[n] * pt.ones(mol_size, device=device) )
    xa_mol = batchMolecules( xa_batched )
    xb_mol = batchMolecules( xb_batched )
    s = pt.cat( s_values )

    molecule_path, _ = tp_network( xa_mol, xb_mol, s )
    x = molecule_path.x.detach().cpu().numpy() # n_images * mol_size * 3
    x = x.reshape(n_images, mol_size, 3)
    return x, xa_mol, xb_mol, s, molecule_path

def evaluateMaxForce( xtb : XTBPotential,#context : mm.Context,
                      path : np.ndarray,
                      k : float, ) -> float:
    E_np, F_np = evaluate_path( xtb, path )
    F_neb = neb_force( path, E_np, F_np, k )

    F_rms_i = np.sqrt( np.mean(F_neb**2, axis=(-2,-1)) )
    maxF = float( np.max(F_rms_i) )

    return maxF

def runNEB( tp_network : pt.nn.Module,
            diffusion_network : pt.nn.Module,
            xtb : XTBPotential, #context: mm.Context,
            trajectory : dict,
            device : pt.device ):
    KJ_MOL_PER_EV = 96.48533212331002
    k = 1.0 * KJ_MOL_PER_EV          # kJ/mol/Å², equivalent to 1 eV/Å²
    force_tol = 0.03 * KJ_MOL_PER_EV # k

    # Initial Guess : the Newton model
    path0_A, xa_mol, xb_mol, s, newton_path = evaluateML( tp_network, trajectory["Z"], trajectory["xA"], trajectory["xB"], trajectory["GA"], trajectory["GB"], device )
    maxF = evaluateMaxForce( xtb, path0_A, k )
    print( f'Max Newton NEB Force {maxF} [kJ/(mol A)]')

    # Generate a few samples using the diffusion model
    n_samples = 10
    residual_scale = 0.15
    T = 100
    best_initial = path0_A
    best_F = maxF

    # print( 'Generating Diffusion Samples')
    # samples = sample_path( diffusion_network, xa_mol, xb_mol, s, newton_path, residual_scale, T, n_samples )
    # print( 'Evaluating Forces ')
    # for count in range(n_samples):
    #     x_sample = samples[count, :,:,:].cpu().numpy()
    #     maxF = evaluateMaxForce( context, x_sample, k )
    #     if maxF < best_F:
    #         best_F = maxF
    #         best_initial = x_sample
    #     print( f'Max. NEB Force for sample {count}: {maxF} [eV / A]')

    # Finally: run NEB
    n_steps = 1000
    lr = 1e-3
    max_step_A = 0.02
    path_opt_A, E_opt_kJ, best_force = run_neb_xtb( trajectory["Z"], best_initial, n_steps, lr, k, max_step_A, force_tol, max_workers=8 )

    maxF = evaluateMaxForce( xtb, path_opt_A, k )
    print( f'Max NEB Force after optimization: {maxF} [kJ/(mol A)]' )
    print( E_opt_kJ )

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
    device = pt.device('mps')
    dtype = pt.float32
    tp_network = loadNewtonModel( './MLModel/', device, dtype )
    diffusion_network = loadDiffusionModel( './MLModel/', device, dtype )
    
    print(trajectory.keys())
    # context = create_xtb_context( trajectory["Z"] )
    xtb = XTBPotential( trajectory["Z"] )
    runNEB( tp_network, diffusion_network, xtb, trajectory, device )