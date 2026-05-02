import numpy as np
import torch as pt
import openmm as mm

from chemdm.xtbSetup import create_xtb_context
from chemdm.nebXtb import run_neb_xtb
from chemdm.NewtonE3NN import NewtonE3NN
from chemdm.MolecularEmbeddingNetwork import MolecularEmbeddingGNN
from chemdm.MoleculeGraph import MoleculeGraph, batchMolecules

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

def evaluateML( Z : np.ndarray, 
               xA : np.ndarray, 
               xB : np.ndarray,
               Ga : np.ndarray,
               Gb : np.ndarray) -> np.ndarray:
    mol_size = len(Z)

    # Global molecular information
    d_cutoff = 5.0
    n_rbf = 10

    # Endpoint embedding networks
    embedding_state_size = 64
    embedding_message_size = 64
    n_embedding_layers = 5
    xA_embedding = MolecularEmbeddingGNN( embedding_state_size, embedding_message_size, n_embedding_layers, d_cutoff )
    xB_embedding = MolecularEmbeddingGNN( embedding_state_size, embedding_message_size, n_embedding_layers, d_cutoff )

    # E3NN transition-path network
    irreps_node_str = "48x0e + 16x1o + 16x1e + 8x2e"
    n_refinement_steps = 7
    tp_network = NewtonE3NN(
        xA_embedding_network=xA_embedding,
        xB_embedding_network=xB_embedding,
        irreps_node_str=irreps_node_str,
        n_refinement_steps=n_refinement_steps,
        d_cutoff=d_cutoff,
        n_freq=8,
        n_rbf=n_rbf,
    )

    # Load the parameters from file
    state_dict = pt.load( './MLModel/best_gnn.pth', map_location=pt.device("cpu"), weights_only=True )
    tp_network.load_state_dict( state_dict )

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
    return x

def runNEB( context: mm.Context,
            trajectory : dict ):
    path0_A = evaluateML( trajectory["Z"], trajectory["xA"], trajectory["xB"], trajectory["GA"], trajectory["GB"])

    # path0_A = np.asarray( trajectory["x"] )
    print(path0_A.shape)

    n_steps = 1000
    lr = 1e-3
    k = 1.0 # eV / A^2
    max_step_A = 0.02
    force_tol = 0.03  # eV / A
    path_opt_A, E_opt_eV, best_force = run_neb_xtb( context, path0_A, n_steps, lr, k, max_step_A, force_tol )

    print( best_force )

if __name__ == '__main__':
    data_dir  = Path( "/Users/hannesvdc/Open Numerics/ReactionStudio/data" )
    split = "val"
    molecule_map = build_molecule_reaction_map( data_dir, split )
    print(f"Found {len(molecule_map)} molecules: ", molecule_map.keys() )

    molecule_name = input( "Enter a molecule: " )
    reaction_ids = molecule_map[molecule_name]

    print( f"Reactions for molecule {molecule_name}: ", reaction_ids)
    reaction_id = int( input( "Select reaction number: " ) )

    filename = REACTION_FILE_TEMPLATE.format(split=split, reaction_id=reaction_id, molecule=molecule_name )
    with open( data_dir / filename, "r" ) as jsonfile:
        trajectory = json.load( jsonfile )
        print( "Reaction Loaded." )
    
    print(trajectory.keys())
    context = create_xtb_context( trajectory["Z"] )
    runNEB( context, trajectory )