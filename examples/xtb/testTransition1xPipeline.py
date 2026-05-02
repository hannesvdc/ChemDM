import numpy as np

import openmm as mm
import openmm.unit as unit

from chemdm.xtbSetup import create_xtb_context
from safeOptimizer import minimize_with_adam

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
    return {
        molecule: sorted(reaction_ids) for molecule, reaction_ids in sorted(molecule_to_reactions.items())
    }

def runRelaxation( context: mm.Context,
                   trajectory : dict,
                   verbose : bool = False ) -> tuple[np.ndarray, dict]:

    # Create the xTB system
    positions_A = np.array( trajectory["xA"], dtype=float )
    context.setPositions( 0.1 * positions_A * unit.nanometer ) # type: ignore

    # Run our Adam equilibration code
    print_every = 10 if verbose else 100
    minimized_positions_A, info = minimize_with_adam(
        context=context,
        positions_A=positions_A,
        n_steps=10_000,
        lr=1e-3,
        force_tolerance_ev_A=0.02,
        max_step_A=0.02, 
        print_every=print_every)
    
    if verbose:
        print( "Initial Positions: " )
        print( positions_A )
        print( "Optimized Positions: " )
        print( minimized_positions_A )

    return minimized_positions_A, info

if __name__ == '__main__':
    data_dir  = Path( "/Users/hannesvdc/Open Numerics/ReactionStudio/data" )
    molecule_map = build_molecule_reaction_map( data_dir, "train" )
    print(f"Found {len(molecule_map)} molecules: ", molecule_map.keys() )

    molecule_name = input( "Enter a molecule: " )
    reaction_ids = molecule_map[molecule_name]

    print( f"Reactions for molecule {molecule_name}: ", reaction_ids)
    reaction_id = int( input( "Select reaction number: " ) )

    filename = REACTION_FILE_TEMPLATE.format(split="train", reaction_id=reaction_id, molecule=molecule_name )
    with open( data_dir / filename, "r" ) as jsonfile:
        trajectory = json.load( jsonfile )
        print( "Reaction Loaded." )
    
    context = create_xtb_context( trajectory["Z"] )
    runRelaxation( context, trajectory, verbose=True )