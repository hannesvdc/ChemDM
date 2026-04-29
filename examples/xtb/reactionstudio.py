import numpy as np

import openmm as mm

from xtbSetup import create_xtb_context
from neb import run_neb_xtb, normalized_arclengths

import copy
import json
import argparse

def runNEB( context: mm.Context,
            trajectory : dict ):
    path0_A = np.asarray( trajectory["x"] )
    print(path0_A.shape)

    n_steps = 1000
    lr = 1e-3
    k = 1.0 # eV / A^2
    max_step_A = 0.02
    force_tol = 0.03  # eV / A
    path_opt_A, E_opt_eV, best_force = run_neb_xtb( context, path0_A, n_steps, lr, k, max_step_A, force_tol )

    return path_opt_A, E_opt_eV, best_force

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument( '--input_file', required=True, dest='input_file', nargs='?' )
    parser.add_argument( '--output_file', required=True, dest='output_file', nargs='?' )
    return parser.parse_args()

if __name__ == '__main__':
    args = parseArguments()
    input_file = args.input_file

    # Load the trajectory end points
    with open(input_file, "r") as file:
        input_data = json.load( file )
    Z = np.asarray( input_data["Z"] )

    # Create the xTB system and context
    context = create_xtb_context( Z )

    # Run Nudged-Elastic Band.
    path_opt, E_opt_eV, best_force = runNEB( context, input_data )
    s = normalized_arclengths( path_opt )

    # Store the optimal path as a new trajectory
    output_data = copy.deepcopy( input_data )
    output_data["x"] = path_opt.tolist()
    output_data["s"] = s.tolist()

    # JSON dump
    output_file = args.output_file
    with open( output_file, "w" ) as file:
        json.dump( output_data, file )