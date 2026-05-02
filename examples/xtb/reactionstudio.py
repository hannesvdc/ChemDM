import numpy as np
import torch as pt
import openmm as mm

from chemdm.xtbSetup import create_xtb_context
from chemdm.nebXtb import run_neb_xtb, normalized_arclengths

from chemdm.NewtonE3NN import NewtonE3NN
from chemdm.MolecularEmbeddingNetwork import MolecularEmbeddingGNN
from chemdm.MoleculeGraph import MoleculeGraph, batchMolecules

import json
import copy
import json
import argparse

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
    state_dict = pt.load( '/Users/hannesvdc/Research/Side-Projects/ChemDM/examples/xtb/MLModel/best_gnn.pth', map_location=pt.device("cpu"), weights_only=True )
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
    #path0_A = np.asarray( trajectory["x"] )
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