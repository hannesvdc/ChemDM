import random
import numpy as np
import torch as pt
import matplotlib.pyplot as plt

from pathlib import Path
import json

from chemdm.TransitionPathDataset import TransitionPathDataset
from chemdm.MoleculeGraph import BatchedMoleculeGraph, MoleculeGraph, batchMolecules
from chemdm.MolecularEmbeddingNetwork import MolecularEmbeddingGNN
from chemdm.NewtonE3NN import NewtonE3NN

from typing import List

def loadNewtonModel( store_root : str, device : pt.device, dtype : pt.dtype) -> NewtonE3NN:
    newton_weights = pt.load( Path(store_root) / 'best_gnn.pth', map_location=device, weights_only=True )

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
    tp_network.load_state_dict( newton_weights )
    tp_network.to( dtype=dtype )

    return tp_network

@pt.no_grad
def evaluateML( tp_network : NewtonE3NN,
                s : pt.Tensor,
                Z : pt.Tensor, 
               xA : pt.Tensor, 
               xB : pt.Tensor,
               Ga : pt.Tensor,
               Gb : pt.Tensor) -> tuple[pt.Tensor, List[pt.Tensor]]:
    mol_size = len(Z)
    n_images = len(s)
    path_shape = (n_images, mol_size, 3)

    # Evaluate
    xa_graph = MoleculeGraph(Z, xA, Ga)
    xb_graph = MoleculeGraph(Z, xB, Gb)
    xa_batched = [xa_graph] * n_images
    xb_batched = [xb_graph] * n_images
    xa_mol = batchMolecules( xa_batched )
    xb_mol = batchMolecules( xb_batched )
    s_values = [ s_i.expand(mol_size) for s_i in s ]
    s_values = pt.cat( s_values )

    molecule_path, intermediate_states = tp_network( xa_mol, xb_mol, s_values )
    x = molecule_path.x.detach() # n_images * mol_size * 3
    x = pt.reshape( x, path_shape )
    for ii in range(len(intermediate_states)):
        intermediate_states[ii] = pt.reshape( intermediate_states[ii].x, path_shape )
    return x, intermediate_states

def evaluateMoleculeErrors( layer_states : List[pt.Tensor], x_ref : pt.Tensor ) -> np.ndarray:
    assert x_ref.shape == layer_states[0].shape
    stacked_layers = pt.stack( layer_states, dim=3 ) # (n_images, mol_size, 3, n_layers)
    se = pt.sum( (stacked_layers - x_ref[:,:,:,None])**2, dim=2 ) # (n_images, mol_size, n_layers)
    mse = pt.mean( se, dim=(0,1) ) # (n_layers,)
    return mse.cpu().detach().numpy()

def main():
    with open( './data_config.json', "r" ) as f:
        data_config = json.load( f )
    data_directory = data_config["data_folder"]
    store_root = data_config["store_root"]
    
    # Load the nework
    device = pt.device( 'cpu' )
    dtype = pt.float32
    network = loadNewtonModel( store_root, device, dtype )
    n_layers = network.n_refinement_steps+1

    # Load the [train, val, test] dataset and measure the molecular error per iteration
    for kind in ['train', 'val', 'test']:
        print( 'Evaluating', kind, 'Dataset' )
        dataset = TransitionPathDataset( kind, data_directory )
        n_molecules = len(dataset)

        molecule_errors = np.zeros( (n_molecules, n_layers) )
        for n in range( n_molecules ):
            if n % 100 == 0:
                print( 'Reaction', n )
            trajectory = dataset[n][-1]
            Z = trajectory.Z.to( dtype=pt.int )
            xA = trajectory.xA.to( dtype=pt.float32 )
            Ga = trajectory.GA.to( dtype=pt.int )
            xB = trajectory.xB.to( dtype=pt.float32 )
            Gb = trajectory.GB.to( dtype=pt.int )
            s = trajectory.s.to( dtype=pt.float32 )
            x_ref = trajectory.x.to( dtype=pt.float32 )

            _, layer_states = evaluateML( network, s, Z, xA, xB, Ga, Gb )
            errors = evaluateMoleculeErrors( layer_states, x_ref )
            molecule_errors[n,:] = errors

        np.save( './experiments/' + kind + '_errors.npy', molecule_errors )

def plot():
    # Load train, val and test convergence
    for kind in ['train', 'val', 'test']:
        molecule_errors = np.load( './experiments/' + kind + '_errors.npy' )
        n_layers = molecule_errors.shape[1]

        # subsample the training errors.
        if kind == 'train':
            n_paths = 100
            indices = random.sample(range(molecule_errors.shape[0]), n_paths)
            molecule_errors = molecule_errors[indices,:]
        rel_errors = molecule_errors / molecule_errors[:,0:1]

        # Plot the per-layer errors of all molecues in log-scale. Is there a decay?
        layers = np.arange( n_layers )
        plt.figure()
        plt.semilogy( layers, rel_errors.T )
        plt.title( f'{kind} Relative Errors' )
        plt.xlabel( 'Layer' )
        plt.ylabel( 'Molecule Error' )
    plt.show()

def parseArguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument( '--plot', action='store_true', dest='plot' )
    return parser.parse_args()

if __name__ == '__main__':
    args = parseArguments()
    if hasattr( args, 'plot') and bool(args.plot):
        plot()
    else:
        main()