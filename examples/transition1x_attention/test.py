import random
import numpy as np
import torch as pt
import matplotlib.pyplot as plt

from pathlib import Path
import json

from chemdm.TransitionPathDataset import TransitionPathDataset
from EquivariantTransformer import EquivariantTransformer
from chemdm.MoleculeGraph import MoleculeGraph, batchMolecules, Molecule
from chemdm.util import formula_from_Z

from typing import List

def loadAttentionModel( store_root : Path, device : pt.device, dtype : pt.dtype) -> EquivariantTransformer:
    newton_weights = pt.load( Path(store_root) / 'best_gnn.pth', map_location=device, weights_only=True )

    d_cutoff = 5.0
    n_rbf = 10

    # E3NN transition-path network
    irreps_node_str = "48x0e + 16x1o + 16x1e + 8x2e"
    irreps_qk_str = "16x0e + 8x1o + 4x1e"
    n_refinement_steps = 7
    tp_embedding_dim = 64
    tp_embedding_hidden_dim = 128
    tp_embedding_hidden_layers = 2
    tp_network = EquivariantTransformer( irreps_node_str=irreps_node_str,
                                         irreps_qk_str=irreps_qk_str,
                                         n_refinement_steps=n_refinement_steps,
                                         d_cutoff=d_cutoff,
                                         n_freq=8,
                                         n_rbf=n_rbf,
                                         tp_embedding_dim=tp_embedding_dim,
                                         tp_embedding_hidden_dim=tp_embedding_hidden_dim,
                                         tp_embedding_hidden_layers=tp_embedding_hidden_layers )
    tp_network.load_state_dict( newton_weights )
    tp_network.to( device=device, dtype=dtype )

    return tp_network

@pt.no_grad
def evaluateML( tp_network : EquivariantTransformer,
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
    xa_batched : List[Molecule] = [xa_graph] * n_images
    xb_batched : List[Molecule] = [xb_graph] * n_images
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
    experiment_name = "single_head_self_attention"
    store_root = Path( data_config["store_root"] ) / experiment_name

    # Load the nework. There is a memory overflow issue on mps, so we must use cpu
    device = pt.device( 'cpu' )
    dtype = pt.float32
    network = loadAttentionModel( store_root, device, dtype )
    n_layers = network.n_refinement_steps+1

    # Per split: evaluate, save the layer-MSE matrix, and remember the
    # dataset + final-layer RMSE for the post-eval analysis.
    splits = ['train', 'val', 'test']
    per_split: dict[str, tuple[np.ndarray, TransitionPathDataset]] = {}
    for kind in splits:
        print( 'Evaluating', kind, 'Dataset' )
        dataset = TransitionPathDataset( kind, data_directory )
        n_molecules = len(dataset)

        molecule_errors = np.zeros( (n_molecules, n_layers) )
        for n in range( n_molecules ):
            if n % 100 == 0:
                print( 'Reaction', n )
            trajectory = dataset[n][-1]
            Z = trajectory.Z.to( device=device, dtype=pt.int )
            xA = trajectory.xA.to( device=device, dtype=pt.float32 )
            Ga = trajectory.GA.to( device=device, dtype=pt.int )
            xB = trajectory.xB.to( device=device, dtype=pt.float32 )
            Gb = trajectory.GB.to( device=device, dtype=pt.int )
            s = trajectory.s.to( device=device, dtype=pt.float32 )
            x_ref = trajectory.x.to( device=device, dtype=pt.float32 )

            _, layer_states = evaluateML( network, s, Z, xA, xB, Ga, Gb )
            errors = evaluateMoleculeErrors( layer_states, x_ref )
            molecule_errors[n,:] = errors

        np.save( store_root /  str(kind + '_errors.npy'), molecule_errors )

        # MSE is mean over (image, atom) of summed-squared-xyz, so sqrt is
        # the standard per-atom RMSD in Å.
        rmse = np.sqrt( np.maximum(molecule_errors[:, -1], 0.0) )
        per_split[kind] = (rmse, dataset)

    # Summary table.
    print()
    for kind, (rmse, _) in per_split.items():
        pcts = np.percentile( rmse, [50, 90, 95, 99] )
        print(
            f'{kind:>5s}  n={len(rmse):>6d}  '
            f'mean={rmse.mean():.3f}  median={pcts[0]:.3f}  '
            f'p90={pcts[1]:.3f}  p95={pcts[2]:.3f}  p99={pcts[3]:.3f}  '
            f'max={rmse.max():.3f}  [Å]'
        )

    # Top-N worst molecules per split (identity from the dataset already in scope).
    n_worst = 5
    for kind, (rmse, dataset) in per_split.items():
        worst_idx = np.argsort( -rmse )[:n_worst]
        print(f'\nTop {n_worst} worst {kind} molecules (by final-layer RMSE):')
        for rank, idx in enumerate( worst_idx, start=1 ):
            traj = dataset[int(idx)][-1]
            formula = formula_from_Z( traj.Z )
            n_atoms = int(traj.Z.numel())
            n_images = int(traj.s.numel())
            print(
                f'  #{rank}  idx={int(idx):>5d}  RMSE={rmse[idx]:.3f} Å  '
                f'formula={formula:<14s} (N={n_atoms}, images={n_images})  '
                f'file={dataset.file_names[int(idx)]}'
            )

    # Histograms (one subplot per split).
    fig, axes = plt.subplots( 1, len(per_split), figsize=(5 * len(per_split), 4), squeeze=False )
    for ax, (kind, (rmse, _)) in zip( axes[0], per_split.items() ):
        ax.hist( rmse, bins=50, edgecolor='black' )
        ax.set_xlabel('per-molecule RMSE [Å]')
        ax.set_ylabel('# molecules')
        ax.set_title( f'{kind}  (n={len(rmse)}, median={np.median(rmse):.3f} Å)' )
        ax.grid( axis='y', alpha=0.3 )
    fig.suptitle('Final-layer per-molecule RMSE distributions')
    plt.tight_layout()
    plt.show()

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


if __name__ == '__main__':
    main()