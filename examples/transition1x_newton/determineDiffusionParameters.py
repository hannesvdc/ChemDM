"""
This script is to determine good parameters $(T, \\beta_{\\max})$ for the denoising / diffusion
transition path generator. They must be chosen such that the forward diffusion, applied to the
diffusion input, becomes standard normally distributed, i.e., 'pure noise'. 

Since the diffusion model is 'applied' to the residual between the actual TP and the 'Newton' guess,
this most be the input to the forward process. This script therefore loads the Newton model.
"""

import math
import torch as pt
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

import json

from chemdm.TransitionPathDataset import TransitionPathDataset
from chemdm.MoleculeGraph import BatchedMoleculeGraph, MoleculeGraph, Molecule, batchMolecules
from chemdm.Trajectory import Trajectory, enforceCOM

from typing import Tuple, List

from loadNewtonModel import loadNewtonModel

def collate_trajectory( trajectory : Trajectory ) -> tuple[List[Molecule], List[Molecule], List[float], pt.Tensor]:
    n_images = len( trajectory.s )
    s = trajectory.s
    Z = trajectory.Z
    xA = trajectory.xA
    xB = trajectory.xB
    GA = trajectory.GA
    GB = trajectory.GB

    mol_size = len(Z)
    xa_batched, xb_batched, s_values = [], [], []
    for n in range(n_images):
        xa_batched.append( MoleculeGraph(Z, xA, GA) )
        xb_batched.append( MoleculeGraph(Z, xB, GB) )
        s_values.append( s[n] * pt.ones(mol_size, dtype=pt.float32) )
    x_refs = pt.reshape( trajectory.x, ( n_images*mol_size, 3) ) # (n_images, mol_size, 3)

    return xa_batched, xb_batched, s_values, x_refs

def collate_molecules(trajectories : List
                     ) -> Tuple[BatchedMoleculeGraph, BatchedMoleculeGraph, pt.Tensor, pt.Tensor]:
    """
    Batching molecules and random points on the path.
    """
    assert len(trajectories) == 1
    trajectories = trajectories[0]

    # Sample random points on the trajectory for each molecule
    xA_molecules = []
    xB_molecules = []
    s_list = []
    x_list = []
    for trajectory in trajectories:
        trajectory = enforceCOM( trajectory )
        xa_mols, xb_mols, s_values, x_refs = collate_trajectory( trajectory )
        xA_molecules.extend( xa_mols )
        xB_molecules.extend( xb_mols )
        s_list.extend( s_values )
        x_list.append( x_refs )

    s = pt.cat( s_list ) # (N_atoms,)
    x_ref = pt.cat( x_list, dim=0 ) # (N_atoms,3)

    xA = batchMolecules( xA_molecules )
    xB = batchMolecules( xB_molecules )
    return xA, xB, s, x_ref

def main( ):
    pt.set_grad_enabled(False)

    with open( './data_config.json', "r" ) as f:
        data_config = json.load( f )
    data_directory = data_config["data_folder"]
    root = data_config.get( "store_root" )

    # Load the full transition1x dataset
    dataset = TransitionPathDataset( "train", data_directory )
    loader = DataLoader( dataset,
                         batch_size=1,
                         shuffle=False,
                         num_workers=8,
                         collate_fn=collate_molecules, 
                         prefetch_factor=2)
    
    # Create the network
    device = pt.device( 'cpu' )
    dtype = pt.float32
    tp_network = loadNewtonModel( root, device, dtype )

    # Diffusion model - Cosine schedule (Nichol & Dhariwal, 2021)
    T = 100
    cosine_s = 0.008
    steps = pt.linspace(0, T, T + 1, dtype=pt.float64)
    alpha_bar_full = pt.cos( (steps / T + cosine_s) / (1.0 + cosine_s) * (math.pi / 2.0) ) ** 2
    alpha_bar_full = alpha_bar_full / alpha_bar_full[0]

    # T entries for diffusion indices t = 0, ..., T-1.
    # alpha_bar[0] is nearly clean, alpha_bar[-1] is nearly pure noise.
    alpha_bar = alpha_bar_full[1:].float()
    sqrt_alpha_bar = pt.sqrt(alpha_bar)
    sqrt_one_minus_alpha_bar = pt.sqrt(1.0 - alpha_bar)

    print( sqrt_alpha_bar[-1] )
#    noises = []
    sum_sq = 0.0
    count = 0
    for reaction_idx, (xA, xB, s, x_ref) in enumerate( loader ):
        print( f'Reaction {reaction_idx}' )
        xA.to( dtype=dtype )
        xB.to( dtype=dtype )
        s = s.to( dtype=dtype )

        # Evaluate the forward network
        x_newton, _ = tp_network( xA, xB, s )
        x_newton = x_newton.x
        c0 = x_ref - x_newton

        # Compute a streaming residual scale
        sum_sq += (c0 ** 2).sum()
        count += c0.numel()

        # Step forward using 
        eps = pt.randn_like(c0)
        t_idx = T - 1 # unnormalized
        cT = sqrt_alpha_bar[t_idx] * c0 + sqrt_one_minus_alpha_bar[t_idx] * eps

        # Reshape to (n_molecules, mol_size, 3)
        mol_size = int( pt.sum( xA._molecule_id == xA._molecule_id[0] ) )
        n_molecules = len( xA.Z ) // mol_size
        cT = pt.reshape( cT, (n_molecules, mol_size, 3) )

        # Append to the storage dictionary
        # if mol_size in noise_per_size:
        #    noise_per_size[mol_size] = pt.cat( (noise_per_size[mol_size], cT), dim=0 )
        #else:
        #    noise_per_size[mol_size] = cT

        # Plot per reaction
#        noises.append( pt.flatten( cT ) )

        # if len(noises) == 10:
        #     cT_plot = pt.cat( noises )
        #     noises = []
        #     n_bins = int( math.sqrt( len( cT_plot ) ) )
        #     plt.figure()
        #     plt.hist( cT_plot.numpy(), bins=n_bins, density=True )
        #     plt.title( f"Reaction {reaction_idx}" )
        #     plt.xlabel( f"Noise" )
        #     plt.show()

    residual_scale = math.sqrt(sum_sq / count)
    print( f"Dataset Residual Scale: {residual_scale}" )


if __name__ == '__main__':
    main()