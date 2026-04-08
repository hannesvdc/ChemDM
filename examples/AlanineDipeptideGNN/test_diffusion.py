import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import json
import math
import numpy as np
import torch as pt
import matplotlib.pyplot as plt

from pathlib import Path

from chemdm.MoleculeGraph import MoleculeGraph, batchMolecules, BatchedMoleculeGraph, Molecule
from chemdm.MolecularEmbeddingNetwork import MolecularEmbeddingGNN
from chemdm.TransitionPathDiffusionNetwork import TransitionPathDiffusionGNN
from chemdm.DDPMSchedule import DDPMSchedule
from chemdm.Trajectory import Trajectory
from TrajectoryDataset import TrajectoryDataset
from system import compute_phi_psi_from_xyz

from typing import List


@pt.no_grad()
def sample_transition_path(
    network : TransitionPathDiffusionGNN,
    schedule : DDPMSchedule,
    xA_graph : Molecule,
    xB_graph : Molecule,
    s_grid : pt.Tensor,
    device : pt.device,
    dtype : pt.dtype,
    n_steps : int = 2,
    eta : float = 0.00,
) -> pt.Tensor:
    """
    Sample a full transition path using DDIM reverse diffusion.

    All s values are batched together for efficiency: at each reverse step,
    the network processes all s values in one forward pass.

    Arguments
    ---------
    network : the trained diffusion GNN.
    schedule : the DDPM schedule.
    xA_graph, xB_graph : molecules (one copy per s value).
    s_grid : (n_s,) grid of arclength values.
    device, dtype : torch device and dtype.
    n_steps : number of DDIM reverse steps (subsampled from T).
    eta : DDIM stochasticity. 0 = deterministic, 1 = DDPM-like.

    Returns
    -------
    x_path : (n_s, n_atoms, 3) predicted clean positions along the path.
    """
    T = schedule.T
    N_total = len( xA_graph.Z )
    n_atoms_per_mol = N_total // len( s_grid )

    # Expand s to per-atom: each molecule in the batch gets one s value
    s = pt.repeat_interleave( s_grid, n_atoms_per_mol ).to( device=device, dtype=dtype )

    # Build a subsequence of timesteps evenly spaced over {0, ..., T-1}
    timesteps = pt.linspace( T - 1, 0, n_steps + 1, dtype=pt.long )
    timesteps = pt.unique( timesteps, sorted=True )
    timesteps = pt.flip( timesteps, [0] )  # descending: T-1, ..., 0

    # Start from pure noise
    x_t = pt.randn( N_total, 3, device=device, dtype=dtype )

    # DDIM reverse loop
    for i in range( len(timesteps) - 1 ):
        t_val      = timesteps[i].item()
        t_prev_val = timesteps[i + 1].item()

        t_int      = pt.full( (N_total,), t_val,      dtype=pt.long, device=device )
        t_prev_int = pt.full( (N_total,), t_prev_val,  dtype=pt.long, device=device )
        t_normalized = t_int.float() / (T - 1)

        x_0_pred = network( x_t, xA_graph, xB_graph, s, t_normalized )
        x_t = schedule.ddim_sample_step( x_0_pred.x, x_t, t_int, t_prev_int, eta=eta )

    # Final step: predict x_0 directly at the last timestep
    t_int = pt.full( (N_total,), timesteps[-1].item(), dtype=pt.long, device=device )
    t_normalized = t_int.float() / (T - 1)
    x_0_pred = network( x_t, xA_graph, xB_graph, s, t_normalized )

    # Reshape to (n_s, n_atoms, 3)
    x_path = x_0_pred.x.reshape( len(s_grid), n_atoms_per_mol, 3 )
    return x_path


def build_batched_endpoints(
    trajectory : Trajectory,
    n_copies : int,
    device : pt.device,
    dtype : pt.dtype,
) -> tuple[BatchedMoleculeGraph, BatchedMoleculeGraph]:
    """
    Create batched molecule graphs with n_copies of the same (xA, xB) pair.
    """
    xA_list = []
    xB_list = []
    for _ in range( n_copies ):
        xA_list.append( MoleculeGraph( trajectory.Z, trajectory.xA, trajectory.GA ) )
        xB_list.append( MoleculeGraph( trajectory.Z, trajectory.xB, trajectory.GB ) )

    xA = batchMolecules( xA_list ).to( device=device, dtype=dtype )
    xB = batchMolecules( xB_list ).to( device=device, dtype=dtype )
    return xA, xB


def unwrap_angle_path( theta : pt.Tensor ) -> pt.Tensor:
    """Unwrap an angle path to remove jumps at the ±pi boundary."""
    out = pt.clone( theta )
    for i in range( 1, len(out) ):
        d = out[i] - out[i - 1]
        if d > math.pi:
            out[i:] -= 2 * math.pi
        elif d < -math.pi:
            out[i:] += 2 * math.pi
    return out


def plot_path_in_phi_psi(
    sampled_paths : List[pt.Tensor],
    ground_truth : pt.Tensor,
    title : str = "",
):
    """
    Plot sampled transition paths and ground truth in (phi, psi) space.

    sampled_paths : list of (n_s, n_atoms, 3) tensors, one per sample.
    ground_truth : (n_images, n_atoms, 3) the reference NEB path.
    """
    fig, ax = plt.subplots( figsize=(7, 6) )

    # Ground truth
    gt_phi, gt_psi = compute_phi_psi_from_xyz( ground_truth )
    gt_phi = unwrap_angle_path( gt_phi )
    gt_psi = unwrap_angle_path( gt_psi )
    ax.plot( np.degrees(gt_phi.numpy()), np.degrees(gt_psi.numpy()),
             "k-o", ms=4, lw=2, label="Ground truth (NEB)", zorder=10 )

    # Sampled paths
    for i, path in enumerate( sampled_paths ):
        phi, psi = compute_phi_psi_from_xyz( path )
        phi = unwrap_angle_path( phi )
        psi = unwrap_angle_path( psi )
        label = "Sampled paths" if i == 0 else None
        ax.plot( np.degrees(phi.numpy()), np.degrees(psi.numpy()),
                 "-o", ms=3, alpha=0.5, label=label )

    ax.set_xlabel( r"$\phi$ [deg]" )
    ax.set_ylabel( r"$\psi$ [deg]" )
    ax.legend()
    ax.set_title( title )
    fig.tight_layout()


def main():
    # config
    with open( "./data_config.json", "r" ) as config_file:
        data_config = json.load( config_file )
    data_directory = data_config["data_folder"]
    exp_name  = data_config.get( "name", "" )
    device = pt.device( "cpu" )
    dtype  = pt.float32

    # load model
    T_diffusion = 1000
    schedule = DDPMSchedule( T=T_diffusion, schedule="cosine" )

    embedding_state_size = 64
    embedding_message_size = 64
    n_embedding_layers = 5
    n_tp_layers = 5
    tp_message_size = 64
    d_cutoff = 1.0 # nm
    xA_embedding = MolecularEmbeddingGNN( embedding_state_size, embedding_message_size, n_embedding_layers, d_cutoff )
    xB_embedding = MolecularEmbeddingGNN( embedding_state_size, embedding_message_size, n_embedding_layers, d_cutoff )
    network = TransitionPathDiffusionGNN( xA_embedding, xB_embedding, tp_message_size, n_tp_layers, d_cutoff )

    model_path = f"./models/{exp_name}best_diffusion_gnn.pth"
    network.load_state_dict( pt.load( model_path, map_location=device, weights_only=True ) )
    network.to( device=device, dtype=dtype )
    network.eval()
    print( f"Loaded model from {model_path}" )

    # load data
    dataset = TrajectoryDataset( outdir=Path(data_directory) )
    print( f"Dataset size: {len(dataset)}" )

    # sample paths
    n_s = 21
    s_grid = pt.linspace( 0.0, 1.0, n_s )
    n_samples = 5  # number of independent samples per trajectory
    n_test_trajectories = 5 # number of trajectories to test

    # Pick evenly spaced trajectories from the dataset
    test_indices = np.linspace( 0, len(dataset) - 1, n_test_trajectories, dtype=int )

    for idx in test_indices:
        trajectory = dataset[idx]
        xA_batch, xB_batch = build_batched_endpoints( trajectory, n_s, device, dtype )

        # Sample multiple paths
        sampled_paths = []
        for sample_idx in range( n_samples ):
            print( f"  Trajectory {idx}, sample {sample_idx + 1}/{n_samples}...", flush=True )
            x_path = sample_transition_path(
                network, schedule, xA_batch, xB_batch, s_grid, device, dtype
            )
            sampled_paths.append( x_path.cpu() )

        # Ground truth path
        ground_truth = trajectory.x  # (n_images, n_atoms, 3)

        plot_path_in_phi_psi(
            sampled_paths, ground_truth,
            title=f"Trajectory {idx}"
        )

    plt.show()


if __name__ == "__main__":
    main()