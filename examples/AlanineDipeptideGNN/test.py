import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import math
import numpy as np
import torch as pt
import matplotlib.pyplot as plt

from pathlib import Path

from chemdm.MoleculeGraph import MoleculeGraph, batchMolecules, BatchedMoleculeGraph, Molecule
from chemdm.MolecularEmbeddingNetwork import MolecularEmbeddingGNN
from chemdm.TransitionPathNetwork import TransitionPathGNN
from chemdm.Trajectory import Trajectory
from TrajectoryDataset import TrajectoryDataset
from system import compute_phi_psi_from_xyz

@pt.no_grad()
def predict_transition_path(
    network : TransitionPathGNN,
    xA_graph : Molecule,
    xB_graph : Molecule,
    s_grid : pt.Tensor,
    device : pt.device,
    dtype : pt.dtype,
) -> pt.Tensor:
    """
    Predict a full transition path using the regression GNN.

    Arguments
    ---------
    network : the trained regression GNN.
    xA_graph, xB_graph : batched molecules (one copy per s value).
    s_grid : (n_s,) grid of arclength values.
    device, dtype : torch device and dtype.

    Returns
    -------
    x_path : (n_s, n_atoms, 3) predicted positions along the path.
    """
    N_total = len( xA_graph.Z )
    n_atoms_per_mol = N_total // len( s_grid )

    s = pt.repeat_interleave( s_grid, n_atoms_per_mol ).to( device=device, dtype=dtype )

    x_pred = network( xA_graph, xB_graph, s )

    x_path = x_pred.x.reshape( len(s_grid), n_atoms_per_mol, 3 )
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
    predicted_path : pt.Tensor,
    ground_truth : pt.Tensor,
    title : str = "",
):
    """
    Plot predicted transition path and ground truth in (phi, psi) space.

    predicted_path : (n_s, n_atoms, 3) tensor.
    ground_truth : (n_images, n_atoms, 3) the reference NEB path.
    """
    fig, ax = plt.subplots( figsize=(7, 6) )

    # Ground truth
    gt_phi, gt_psi = compute_phi_psi_from_xyz( ground_truth )
    gt_phi = unwrap_angle_path( gt_phi )
    gt_psi = unwrap_angle_path( gt_psi )
    ax.plot( np.degrees(gt_phi.numpy()), np.degrees(gt_psi.numpy()),
             "k-o", ms=4, lw=2, label="Ground truth (NEB)", zorder=10 )

    # Predicted path
    phi, psi = compute_phi_psi_from_xyz( predicted_path )
    phi = unwrap_angle_path( phi )
    psi = unwrap_angle_path( psi )
    ax.plot( np.degrees(phi.numpy()), np.degrees(psi.numpy()),
             "b-o", ms=3, alpha=0.7, label="Predicted path" )

    ax.set_xlabel( r"$\phi$ [deg]" )
    ax.set_ylabel( r"$\psi$ [deg]" )
    ax.legend()
    ax.set_title( title )
    fig.tight_layout()


def main():
    device = pt.device( "cpu" )
    dtype  = pt.float32

    # Load model (matching train.py architecture)
    embedding_state_size = 64
    embedding_message_size = 64
    n_embedding_layers = 5
    n_tp_layers = 5
    tp_message_size = 64
    d_cutoff = 1.0  # nm

    xA_embedding = MolecularEmbeddingGNN( embedding_state_size, embedding_message_size, n_embedding_layers, d_cutoff )
    xB_embedding = MolecularEmbeddingGNN( embedding_state_size, embedding_message_size, n_embedding_layers, d_cutoff )
    network = TransitionPathGNN( xA_embedding, xB_embedding, tp_message_size, n_tp_layers, d_cutoff )

    model_path = "./models/gnn.pth"
    network.load_state_dict( pt.load( model_path, map_location=device, weights_only=True ) )
    network.to( device=device, dtype=dtype )
    network.eval()
    print( f"Loaded model from {model_path}" )

    # Load data
    dataset = TrajectoryDataset( outdir=Path("outputs") )
    print( f"Dataset size: {len(dataset)}" )

    # Predict paths
    n_s = 21
    s_grid = pt.linspace( 0.0, 1.0, n_s )
    n_test_trajectories = 5

    test_indices = [101, 201, 401, 801, 1601]

    for idx in test_indices:
        trajectory = dataset[idx]
        xA_batch, xB_batch = build_batched_endpoints( trajectory, n_s, device, dtype )

        print( f"  Predicting trajectory {idx}...", flush=True )
        x_path = predict_transition_path(
            network, xA_batch, xB_batch, s_grid, device, dtype
        )

        ground_truth = trajectory.x
        plot_path_in_phi_psi( x_path.cpu(), ground_truth, title=f"Trajectory {idx}" )

    plt.show()


if __name__ == "__main__":
    main()
