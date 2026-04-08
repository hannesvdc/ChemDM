"""
Sanity check: push transition1x data through the forward diffusion process
at t = T-1 and verify the resulting distribution is close to N(0, I).

Because molecules have varying numbers of atoms, the check is done:
  1. Globally (all Cartesian components pooled).
  2. Stratified by molecule size (number of atoms).
"""

import json
import random
import numpy as np
import torch as pt
import matplotlib.pyplot as plt
from scipy.stats import norm
from collections import defaultdict

from chemdm.DDPMSchedule import DDPMSchedule
from chemdm.TransitionPathDataset import TransitionPathDataset
from chemdm.Trajectory import enforceCOM, alignToReactant


def load_positions( dataset : TransitionPathDataset,
                    max_reactions : int = 500,
                  ) -> list:
    """
    Load positions from the dataset, applying the same COM + Kabsch
    preprocessing used during training.  Returns a list of (n_atoms, x_0)
    tuples where x_0 is (n_atoms, 3).
    """
    samples = []   # list of (n_atoms, x_0_tensor)
    n_reactions = min( len(dataset), max_reactions )
    for idx in range( n_reactions ):
        tp_list = dataset[idx]
        for trajectory in tp_list:
            trajectory = enforceCOM( trajectory )
            trajectory = alignToReactant( trajectory )

            n_atoms = trajectory.Z.shape[0]
            # Sample a few random frames per trajectory to keep memory bounded
            n_images = trajectory.x.shape[0]
            frame_indices = random.sample( range(n_images), min(5, n_images) )
            for k in frame_indices:
                samples.append( (n_atoms, trajectory.x[k]) )  # (n_atoms, 3)

    return samples


def check_forward_diffusion( samples : list,
                             schedule : DDPMSchedule,
                             schedule_name : str,
                             T : int ):
    """
    Forward-diffuse all samples to t = T-1 and check Gaussianity.
    """
    # Group samples by molecule size
    by_size = defaultdict( list )
    for n_atoms, x0 in samples:
        by_size[n_atoms].append( x0 )

    # --- Global check (all sizes pooled) ---
    all_x0 = pt.cat( [x0 for _, x0 in samples], dim=0 )   # (N_total, 3)
    t = pt.full( (all_x0.shape[0],), T - 1, dtype=pt.long )
    x_T, _ = schedule.q_sample( all_x0, t )

    flat = x_T.numpy().flatten()
    print( f"\n{'='*60}" )
    print( f"{schedule_name} schedule  (T = {T})" )
    print( f"{'='*60}" )
    print( f"  alpha_bar[T-1]  = {schedule.alpha_bar[-1].item():.2e}" )
    print( f"  Total atoms     = {all_x0.shape[0]}" )
    print( f"  Global  mean    = {flat.mean():.4f}   (want ~0)" )
    print( f"  Global  std     = {flat.std():.4f}   (want ~1)" )
    print( f"  Global  min/max = {flat.min():.4f} / {flat.max():.4f}" )

    # --- Plot global ---
    fig, axes = plt.subplots( 1, 2, figsize=(12, 5) )
    fig.suptitle( f"Global — {schedule_name} schedule, T={T}" )

    ax = axes[0]
    ax.hist( flat, bins=200, density=True, alpha=0.7, label=f"x_T" )
    z = np.linspace( -4, 4, 300 )
    ax.plot( z, norm.pdf(z), "r-", lw=2, label="N(0, 1)" )
    ax.set_xlabel( "Value" )
    ax.set_ylabel( "Density" )
    ax.legend()
    ax.set_xlim( -5, 5 )

    ax = axes[1]
    sorted_flat = np.sort( flat )
    n = len( sorted_flat )
    theoretical = norm.ppf( (np.arange(1, n+1) - 0.5) / n )
    stride = max( 1, n // 5000 )
    ax.scatter( theoretical[::stride], sorted_flat[::stride], s=1, alpha=0.5 )
    ax.plot( [-4, 4], [-4, 4], "r-", lw=2 )
    ax.set_xlabel( "Theoretical quantiles" )
    ax.set_ylabel( "Sample quantiles" )
    ax.set_title( "QQ plot" )
    ax.set_xlim( -4, 4 ); ax.set_ylim( -4, 4 )
    ax.set_aspect( "equal" )
    fig.tight_layout()

    # --- Per-size statistics ---
    sizes = sorted( by_size.keys() )
    print( f"\n  Per molecule size:" )
    print( f"  {'n_atoms':>7s}  {'count':>6s}  {'mean':>8s}  {'std':>8s}  {'min':>8s}  {'max':>8s}" )

    size_labels, size_means, size_stds = [], [], []
    for n_atoms in sizes:
        x0_group = pt.stack( by_size[n_atoms] )          # (M, n_atoms, 3)
        M = x0_group.shape[0]
        x0_flat = x0_group.reshape( -1, 3 )               # (M * n_atoms, 3)
        t_group = pt.full( (x0_flat.shape[0],), T - 1, dtype=pt.long )
        xT_group, _ = schedule.q_sample( x0_flat, t_group )
        vals = xT_group.numpy().flatten()
        print( f"  {n_atoms:7d}  {M:6d}  {vals.mean():8.4f}  {vals.std():8.4f}  "
               f"{vals.min():8.4f}  {vals.max():8.4f}" )

        size_labels.append( n_atoms )
        size_means.append( vals.mean() )
        size_stds.append( vals.std() )

    # --- Bar chart of mean and std per size ---
    if len(sizes) > 1:
        fig2, (ax1, ax2) = plt.subplots( 2, 1, figsize=(10, 6), sharex=True )
        fig2.suptitle( f"Per-size statistics — {schedule_name} schedule, T={T}" )

        ax1.bar( range(len(sizes)), size_means, color="steelblue" )
        ax1.axhline( 0, color="red", ls="--", lw=1 )
        ax1.set_ylabel( "Mean" )
        ax1.set_title( "Mean of x_T per molecule size (should be ~0)" )

        ax2.bar( range(len(sizes)), size_stds, color="darkorange" )
        ax2.axhline( 1, color="red", ls="--", lw=1 )
        ax2.set_ylabel( "Std" )
        ax2.set_xlabel( "Molecule size (n_atoms)" )
        ax2.set_title( "Std of x_T per molecule size (should be ~1)" )
        ax2.set_xticks( range(len(sizes)) )
        ax2.set_xticklabels( [str(s) for s in sizes], rotation=45, ha="right" )

        fig2.tight_layout()

    # --- Per-component check for a few representative sizes ---
    representative = sizes[:3] + sizes[-3:]   # smallest and largest
    representative = sorted( set(representative) )
    for n_atoms in representative:
        x0_group = pt.stack( by_size[n_atoms] )
        x0_flat = x0_group.reshape( -1, 3 )
        t_group = pt.full( (x0_flat.shape[0],), T - 1, dtype=pt.long )
        xT_group, _ = schedule.q_sample( x0_flat, t_group )

        fig3, axes3 = plt.subplots( 1, 3, figsize=(14, 4) )
        fig3.suptitle( f"n_atoms = {n_atoms} — {schedule_name}, T={T}" )
        for c, label in enumerate( ["x", "y", "z"] ):
            ax = axes3[c]
            vals = xT_group[:, c].numpy()
            ax.hist( vals, bins=100, density=True, alpha=0.7 )
            ax.plot( z, norm.pdf(z), "r-", lw=2 )
            ax.set_title( f"{label}:  μ={vals.mean():.3f}  σ={vals.std():.3f}" )
            ax.set_xlim( -5, 5 )
        fig3.tight_layout()


def main():
    with open( "./data_config.json", "r" ) as config_file:
        data_config = json.load( config_file )
    data_directory = data_config["data_folder"]

    T = 1000
    max_reactions = 500   # limit to keep runtime reasonable

    dataset = TransitionPathDataset( "train", data_directory )
    print( f"Loading positions from {min(len(dataset), max_reactions)} reactions..." )
    samples = load_positions( dataset, max_reactions=max_reactions )
    print( f"Collected {len(samples)} frames" )

    for schedule_name in ["linear", "cosine"]:
        schedule = DDPMSchedule( T=T, schedule=schedule_name )
        check_forward_diffusion( samples, schedule, schedule_name, T )

    plt.show()


if __name__ == "__main__":
    main()
