"""
Sanity check: push the full dataset through forward diffusion at t=T-1
and verify the resulting distribution is close to N(0, I).
"""
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import numpy as np
import torch as pt
import matplotlib.pyplot as plt
from scipy.stats import norm
from pathlib import Path

from chemdm.DDPMSchedule import DDPMSchedule
from TrajectoryDataset import TrajectoryDataset


def main():
    dataset = TrajectoryDataset( outdir=Path("outputs") )

    # Collect all clean positions x_0
    x0_list = []
    for idx in range( len(dataset) ):
        trajectory = dataset[idx]
        x0_list.append( trajectory.x.reshape(-1, 3) )  # (n_images * n_atoms, 3)
    x_0 = pt.cat( x0_list, dim=0 )  # (N_total, 3)
    print( f"Total position samples: {x_0.shape[0]}" )

    # Forward diffuse to t = T-1
    T = 1000
    for schedule_name in ["linear", "cosine"]:
        schedule = DDPMSchedule( T=T, schedule=schedule_name )

        t = pt.full( (x_0.shape[0],), T - 1, dtype=pt.long )
        x_T, _ = schedule.q_sample( x_0, t )

        # Flatten to a 1D array for histogramming
        samples = x_T.numpy().flatten()

        fig, axes = plt.subplots( 1, 2, figsize=(12, 5) )

        # Histogram
        ax = axes[0]
        ax.hist( samples, bins=200, density=True, alpha=0.7, label=f"x_T ({schedule_name})" )
        z = np.linspace( -4, 4, 300 )
        ax.plot( z, norm.pdf(z), "r-", lw=2, label="N(0, 1)" )
        ax.set_xlabel( "Value" )
        ax.set_ylabel( "Density" )
        ax.set_title( f"{schedule_name} schedule, T={T}" )
        ax.legend()
        ax.set_xlim( -5, 5 )

        # QQ plot
        ax = axes[1]
        sorted_samples = np.sort( samples )
        n = len( sorted_samples )
        theoretical = norm.ppf( (np.arange(1, n+1) - 0.5) / n )
        # Subsample for plotting speed
        stride = max( 1, n // 5000 )
        ax.scatter( theoretical[::stride], sorted_samples[::stride], s=1, alpha=0.5 )
        ax.plot( [-4, 4], [-4, 4], "r-", lw=2 )
        ax.set_xlabel( "Theoretical quantiles" )
        ax.set_ylabel( "Sample quantiles" )
        ax.set_title( f"QQ plot ({schedule_name})" )
        ax.set_xlim( -4, 4 )
        ax.set_ylim( -4, 4 )
        ax.set_aspect( "equal" )

        fig.tight_layout()

        # Print summary statistics
        print( f"\n{schedule_name} schedule (T={T}):" )
        print( f"  alpha_bar[T-1] = {schedule.alpha_bar[-1].item():.2e}" )
        print( f"  mean = {samples.mean():.4f}  (should be ~0)" )
        print( f"  std  = {samples.std():.4f}  (should be ~1)" )
        print( f"  min  = {samples.min():.4f}  max = {samples.max():.4f}" )

    plt.show()


if __name__ == "__main__":
    main()
