import torch as pt
from torch.autograd.functional import hessian
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

from chemdm import NEB
from MullerBrown import potential, get_fixed_points, plotHelper

def inverseHessianAt( x : pt.Tensor ) -> pt.Tensor:
    x = x.detach().requires_grad_(True)
    def f_scalar(z):
        return potential(z[None, :]).squeeze()  # f accepts (N,d); here N=1
    H = hessian( f_scalar, x )
    return pt.inverse( H )

def generateSample( fp : pt.Tensor,
                     L : pt.Tensor, 
                     sigma : float ) -> pt.Tensor:
    z = pt.randn( (2,), generator=generator )
    x = fp + sigma * ( L @ z )
    return x

def generateNEBTrajectories( fp_1 : pt.Tensor, 
                             fp_2 : pt.Tensor, 
                             fp_3 : pt.Tensor,
                             invH1 : pt.Tensor,
                             invH2 : pt.Tensor,
                             invH3 : pt.Tensor,
                             N : int, 
                             k : float, 
                             n_steps : int, 
                             n_data_trajectories : int,
                             generator : pt.Generator,
                            ) -> Tuple[pt.Tensor, pt.Tensor]:
    width = len(str(n_data_trajectories))

    # Calculate the Cholesky factorization of the Hessians for proper sampling
    sigma = 1.0
    jitter = 1e-10
    L1 = pt.linalg.cholesky(invH1 + jitter * pt.eye(2))
    L2 = pt.linalg.cholesky(invH2 + jitter * pt.eye(2))
    L3 = pt.linalg.cholesky(invH3 + jitter * pt.eye(2))
    fps = [fp_1, fp_2, fp_3]
    Ls = [L1, L2, L3]
    sample_pairs = [ [0,1], [1,0], [1,2], [2,1] ]

    trajectories = pt.zeros( (N+1, 2, n_data_trajectories) )
    s_store = pt.zeros( (N+1, n_data_trajectories) )
    for n in range( n_data_trajectories ):
        print( f"\rGenerating {n+1:{width}} / {n_data_trajectories}", end="", flush=True)

        pair_idx = pt.randint(0, 4, (1,), generator=generator)
        sp = sample_pairs[pair_idx][0]
        ep = sample_pairs[pair_idx][1]
        xA = generateSample( fps[sp], Ls[sp], sigma )
        xB = generateSample( fps[ep], Ls[ep], sigma )

        _, neb_trajectory = NEB.computeMEP( potential, xA, xB, N, k, n_steps, verbose=False )
        trajectories[:,:,n] = neb_trajectory

        # Calulate the normalized arclengths.
        diffs = neb_trajectory[1:,:] - neb_trajectory[:-1,:]
        seglen = pt.linalg.norm(diffs, dim=1) # (N,)
        cumlen = pt.cat([pt.zeros(1,), pt.cumsum(seglen, dim=0)], dim=0) 
        s = cumlen / cumlen[-1]  
        s_store[:, n] = s

    print("", end="\n", flush=True)
    return trajectories, s_store

def variance_vs_s(
    trajectories: pt.Tensor,   # (N+1, 2, B)
    s_store: pt.Tensor,        # (N+1, B), normalized arclength per trajectory
    M: int = 200,              # number of points on common s-grid
    eps: float = 1e-12
) -> Tuple[pt.Tensor, pt.Tensor, pt.Tensor, pt.Tensor]:
    """
    Returns:
      s_grid: (M,)
      radial_var: (M,)     E[ ||x - mean||^2 ] at each s
      mean_path: (M,2)
    """
    assert trajectories.ndim == 3 and trajectories.shape[1] == 2
    Np1, _, B = trajectories.shape
    assert s_store.shape == (Np1, B)

    # Common s grid
    s_grid = pt.linspace(0.0, 1.0, M )

    # Resampled trajectories on common grid: (M,2,B)
    X = pt.empty( (M, 2, B) )

    for b in range(B):
        sb = s_store[:, b]          # (N+1,)
        xb = trajectories[:, :, b]  # (N+1,2)

        # Ensure sb is nondecreasing (numerical safety)
        sb = pt.maximum(sb, pt.cat([sb[:1], sb[:-1]]))  # monotone-ish
        sb = sb / (sb[-1] + eps)

        # For each s in s_grid, find interval [sb[i], sb[i+1]]
        idx = pt.searchsorted(sb, s_grid, right=True) - 1
        idx = idx.clamp(0, Np1 - 2)

        s0 = sb[idx]           # (M,)
        s1 = sb[idx + 1]       # (M,)
        x0 = xb[idx]           # (M,2)
        x1 = xb[idx + 1]       # (M,2)

        w = (s_grid - s0) / (s1 - s0 + eps)   # (M,)
        X[:, :, b] = x0 + w[:, None] * (x1 - x0)

    # Mean path and radial variance at each s_grid point
    mean_path = X.mean(dim=2)                    # (M,2)
    radial_var = ((X - mean_path[:, :, None])**2).sum(dim=1).mean(dim=1)  # (M,)

    return s_grid, X, radial_var, mean_path

if __name__ == '__main__':
    # Generate random starting and end points around the local minima and run NEB for every pair.
    fp_1 = get_fixed_points()[4,:]
    xS1 = get_fixed_points()[3,:]
    fp_2 = get_fixed_points()[2,:]
    xS2 = get_fixed_points()[1,:]
    fp_3 = get_fixed_points()[0,:]

    # Compute the local hessian at the local minima
    invH_1 = inverseHessianAt( fp_1 )
    invH_2 = inverseHessianAt( fp_2 )
    invH_3 = inverseHessianAt( fp_3 )

    # NEB parameters
    N = 100
    k = 10.0
    n_steps = 1000

    # For every (xA, xB), run NEB
    generator = pt.Generator( )
    n_train_trajectories = 1000
    n_valid_trajectories = 100
    print('Generating Training Trajectories')
    train_trajectories, train_arclengths = generateNEBTrajectories( fp_1, fp_2, fp_3, invH_1, invH_2, invH_3, 
                                                                    N, k, n_steps, n_train_trajectories, generator)
    print('Generating Validation Trajectories')
    valid_trajectories, valid_arclengths = generateNEBTrajectories( fp_1, fp_2, fp_3, invH_1, invH_2, invH_3, 
                                                                    N, k, n_steps, n_valid_trajectories, generator)

    # Interpolate the trajectories on a fixed grid of normalized arclength values
    s_grid, train_traj_int, radial_var, mean_path = variance_vs_s(train_trajectories, train_arclengths, M=200)
    _, valid_traj_int, _, _ = variance_vs_s(valid_trajectories, valid_arclengths, M=200)

    # Store
    np.save( './data/train_extended_trajectories.npy', train_traj_int.numpy() )
    np.save( './data/valid_extended_trajectories.npy', valid_traj_int.numpy() )
    np.save( './data/s_grid.npy', s_grid.numpy() )

    # Plot the validation trajectories
    # Contour plot of the MB potential.
    fig, ax = plotHelper( )
    for traj_index in range( train_trajectories.shape[2] ):
        ax.plot( train_traj_int[:,0,traj_index], train_traj_int[:,1,traj_index], marker='.' )
    ax.scatter( xS1[0], xS1[1], marker='x', label='SP')
    ax.scatter( xS2[0], xS2[1], marker='x', label='SP')
    ax.set_xlabel( r"$x$" )
    ax.set_ylabel( r"$y$" )
    ax.legend()
    ax.set_title( "Trajectories" )

    plt.figure()
    arclengths = pt.linspace(0.0, 1.0, N+1)
    plt.plot(s_grid.numpy(), radial_var.numpy(), label=r"$\text{Var}(s)$")
    plt.xlabel(r"$s$")
    plt.legend()
    plt.title("Radial Variance")
    plt.show()