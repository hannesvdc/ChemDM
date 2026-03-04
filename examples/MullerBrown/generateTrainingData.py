import torch as pt
from torch.autograd.functional import hessian
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

from chemdm import NEB
from MullerBrown import potential, get_fixed_points

def inverseHessianAt( x : pt.Tensor ) -> pt.Tensor:
    x = x.detach().requires_grad_(True)
    def f_scalar(z):
        return potential(z[None, :]).squeeze()  # f accepts (N,d); here N=1
    H = hessian( f_scalar, x )
    return pt.inverse( H )

def generateNEBTrajectories( fp_1 : pt.Tensor, 
                             fp_2 : pt.Tensor, 
                             invH1 : pt.Tensor,
                             invH2 : pt.Tensor,
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

    trajectories = pt.zeros( (N+1, 2, n_data_trajectories) )
    s_store = pt.zeros( (N+1, n_data_trajectories) )
    for n in range( n_data_trajectories ):
        print( f"\rGenerating {n+1:{width}} / {n_data_trajectories}", end="", flush=True)

        # correlated Gaussian perturbations
        zA = pt.randn( (2,), generator=generator )
        zB = pt.randn( (2,), generator=generator )

        xA = fp_1 + sigma * ( L1 @ zA )
        xB = fp_2 + sigma * ( L2 @ zB )

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

# Generate random starting and end points around the local minima and run NEB for every pair.
fp_1 = get_fixed_points()[4,:]
xS = get_fixed_points()[3,:]
fp_2 = get_fixed_points()[2,:]

# Compute the local hessian at the local minima
invH_1 = inverseHessianAt( fp_1 )
invH_2 = inverseHessianAt( fp_2 )

# NEB parameters
N = 100
k = 0.1
n_steps = 1000

# For every (xA, xB), run NEB
generator = pt.Generator( )
n_train_trajectories = 1000
n_valid_trajectories = 100
print('Generating Training Trajectories')
train_trajectories, train_arclengths = generateNEBTrajectories( fp_1, fp_2, invH_1, invH_2, N, k, n_steps, n_train_trajectories, generator)
print('Generating Validation Trajectories')
valid_trajectories, valid_arclengths = generateNEBTrajectories( fp_1, fp_2, invH_1, invH_2, N, k, n_steps, n_valid_trajectories, generator)

# Interpolate the trajectories on a fixed grid of normalized arclength values
s_grid, train_traj_int, radial_var, mean_path = variance_vs_s(train_trajectories, train_arclengths, M=200)
_, valid_traj_int, _, _ = variance_vs_s(valid_trajectories, valid_arclengths, M=200)

# Store
np.save( './data/train_trajectories.npy', train_traj_int.numpy() )
np.save( './data/valid_trajectories.npy', valid_traj_int.numpy() )
np.save( './data/s_grid.npy', s_grid.numpy() )

# Plot the validation trajectories
# Contour plot of the MB potential.
n_plot_points = 1001
x_min = -1.2
x_max = 1.0
y_min = -0.4
y_max = 1.8
X = pt.linspace( x_min, x_max, n_plot_points)
Y = pt.linspace( y_min, y_max, n_plot_points)
X, Y = pt.meshgrid(X, Y, indexing="ij")
XY = pt.cat( (X.flatten()[:,None], Y.flatten()[:,None]), dim=1 )
Z = potential( XY )
Z = Z.reshape( (n_plot_points, n_plot_points) )

plt.contour( X, Y, Z, levels=101 )
for traj_index in range( train_trajectories.shape[2] ):
    plt.plot( train_traj_int[:,0,traj_index], train_traj_int[:,1,traj_index], marker='.' )
plt.scatter( xS[0], xS[1], marker='x', label='SP')
plt.xlabel( r"$x$" )
plt.ylabel( r"$y$" )
plt.legend()
plt.title( "Validation Trajectories" )

plt.figure()
arclengths = pt.linspace(0.0, 1.0, N+1)
plt.plot(s_grid.numpy(), radial_var.numpy(), label=r"$\text{Var}(s)$")
plt.xlabel(r"$s$")
plt.legend()
plt.title("Radial Variance")
plt.show()