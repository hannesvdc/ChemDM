import torch as pt
from torch.autograd.functional import hessian
import numpy as np
import matplotlib.pyplot as plt

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
                            ) -> pt.Tensor:
    width = len(str(n_data_trajectories))

    # Calculate the Cholesky factorization of the Hessians for proper sampling
    sigma = 1.0
    jitter = 1e-10
    L1 = pt.linalg.cholesky(invH1 + jitter * pt.eye(2))
    L2 = pt.linalg.cholesky(invH2 + jitter * pt.eye(2))

    trajectories = pt.zeros( (N+1, 2, n_data_trajectories) )
    for n in range( n_data_trajectories ):
        print( f"\rGenerating {n+1:{width}} / {n_data_trajectories}", end="", flush=True)

        # correlated Gaussian perturbations
        zA = pt.randn( (2,), generator=generator )
        zB = pt.randn( (2,), generator=generator )

        xA = fp_1 + sigma * ( L1 @ zA )
        xB = fp_2 + sigma * ( L2 @ zB )

        _, neb_trajectory = NEB.computeMEP( potential, xA, xB, N, k, n_steps, verbose=False )
        trajectories[:,:,n] = neb_trajectory
    print("", end="\n", flush=True)
    return trajectories

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
train_trajectories = generateNEBTrajectories( fp_1, fp_2, invH_1, invH_2, N, k, n_steps, n_train_trajectories, generator)
print('Generating Validation Trajectories')
valid_trajectories = generateNEBTrajectories( fp_1, fp_2, invH_1, invH_2, N, k, n_steps, n_valid_trajectories, generator)

# Store
np.save( './data/train_trajectories.npy', train_trajectories.numpy() )
np.save( './data/valid_trajectories.npy', valid_trajectories.numpy() )

# Calculate the radial variance per arclength
traj_means = pt.mean( train_trajectories, dim=2 )
x_means = traj_means[:,0]
y_means = traj_means[:,1]
traj_vars = pt.mean( (train_trajectories[:,0,:] - x_means[:,None])**2 + (train_trajectories[:,1,:] - y_means[:,None])**2, dim=1 )

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
    plt.plot( train_trajectories[:,0,traj_index], train_trajectories[:,1,traj_index], marker='.' )
plt.scatter( xS[0], xS[1], marker='x', label='SP')
plt.xlabel( r"$x$" )
plt.ylabel( r"$y$" )
plt.legend()
plt.title( "Validation Trajectories" )

plt.figure()
arclengths = pt.linspace(0.0, 1.0, N+1)
plt.plot(arclengths.numpy(), traj_vars.numpy(), label=r"$\text{Var}(s)$")
plt.xlabel(r"$s$")
plt.legend()
plt.show()