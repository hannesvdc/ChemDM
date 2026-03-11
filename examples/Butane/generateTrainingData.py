import math
import torch as pt
import numpy as np
import matplotlib.pyplot as plt

import chemdm.NEB as NEB
from Butane import Butane, potential_internal, internalToCartesian, cartesianToInternal

from typing import Tuple

def generateRandomMolecules( phi_center : int,
                             B : int,
                             gen : pt.Generator,
                            ) -> pt.Tensor:
    """
    Generate `B` random butane molecules in the basin of attraction around the local minimum 
    given by `phi_center`:
        phi_center = -1 centers around phi = -1.14
        phi_center =  0 centers around phi = +- pi
        phi_center =  1 centers around phi = +1.14
    """

    # Sample the internal coordinates
    ka = Butane.k_theta
    theta0 = Butane.theta0
    theta_1_samples = theta0 + pt.randn( (B,1), generator=gen ) / math.sqrt( ka )
    theta_2_samples = theta0 + pt.randn( (B,1), generator=gen ) / math.sqrt( ka )
    phi = 2.0*phi_center + 0.05 * pt.randn( (B,1), generator=gen) # centered around -2, 0 or 2

    phi = phi % (2.0 * math.pi) - math.pi # between -pi and pi but centered at +- pi + (0.0, -2.0, 2.0)
    cos_phi = pt.cos( phi )
    sin_phi = pt.sin( phi )

    return pt.cat( ( pt.cos(theta_1_samples), pt.cos(theta_2_samples), cos_phi, sin_phi), dim=1 )

def sampleE3Invariant( gen : pt.Generator ) -> Tuple[pt.Tensor, pt.Tensor]:
    # Sample a random 3 x 3 matrix and calculate the QR decomposition
    M = pt.randn( (3,3,), generator=gen )
    Q, _ = pt.linalg.qr( M, mode='complete' )
    R = Q * pt.linalg.det( Q ) # ensure det(Q) = +1
    t = pt.randn( (3,), generator=gen )
    return R, t

def generateNEBTrajectories( N : int, 
                             k : float, 
                             n_steps : int, 
                             n_data_trajectories : int,
                             *,
                             generator : pt.Generator,
                            ) -> Tuple[pt.Tensor, pt.Tensor, pt.Tensor]:
    possible_paths = [ [0, 1], [1, 0], [1, -1], [-1, 1]]

    # Genrate paths in Cartesian coordinates for continuity
    def generate_initial_path( xA: pt.Tensor, # (4,)
                               xB : pt.Tensor,  # (4,)
                               t_grid : pt.Tensor, # (N,)
                             ):
        t_grid = t_grid.flatten()
        # xA_c = internalToCartesian( xA[None,:] )
        # xB_c = internalToCartesian( xB[None,:] )
        path = xA[None,:] + (xB[None,:] - xA[None,:]) * t_grid[:,None]
        path[:,2:4] = path[:,2:4] / pt.sqrt( path[:,2:3]**2 + path[:,3:4]**2 ) # rescale so cos**2 + sin**2 = 1
        # path_c = xA_c + (xB_c - xA_c) * t_grid[:,None,None]
        # path = cartesianToInternal( path_c )
        return path

    lr = 1e-4
    x = pt.zeros( (n_data_trajectories, N+1, 4, 3) )
    s_store = pt.zeros( (n_data_trajectories, N+1) )
    optimal_vals = pt.zeros( (n_data_trajectories,) )
    for n in range( n_data_trajectories ):
        print(n)
        path = pt.randint(0, 4, (1,), generator=generator )
        q1 = generateRandomMolecules( possible_paths[path][0], 1, generator ).flatten() # (4,)
        q2 = generateRandomMolecules( possible_paths[path][1], 1, generator ).flatten()

        # Sample a random rotation matrix to ensure equivariance during training (the network should enforce this!)
        _, q_path, F_optimal = NEB.computeMEP( potential_internal, q1, q2, N, k, n_steps, lr=lr, verbose=False, generate_initial_path=generate_initial_path)
        x_path = internalToCartesian( q_path ) # (N, 4, 3)
        print(F_optimal)

        # Sample a random matrix in E(3) to train / verify equivariance
        R, t = sampleE3Invariant( generator )
        x_path = pt.einsum( 'ij,bfj->bfi', R, x_path) + t[None,None,:] # (N,4,3)

        # Calulate the normalized arclengths in Cartesian coordinates
        diffs = x_path[1:,:,:] - x_path[:-1,:,:]
        seglen = pt.linalg.norm(diffs.flatten(start_dim=1), dim=1) # (N,)
        cumlen = pt.cat([pt.zeros(1,), pt.cumsum(seglen, dim=0)], dim=0) 
        s = cumlen / cumlen[-1]  

        # Re-parametrize by arclength on a grid

        # Store
        x[n,:,:,:] = x_path
        s_store[n,:] = s
        optimal_vals[n] = F_optimal

    return x, s_store, optimal_vals

def generateTrainingData():
    # NEB parameters
    N = 100
    k = 1e5
    n_steps = 75_000

    generator = pt.Generator()

    # Generate training and validation sample trajectories
    n_train_trajectories = 1000
    n_valid_trajectories = 100
    print('Generating Training Trajectories')
    train_trajectories, train_arclengths, train_optimal_vals = generateNEBTrajectories( N, k, n_steps, n_train_trajectories, generator=generator )
    print('Generating Validation Trajectories')
    valid_trajectories, valid_arclengths, valid_optimal_vals = generateNEBTrajectories( N, k, n_steps, n_valid_trajectories, generator=generator )

    # Store
    np.save( './data/train_extended_trajectories.npy', train_trajectories.detach().numpy() )
    np.save( './data/valid_extended_trajectories.npy', valid_trajectories.detach().numpy() )
    np.save( './data/train_arclengths.npy', train_arclengths.detach().numpy() )
    np.save( './data/valid_arclengths.npy', valid_arclengths.detach().numpy() )
    np.save( './data/train_optimal_vals.npy', train_optimal_vals.detach().numpy() )
    np.save( './data/valid_optimal_vals.npy', valid_optimal_vals.detach().numpy() )

    # Plot all trajectories for testing purposes.
    internal_coords = cartesianToInternal( train_trajectories ) # ( n_trajectories, N, 4 )
    cos_phi = internal_coords[:,:,2]
    sin_phi = internal_coords[:,:,3]
    phi = pt.atan2( sin_phi, cos_phi )
    for idx in range(n_train_trajectories):
        plt.plot( cos_phi[idx,:], sin_phi[idx,:])
    plt.xlabel( r"$\cos(\phi)$" )
    plt.ylabel( r"$\sin(\phi)$" )
    plt.title( "NEB Trajectories" )
    plt.legend()
    plt.figure()
    for idx in range(n_train_trajectories):
        plt.plot(train_arclengths[idx,:], phi[idx,:], label=r"$\phi(s)$")
    plt.show()

if __name__ == '__main__':
    generateTrainingData( )