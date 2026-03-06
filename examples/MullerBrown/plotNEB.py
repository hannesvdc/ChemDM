import torch as pt
import matplotlib.pyplot as plt

from chemdm import NEB

from MullerBrown import potential, get_fixed_points, plotHelper

# Call NEB
xA = get_fixed_points()[4,:]
xS = get_fixed_points()[3,:]
xB = get_fixed_points()[2,:]
N = 100
k = 0.1
n_steps = 1000
neb0, neb_trajectory = NEB.computeMEP( potential, xA, xB, N, k, n_steps )

# Contour plot of the MB potential.
fig, ax = plotHelper()
ax.plot( neb0[:,0], neb0[:,1], marker='.', label='NEB0')
ax.plot( neb_trajectory[:,0], neb_trajectory[:,1], marker='.', label='NEB')
ax.scatter( xS[0], xS[1], marker='x', label='SP')
ax.set_xlabel( r"$x$" )
ax.set_ylabel( r"$y$" )
ax.legend()
ax.set_title( "NEB on the Muller-Brown Potential" )
plt.show()