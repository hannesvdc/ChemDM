import torch as pt
import matplotlib.pyplot as plt

from chemdm import NEB

from MullerBrown import potential, get_fixed_points

# Call NEB
xA = get_fixed_points()[4,:]
xS = get_fixed_points()[3,:]
xB = get_fixed_points()[2,:]
N = 100
k = 0.1
n_steps = 1000
neb0, neb_trajectory = NEB.computeMEP( potential, xA, xB, N, k, n_steps )

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
plt.plot( neb0[:,0], neb0[:,1], marker='.', label='NEB0')
plt.plot( neb_trajectory[:,0], neb_trajectory[:,1], marker='.', label='NEB')
plt.scatter( xS[0], xS[1], marker='x', label='SP')
plt.xlabel( r"$x$" )
plt.ylabel( r"$y$" )
plt.legend()
plt.title( "NEB on the Muller-Brown Potential" )
plt.show()