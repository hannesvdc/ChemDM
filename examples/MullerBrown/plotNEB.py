import torch as pt
import matplotlib.pyplot as plt

from chemdm import NEB

from MullerBrown import potential, get_fixed_points, plotHelper

# Call NEB
xA = get_fixed_points()[4,:]
xS = get_fixed_points()[3,:]
xB = get_fixed_points()[2,:]
N = 100
k = 100.0
n_steps = 10000
neb0, neb_trajectory,_ = NEB.computeMEP( potential, xA, xB, N, k, n_steps )

# Contour plot of the MB potential.
fig, ax = plotHelper()
# ax.plot( neb0[:,0], neb0[:,1], linewidth=2.0, color='tab:orange')
ax.plot( neb_trajectory[:,0], neb_trajectory[:,1], linewidth=4.0, color='tab:purple' )
ax.scatter( xA[0], xA[1], marker='o', s=100)
ax.scatter( xS[0], xS[1], marker='o', s=100)
ax.scatter( xB[0], xB[1], marker='o', s=100)
ax.text( float(xA[0])+0.03, float(xA[1])+0.01, "R", fontsize=32)
ax.text( float(xB[0])+0.05, float(xB[1])-0.03, "P", fontsize=32)
ax.text( float(xS[0])+0.05, float(xS[1])+0.02, "TS", fontsize=32)
ax.set_xlabel( r"$x$" )
ax.set_ylabel( r"$y$" )
ax.set_axis_off()

ax.set_aspect("equal", adjustable="box")
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
fig.savefig( "transition_path.svg", format="svg", transparent=True )
plt.show()