import os
import torch as pt
import matplotlib.pyplot as plt

from generateTrainingData import inverseHessianAt
from MullerBrown import get_fixed_points, plotHelper

from chemdm.RegressionNetwork import RegressionNetwork
from chemdm.TrajectoryDataset import TrajectoryDataset

# Set global device and dtype, except for the dataloader
dtype = pt.float64
device = pt.device( "cpu" )
pt.set_grad_enabled( False )

data_folder = os.path.abspath( './data' )
train_dataset = TrajectoryDataset( data_folder, "train" )

# Build the complicated FiLM Scoring Network
n_embeddings = 4
hidden_layers = [64, 64, 64]
regression_model = RegressionNetwork( n_embeddings, hidden_layers ).to( device=device, dtype=dtype )
regression_model.load_state_dict( pt.load("./models/muller_brown_regressor.pth", map_location=device, weights_only=True) )

# Sample a new xA and xB
fp_1 = get_fixed_points()[4,:]
xS = get_fixed_points()[3,:]
fp_2 = get_fixed_points()[2,:]
invH_1 = inverseHessianAt( fp_1 )
invH_2 = inverseHessianAt( fp_2 )
jitter = 1e-10
L1 = pt.linalg.cholesky(invH_1 + jitter * pt.eye(2))
L2 = pt.linalg.cholesky(invH_2 + jitter * pt.eye(2))

# correlated Gaussian perturbations
sigma = 1.0
zA = pt.randn( (2,) )
zB = pt.randn( (2,) )
xA = fp_1 + sigma * ( L1 @ zA )
xB = fp_2 + sigma * ( L2 @ zB )
xA_normalized = (xA[None,:] - train_dataset.center) / train_dataset.diff
xB_normalized = (xB[None,:] - train_dataset.center) / train_dataset.diff
print(xA_normalized.shape)

# Propagate a full grid through the network
s_grid = pt.linspace( 0.0, 1.0, 201, device=device, dtype=dtype ) # (B,)
xA_normalized = xA_normalized.expand( len(s_grid), 2 ) # (B,2)
xB_normalized = xB_normalized.expand( len(s_grid), 2 ) # (B,2)
xs = regression_model( xA_normalized, xB_normalized, s_grid) # (B,2)
xs = xs * train_dataset.diff + train_dataset.center
print(xs.shape)

# Plot the path on the MB potential
fig, ax = plotHelper()
ax.plot( xs[:,0], xs[:,1], label='Transition Path' )
ax.scatter( xS[0], xS[1], marker='x', label='SP')
ax.scatter( xA[0], xA[1], marker='x', label=r'$x_A$')
ax.scatter( xB[0], xB[1], marker='x', label=r'$x_B$')
ax.set_xlabel( r"$x$" )
ax.set_ylabel( r"$y$" )
ax.legend()
plt.show()