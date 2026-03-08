import os
import torch as pt
import matplotlib.pyplot as plt

from chemdm.TrajectoryDataset import TrajectoryDataset
from chemdm.ScoreNetwork import ScoreNetwork
from chemdm.timesteppers import sample_sgm_euler

from generateTrainingData import inverseHessianAt
from MullerBrown import get_fixed_points, plotHelper

# Set global device and dtype, except for the dataloader
dtype = pt.float32
device = pt.device("cpu")
pt.set_grad_enabled(False)
pt.set_default_device(device)
pt.set_default_dtype(dtype)

# Load the full dataset
data_folder = os.path.abspath( './data' )
dataset = TrajectoryDataset( data_folder, "train" )

# Load the model
n_embeddings = 4
hidden_layers = [64, 64, 64]
film_hidden_layers = [64, 64]
score_model = ScoreNetwork(n_embeddings, hidden_layers, film_hidden_layers).to( device=device )
score_model.load_state_dict(pt.load("./models/muller_brown_diffusion_extended.pth", map_location=device, weights_only=True))
score_model.eval()

# Sample a new xA and xB
fp_1 = get_fixed_points()[0,:]
xS1 = get_fixed_points()[1,:]
xS2 = get_fixed_points()[3,:]
fp_2 = get_fixed_points()[4,:]
invH_1 = inverseHessianAt( fp_1 )
invH_2 = inverseHessianAt( fp_2 )
jitter = 1e-10
L1 = pt.linalg.cholesky(invH_1 + jitter * pt.eye(2))
L2 = pt.linalg.cholesky(invH_2 + jitter * pt.eye(2))

# correlated Gaussian perturbations
sigma = 1.0
n_paths = 25
zA = pt.randn( (2, n_paths) )
zB = pt.randn( (2, n_paths) )
xA = fp_1[:,None] + sigma * ( L1 @ zA )
xB = fp_2[:,None] + sigma * ( L2 @ zB )
xA_normalized = (xA.T - dataset.center) / dataset.diff
xB_normalized = (xB.T - dataset.center) / dataset.diff

# Generate the SGM solution
print('Backward SDE Simulation..')
dt = 1e-3
s_grid = pt.linspace( 0.0, 1.0, 201, device=device, dtype=dtype ) # (B,)
trajectories = sample_sgm_euler( score_model, xA_normalized, xB_normalized, s_grid, dt ) # (B, S, 2)
trajectories = trajectories * dataset.diff[None,0,:] + dataset.center[None,0,:]
x_traj = trajectories[:,:,0].detach().numpy()
y_traj = trajectories[:,:,1].detach().numpy() # (B, len(s_grid))

# Plot the path on the MB potential
fig, ax = plotHelper()
ax.plot( x_traj.T, y_traj.T )
ax.scatter( xS2[0], xS2[1], marker='x', label='SP')
ax.scatter( xS1[0], xS1[1], marker='x', label='SP')
ax.set_xlabel( r"$x$" )
ax.set_ylabel( r"$y$" )
ax.legend()
plt.show()