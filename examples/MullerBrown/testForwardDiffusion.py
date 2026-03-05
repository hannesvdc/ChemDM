import os
import numpy as np
import matplotlib.pyplot as plt

from chemdm.TrajectoryDataset import TrajectoryDataset
from chemdm.diffusion import forwardSDE

data_directory = './data/'
data_directory = os.path.abspath( data_directory )
dataset = TrajectoryDataset( data_directory, dataset_name="train" )

# Get the complete dataset.
print(len(dataset))
idx = 43957
_, initial_data, _, _ = dataset[idx]
ns = 100_000
initial_data = initial_data[None,:].expand(ns, 2)

# Do forward diffusion on the complete dataset
dt = 1e-2
noise = forwardSDE( initial_data, dt )

# Take a random marginal and plot a histogram
x_vals = noise[:,0]
y_vals = noise[:,1]
z = np.linspace( -4.0, 4.0, 1001 )
nd = np.exp( -z**2 / 2.0 ) / np.sqrt( 2.0 * np.pi )
plt.hist( x_vals.numpy(), density=True, bins=100 )
plt.plot( z, nd )
plt.xlabel( r"$x$")
plt.figure()
plt.hist( y_vals.numpy(), density=True, bins=100 )
plt.plot( z, nd )
plt.xlabel( r"$x$")
plt.show()