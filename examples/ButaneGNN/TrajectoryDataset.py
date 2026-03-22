import numpy as np
import torch as pt
from torch.utils.data import Dataset

from chemdm.Trajectory import Trajectory

class TrajectoryDataset( Dataset ):

    def __init__( self, 
                  dataset_name : str,
                ) -> None:
        super().__init__()

        # Normalization
        self.scale = 1.0

        # Load the trajectory data
        self.trajectories = pt.tensor( np.load( f"./../Butane/data/{dataset_name}_trajectories.npy" ), requires_grad=False) / self.scale # (n_trajectories, n_points, 4, 3)
        self.trajectories = self.trajectories - pt.mean(self.trajectories,dim=2,keepdim=True)
        self.xA = self.trajectories[:,0,:,:] # (n_trajectories, 4, 3)
        self.xB = self.trajectories[:,-1,:,:] # (n_trajectories, 4, 3)
        self.arclengths = pt.tensor( np.load( f"./../Butane/data/{dataset_name}_arclengths.npy" ), requires_grad=False) # (n_trajectories, N)

        # Construct the neighbor graph
        self.Z = pt.tensor([6, 6, 6, 6], dtype=pt.long)
        self.G = pt.tensor( [[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2]], dtype=pt.long )

    def to( self, device : pt.device, dtype : pt.dtype ):
        self.trajectories.to(device=device, dtype=dtype)
        self.xA.to(device=device, dtype=dtype)
        self.xB.to(device=device, dtype=dtype)
        self.arclengths.to(device=device, dtype=dtype)
        self.Z.to(device=device)
        self.G.to(device=device)
        return self
        
    def __len__( self ) -> int:
        return self.trajectories.shape[0] 
    
    def __getitem__( self, idx : int ) -> Trajectory:
        trajectory = Trajectory(self.Z, 
                                self.xA[idx,:,:], 
                                self.xB[idx,:,:], 
                                self.G, 
                                self.G, 
                                self.arclengths[idx,:], 
                                self.trajectories[idx,:,:,] )
        return trajectory