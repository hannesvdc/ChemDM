import numpy as np
import torch as pt
from torch.utils.data import Dataset

from typing import Tuple

class TrajectoryDataset( Dataset ):

    def __init__( self, 
                  dataset_name : str,
                ) -> None:
        super().__init__()

        # Normalization
        self.scale = 2.0

        # Load the trajectory data
        self.trajectories = pt.tensor( np.load( f"./data/{dataset_name}_trajectories.npy" ), requires_grad=False) / self.scale # (n_trajectories, n_points, 4, 3)
        self.xA = self.trajectories[:,0,:,:] # (n_trajectories, 4, 3)
        self.xB = self.trajectories[:,-1,:,:] # (n_trajectories, 4, 3)
        self.arclenghts = pt.tensor( np.load( f"./data/{dataset_name}_arclenghts.npy" ), requires_grad=False ) # (n_trajectories, n_points)
        assert self.trajectories.shape[0] == self.arclenghts.shape[0], \
            f"Number of points per trajectory should be the same as the size of s_grid, but got {self.trajectories.shape[0]} and {self.arclenghts.shape}"
        
    def __len__( self ) -> int:
        return self.trajectories.shape[0] * self.trajectories.shape[1]
    
    def __getitem__( self, idx : int ) -> Tuple[pt.Tensor, pt.Tensor, pt.Tensor, pt.Tensor]:
        s_idx = idx % self.arclenghts.shape[1]
        traj_idx = idx // self.arclenghts.shape[1]
        return self.arclenghts[traj_idx,s_idx], self.trajectories[traj_idx,s_idx,:,:], self.xA[traj_idx,:,:], self.xB[traj_idx,:,:]