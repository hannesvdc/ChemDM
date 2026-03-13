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
        self.scale = 1.0

        # Load the trajectory data
        self.trajectories = pt.tensor( np.load( f"./data/{dataset_name}_trajectories.npy" ), requires_grad=False) / self.scale # (n_trajectories, n_points, 4, 3)
        self.trajectories = self.trajectories - pt.mean(self.trajectories,dim=2,keepdim=True)
        self.xA = self.trajectories[:,0,:,:] # (n_trajectories, 4, 3)
        self.xB = self.trajectories[:,-1,:,:] # (n_trajectories, 4, 3)train_arclengths_filtered
        self.arclengths = pt.tensor( np.load( f"./data/{dataset_name}_arclengths.npy" ), requires_grad=False) # (n_trajectories, N)
        
    def __len__( self ) -> int:
        return self.trajectories.shape[0] * self.trajectories.shape[1]
    
    def __getitem__( self, idx : int ) -> Tuple[pt.Tensor, pt.Tensor, pt.Tensor, pt.Tensor]:
        s_idx = idx % self.arclengths.shape[1] #len(self.s_grid) #
        traj_idx = idx // self.arclengths.shape[1] #len(self.s_grid)#
        return self.arclengths[traj_idx,s_idx], self.trajectories[traj_idx,s_idx,:,:], self.xA[traj_idx,:,:], self.xB[traj_idx,:,:]