import os
import numpy as np
import torch as pt
from torch.utils.data import Dataset

from typing import Optional

class TrajectoryDataset( Dataset ):

    def __init__( self, 
                  data_folder : str,
                  dataset_name : str,
                  *,
                  center : Optional[pt.Tensor] = None,
                  diff : Optional[pt.Tensor] = None,
                ) -> None:
        super().__init__()

        # Normalization
        if center is None or diff is None:
            self.xmin = -1.0
            self.xmax = 0.0
            self.ymin = 0.5
            self.ymax = 1.5
            self.center = pt.tensor([[(self.xmin + self.xmax) / 2,  (self.ymin + self.ymax) / 2]] )
            self.diff = pt.tensor( [[self.xmax - self.xmin, self.ymax-self.ymin]] )
        else:
            self.center = center
            self.diff = diff

        # Load the trajectory data
        s_grid = np.load( os.path.join(data_folder, "s_grid.npy") )
        self.s_grid = pt.tensor( s_grid )
        trajectories = np.load( os.path.join(data_folder, dataset_name+"_trajectories.npy") )
        self.trajectories = (pt.tensor( trajectories ) - self.center[:,:,None]) / self.diff[:,:,None] # shape ( len(s_grid), 2, n_trajectories )
        assert self.trajectories.shape[0] == self.s_grid.shape[0], \
            f"Number of points per trajectory should be the same as the size of s_grid, but got {self.trajectories.shape[0]} and {s_grid.shape}"
        
        # Also store the beginning and end points of each trajectory
        # Already normalized.
        self.xA = self.trajectories[0,:,:]
        self.xB = self.trajectories[-1,:,:]
        
    def __len__( self ) -> int:
        return self.trajectories.shape[0] * self.trajectories.shape[2]
    
    def __getitem__( self, idx : int ):
        s_idx = idx % len( self.s_grid )
        traj_idx = idx // len( self.s_grid )
        return self.s_grid[s_idx], self.trajectories[s_idx,:,traj_idx], self.xA[:,traj_idx], self.xB[:,traj_idx]