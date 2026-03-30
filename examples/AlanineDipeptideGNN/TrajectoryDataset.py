from pathlib import Path
import numpy as np
import torch as pt
from torch.utils.data import Dataset

from chemdm.Trajectory import Trajectory
from typing import List

class TrajectoryDataset( Dataset ):

    def __init__( self,
                  outdir = Path("outputs"),
                ) -> None:
        super().__init__()

        # Allowed Basin pairs
        self.basins = [ ("left_center", "left_wrap"),
                        ("left_wrap", "left_center"),
                        ("right_lower", "right_upper"),
                        ("right_upper", "right_lower"),
                        ("left_center", "right_lower"),
                        ("left_wrap", "right_lower"),
                        ("right_lower", "left_wrap"),
                        ("right_upper", "left_wrap"),]
        
        # Setup the datastructure.
        trajectories = []
        xAs = []
        xBs = []
        arclenghts = []
        
        # Load the filtered paths
        for basin_pair in self.basins:
            basin_A, basin_B = basin_pair
            filtered_path = outdir / f"{basin_A}__{basin_B}__neb_dataset_filtered.npz"
            filtered_trajectories = np.load( filtered_path )["x_opt"] # (n_trajectories, n_images, 22, 3)
            print(filtered_trajectories.shape)
            filtered_arclenghts = self.normalized_arclengths( filtered_trajectories ) # (n_trajectories, n_images)
            print(filtered_arclenghts.shape)

            trajectories.append( pt.tensor(filtered_trajectories) )
            xAs.append( pt.tensor(filtered_trajectories[:,0,:,:]) )
            xBs.append( pt.tensor(filtered_trajectories[:,-1,:,:]) )
            arclenghts.append( pt.tensor(filtered_arclenghts) )

        # Append into one big tensor
        self.trajectories = pt.cat( trajectories, dim=0 )
        self.xA = pt.cat( xAs, dim=0 )
        self.xB = pt.cat( xBs, dim=0 )
        self.arclengths = pt.cat( arclenghts, dim=0 )

        # Construct the neighbor graph shared among all molecules.
        self.Z = pt.tensor( [1, 6, 1, 1, 6, 8, 7, 1, 6, 1, 6, 1, 1, 1, 6, 8, 7, 1, 6, 1, 1, 1], dtype=pt.long )
        self.G = self.symmetrize([ [0,1], [2,1], [3,1], [1,4],  [4,5],  [4,6], 
                                   [6,7], [6,8], [8,9], [8,10], [8,14], [10,11],
                                   [10,12], [10,13], [14,15], [14,16], [16,17],
                                   [16,18], [18,19], [18,20], [18,21]])

    def normalized_arclengths(self, X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        """
        X: shape (n_traj, n_images, n_atoms, 3)

        Returns
        -------
        s: shape (n_traj, n_images)
            normalized arclength, with s[:, 0] = 0 and s[:, -1] = 1
        """
        dX = X[:, 1:] - X[:, :-1]   # (n_traj, n_images-1, n_atoms, 3)
        ds = np.linalg.norm(dX, axis=(2, 3))  # (n_traj, n_images-1)

        # Cumulative arclength, prepend 0
        s = np.concatenate( [np.zeros( (X.shape[0], 1), dtype=X.dtype), np.cumsum(ds, axis=1)], axis=1 )   # (n_traj, n_images)

        # Normalize by total length
        total = s[:, -1:]  # (n_traj, 1)
        s = s / np.maximum(total, eps)

        return s
    
    def symmetrize( self, bond_list : List[List] ):
        """
        Puts bonds in an (E,2) tensor and ensures symmetry.

        bond_list: List of bonds
        """
        bonds = pt.tensor( bond_list )
        assert bonds.ndim == 2 and bonds.shape[1] == 2, f"bonds must be of shape (E, 2)."

        src = bonds[:,0:1]
        dst = bonds[:,1:]
        inv_bonds = pt.cat( (dst, src), dim=1 )
        sym_bonds = pt.cat( (bonds, inv_bonds), dim=0 )
        return pt.unique( sym_bonds, dim=0 )


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
    
if __name__ == '__main__':
    dataset = TrajectoryDataset()