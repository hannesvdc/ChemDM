import os
import torch as pt
from torch.utils.data import Dataset

from chemdm.Trajectory import Trajectory

import pickle
from typing import List, Set

class TransitionPathDataset( Dataset ):
    def __init__( self, 
                  name : str,
                  data_directory : str ):
        super().__init__()

        self.name = name
        self.data_directory = data_directory
        self.device = pt.device( "cpu" )
        self.dtype = pt.float64
        self.loadTransition1xData( )

    def to( self, device : pt.device, dtype : pt.dtype ):
        self.device = device
        self.dtype = dtype
        return self

    def loadTransition1xData( self ) -> None:
        print("Indexing Reactions...")

        prefix = f"{self.name}_reaction_"
        suffix = ".pkl"
        file_names = [
            fn for fn in os.listdir(self.data_directory)
            if fn.startswith(prefix) and fn.endswith(suffix)
        ]
        file_names.sort(
            key=lambda fn: int(fn[len(prefix):-len(suffix)])
        )
        self.n_files = len(file_names)
        self.file_names = file_names

        print(f"...Done ({self.n_files} files)")

    def toBondStructure( self, 
                         Z : List[float],
                         bonds ) -> List[Set[int]]:
        """
        Convert set-type bond structure to List of Lists as used in TransitionPathNetwork.
        """
        bond_structure = [ set() for _ in range(len(Z)) ]
        for bond in bonds:
            i,j = bond
            bond_structure[i].add( j )
            bond_structure[j].add( i )
        return bond_structure

    def __len__( self ) -> int:
        return self.n_files
    
    
    def __getitem__( self, 
                     idx : int) -> Trajectory:
        """
        Just return every piece of information at the current state of the reaction.
        """
        file_name = os.path.join( self.data_directory, f"{self.name}_reaction_{idx}.pkl")
        with open( file_name, "rb" ) as file:
            tp = pickle.load( file )
            Z = pt.tensor(tp["Z"], device=self.device, dtype=pt.long)
            xA = pt.tensor(tp["xA"], requires_grad=False, device=self.device, dtype=self.dtype )
            xB = pt.tensor(tp["xB"], requires_grad=False, device=self.device, dtype=self.dtype )
            s = pt.tensor(tp["s"], requires_grad=False, device=self.device, dtype=self.dtype )
            x = pt.tensor(tp["pos"], requires_grad=False, device=self.device, dtype=self.dtype )
            bondsA = tp["bondsA"]
            bondsB = tp["bondsB"]
        trajectory = Trajectory(Z, xA, xB, bondsA, bondsB, s, x)
        return trajectory
    
if __name__ == '__main__':
    tp_dataset = TransitionPathDataset( "train", "/Users/hannesvdc/transition1x")
    print(tp_dataset[1001])