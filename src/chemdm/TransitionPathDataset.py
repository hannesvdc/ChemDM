import os
import torch as pt
from torch.utils.data import Dataset

from chemdm.Trajectory import Trajectory

import pickle
from typing import List

class TransitionPathDataset( Dataset ):
    def __init__( self, 
                  name : str,
                  data_directory : str ):
        super().__init__()

        self.name = name
        self.data_directory = data_directory
        self.loadTransition1xData( )

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

    def __len__( self ) -> int:
        return self.n_files
    
    def __getitem__( self, 
                     idx : int) -> List[Trajectory]:
        """
        Just return every piece of information at the current state of the reaction.
        """
        file_name = os.path.join( self.data_directory, f"{self.name}_reaction_{idx}.pkl")
        with open( file_name, "rb" ) as file:
            tp_list = pickle.load( file )
        return tp_list
    
if __name__ == '__main__':
    tp_dataset = TransitionPathDataset( "train", "/Users/hannesvdc/transition1x")
    print(tp_dataset[1001])