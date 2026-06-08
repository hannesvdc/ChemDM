import os
from torch.utils.data import Dataset

from chemdm.Trajectory import Trajectory

import pickle
import json
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
        file_names.sort( key=lambda fn: int(fn[len(prefix):-len(suffix)]) )
        self.file_names = file_names
        self.n_files = len( self.file_names )

        print(f"...Done ({self.n_files} files)")

        print("Loading Metadata")
        with open(os.path.join(self.data_directory, f"{self.name}_metadata.json"), "r") as mf:
            metadata = json.load(mf)

        # JSON object keys are strings.
        self.reaction_weights = { int(k): float(v) for k, v in metadata["reaction_weights"].items() }

        # Store weights as a simple list
        self.sample_weights = [ self.reaction_weights[self.reaction_id_from_filename(fn)] for fn in self.file_names ]

    def reaction_id_from_filename(self, fn: str) -> int:
        prefix = f"{self.name}_reaction_"
        suffix = ".pkl"
        return int(fn[len(prefix):-len(suffix)])

    def __len__( self ) -> int:
        return self.n_files
    
    def __getitem__( self, idx : int) -> List[Trajectory]:
        """
        Just return every piece of information at the current state of the reaction.
        """
        data_filename = self.file_names[idx]
        file_name = os.path.join( self.data_directory, data_filename )
        with open( file_name, "rb" ) as file:
            tp_list = pickle.load( file )
        if not isinstance( tp_list, list ):
            return [tp_list]
        return tp_list