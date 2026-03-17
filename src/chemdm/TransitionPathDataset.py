import os
import torch as pt
from torch.utils.data import Dataset

import pickle
from typing import List, Tuple

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
        n_reaction_states = [ 0 ]

        reaction_counter = 0
        while True: # we don't know the total number of reactions
            file_name = os.path.join( self.data_directory, f"{self.name}_reaction_{reaction_counter}.pkl")
            if not os.path.exists( file_name ):
                break

            with open( file_name, "rb" ) as file:
                tp_dict = pickle.load( file )
            n_reaction_states.append( n_reaction_states[-1] + len(tp_dict["s"]) )
            reaction_counter += 1
        self.n_reaction_states = pt.tensor( n_reaction_states, dtype=pt.long )
        print('...Done')

    def toBondStructure( self, 
                         Z : List[float],
                         bonds ) -> List[List[int]]:
        """
        Convert set-type bond structure to List of Lists as used in TransitionPathNetwork.
        """
        bond_structure = [ [] for _ in range(len(Z)) ]
        for bond in bonds:
            i,j = bond
            bond_structure[i].append( j )
            bond_structure[j].append( i )
        return bond_structure

    def __len__( self ) -> int:
        return int(self.n_reaction_states[-1])
    
    def __getitem__( self, 
                    idx : int
                   ) -> Tuple[pt.Tensor, pt.Tensor, pt.Tensor, pt.Tensor, pt.Tensor, List, List]:
        """
        Just return every piece of information at the current state of the reaction.
        """

        # Determine the file index
        reaction_idx = int( pt.searchsorted( self.n_reaction_states, idx, right=True )-1 )
        inner_idx = idx - int(self.n_reaction_states[reaction_idx])

        # Load this specific reaction from file (fast enough on SSD)
        file_name = os.path.join( self.data_directory, f"{self.name}_reaction_{reaction_idx}.pkl")
        if not os.path.exists( file_name ):
            print( "File index exceeds bounds. Aborting" )
            exit(-1)
        with open( file_name, "rb" ) as file:
            tp = pickle.load( file )

        # Extract all relevant information from the dict.
        xA = pt.tensor(tp["xA"], requires_grad=False)
        xB = pt.tensor(tp["xB"], requires_grad=False)
        s = pt.tensor(tp["s"][inner_idx], requires_grad=False)
        x = pt.tensor(tp["pos"][inner_idx,:,:], requires_grad=False)
        Z = tp["Z"]
        bondsA = self.toBondStructure( Z, tp["bondsA"] )
        bondsB = self.toBondStructure( Z, tp["bondsB"] )

        return xA, xB, s, x, pt.tensor(Z, dtype=pt.long), bondsA, bondsB
    
if __name__ == '__main__':
    tp_dataset = TransitionPathDataset( "train", "/Users/hannesvdc/transition1x")
    print(tp_dataset[1001])