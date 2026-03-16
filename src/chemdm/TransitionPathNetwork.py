import torch as pt
import torch.nn as nn

from chemdm.AtomicInformation import AtomicInformation
from chemdm.MLP import MultiLayerPerceptron

class TransitionPathNetwork( nn.Module ):
    """
    Main neural network to predict transition paths, combining chemical information network,
    initial and final state embedding networks, arclength embedding network, and main graph neural network.
    """
    def __init__( self ) -> None:
        super().__init__()

        # Relevant Chemical Information
        self.atomic_information = AtomicInformation()
        atomic_embedding_neurons = [self.atomic_information.numberOfOutputs(), 64, 16]
        self.atomic_info_embedding = MultiLayerPerceptron( atomic_embedding_neurons, act=nn.Sigmoid, name="atom_info" )