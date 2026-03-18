import random
import torch as pt
from chemdm.MolecularEmbeddingNetwork import MolecularEmbeddingNetwork
from chemdm.TransitionPathDataset import TransitionPathDataset

def testEmbeddingNetwork():
    # Load the full dataset from file
    dataset = TransitionPathDataset( "train", "/Users/hannesvdc/transition1x")
    
    # Construct the network
    state_size = 64
    message_size = 128
    n_layers = 10
    d_cutoff = 5.0
    network = MolecularEmbeddingNetwork( state_size, d_cutoff, n_layers, message_size)
    network = network.to( dtype=pt.float64 )

    # Sample the dataset randomly and pass through the network.
    for samples in range(10):
        idx = random.randint( 0, len(dataset), )
        xA, xB, s, x, Z, bondsA, bondsB = dataset[idx]

        print(network(Z, xA, bondsA).shape)
        print(network(Z, xB, bondsB).shape)

def testButane():
    """
    Create a mockup of the butane molecule and check if the network behaves intuitively.
    """
    Z = pt.tensor( [1, 1, 1, 6, 6, 1, 1, 6, 1, 1, 6, 1, 1, 1])
    x = pt.randn( (len(Z),3) )
    G = [ [3], [3], [3], [0, 1, 2, 4], [3, 5, 6], [4], [4], [4, 8, 9, 10], [7], [7], [7, 11, 12, 13], [10], [10], [10]]

    # Construct the network
    state_size = 64
    message_size = 128
    n_layers = 10
    d_cutoff = 5.0
    network = MolecularEmbeddingNetwork( state_size, d_cutoff, n_layers, message_size)
    network = network.to( dtype=pt.float64 )
    print(network(Z, x, G).shape)

if __name__ == '__main__':
    testEmbeddingNetwork()
    testButane()