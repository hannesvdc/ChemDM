import torch as pt
import matplotlib.pyplot as plt

from chemdm.MoleculeGraph import MoleculeGraph
from chemdm.MolecularEmbeddingNetwork import MolecularEmbeddingGNN
from chemdm.TransitionPathNetwork import TransitionPathGNN

from Butane import internalToCartesian, cartesianToInternal
from generateTrainingData import generateRandomMolecules

# Move to the GPU
device = pt.device( "cpu" )
dtype = pt.float32
pt.set_grad_enabled( False )

# Global molecular information
d_cutoff = 5.0 # Amstrong

# Construct the neural network architecture
embedding_state_size = 32
embedding_message_size = 32
n_embedding_layers = 3
xA_embedding = MolecularEmbeddingGNN(embedding_state_size, embedding_message_size, n_embedding_layers, d_cutoff, device=device, dtype=dtype)
xB_embedding = MolecularEmbeddingGNN(embedding_state_size, embedding_message_size, n_embedding_layers, d_cutoff, device=device, dtype=dtype)
n_tp_layers = 3
tp_message_size = 32
tp_network = TransitionPathGNN( xA_embedding, xB_embedding, tp_message_size, n_tp_layers, d_cutoff )
tp_network.load_state_dict( pt.load('./models/gnn.pth', map_location=device) )

# Sample a new xA and xB
Z = pt.tensor([6, 6, 6, 6], dtype=pt.long)
G = pt.tensor( [[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2]], dtype=pt.long )
gen = pt.Generator()
fp_1 = internalToCartesian( generateRandomMolecules(-1, 1, gen) ) # (1, 4, 3)
fp_2 = internalToCartesian( generateRandomMolecules(0, 1, gen) ) # (1, 4, 3)
xA = MoleculeGraph( Z, fp_1[0,:,:], G )
xB = MoleculeGraph( Z, fp_2[0,:,:], G )

# Propagate a full grid through the network
s_grid = pt.linspace( 0.0, 1.0, 2001, device=device, dtype=dtype ) # (B,)
x_eval = pt.zeros( (len(s_grid), 4, 3) )
for s_idx in range( len(s_grid) ):
    s = s_grid[s_idx] * pt.ones((4,)) # keep it a tensor
    xs = tp_network( xA, xB, s )
    x_eval[s_idx,:,:] = xs
xs = x_eval

# Plot torsion transition path
internal_coords = cartesianToInternal( xs )
cos_phi = internal_coords[:,2]
sin_phi = internal_coords[:,3]
phi = pt.atan2( sin_phi, cos_phi )
plt.plot( cos_phi, sin_phi )
plt.xlabel( r"$\cos(\phi)$" )
plt.ylabel( r"$\sin(\phi)$" )
plt.title( "Neural Trajectory" )
plt.legend()
plt.figure()
plt.plot( s_grid, phi, label=r"$\phi(s)$")
plt.xlabel( r"$s$")
plt.ylabel( r"$\phi(x)$" )
plt.show()
