import torch as pt
import matplotlib.pyplot as plt

from generateTrainingData import generateRandomMolecules
from EquivariantNetwork import EquivariantNetwork
from Butane import internalToCartesian, cartesianToInternal

# Set global device and dtype, except for the dataloader
dtype = pt.float64
device = pt.device( "cpu" )
pt.set_grad_enabled( False )

# Build the complicated FiLM Scoring Network
n_embeddings = 8
hidden_layers = [64, 64, 64]
regression_model = EquivariantNetwork( n_embeddings, hidden_layers )
n_params = sum( p.numel() for p in regression_model.parameters() if p.requires_grad )
print(f"Total trainable parameters: {n_params:,}")
regression_model.load_state_dict( pt.load("./models/regressor.pth", map_location=device, weights_only=True) )
regression_model = regression_model.to( device=device, dtype=dtype )

# Sample a new xA and xB
gen = pt.Generator()
fp_1 = internalToCartesian( generateRandomMolecules(-1, 1, gen) ) # (1, 4, 3)
fp_2 = internalToCartesian( generateRandomMolecules(0, 1, gen) ) # (1, 4, 3)

# Propagate a full grid through the network
s_grid = pt.linspace( 0.0, 1.0, 2001, device=device, dtype=dtype ) # (B,)
xA = fp_1.expand( len(s_grid), 4, 3 )
xB = fp_2.expand( len(s_grid), 4, 3 )
xs = regression_model( xA, xB, s_grid )

# Post-process by enforcing the end points exactly. This is optional
f = 10.0
xA_factor = pt.exp(-f*s_grid).reshape( (len(s_grid), 1, 1) )
xB_factor = pt.exp(-f*(1-s_grid)).reshape( (len(s_grid), 1, 1) )
xs = fp_1 * xA_factor + fp_2 * xB_factor + \
    (1.0 - xA_factor) * (1.0 - xB_factor) * xs

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
