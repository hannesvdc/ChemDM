import torch as pt
from torch.nn import MSELoss
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt

from TrajectoryDataset import TrajectoryDataset
from EquivariantNetwork import EquivariantNetwork
from chemdm.util import getGradientNorm

# Set global device and dtype, except for the dataloader
dtype = pt.float32
device = pt.device("mps")

# Load the full dataset
B = 1024
train_dataset = TrajectoryDataset( "train" )
train_loader = DataLoader(train_dataset, B, shuffle=True )
test_dataset = TrajectoryDataset( "valid" )
test_loader = DataLoader( test_dataset, len(test_dataset) )

# Build the complicated FiLM Scoring Network
n_embeddings = 4
hidden_layers = [64, 64, 64]
regression_model = EquivariantNetwork( n_embeddings, hidden_layers ).to( device=device )
n_params = sum( p.numel() for p in regression_model.parameters() if p.requires_grad )
print(f"Total trainable parameters: {n_params:,}")

# Optimizer and learning rate scheduler
lr = 1e-3
n_epochs = 5_000
optimizer = optim.Adam(regression_model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=n_epochs,
    eta_min=1e-6
)

# The loss function
loss_fcn = MSELoss()

counter = []
losses = []
grad_norms = []
validation_counter = []
validation_losses = []
best_loss = float("inf")
for epoch in range(n_epochs):
    current_lr = optimizer.param_groups[0]["lr"]

    regression_model.train()
    epoch_loss = 0.0
    n_batches = 0
    for batch_idx, (s, xs, xA, xB) in enumerate(train_loader):
        optimizer.zero_grad( set_to_none=True )

        # Move to the gpu
        s = s.to( device=device, dtype=dtype )
        xs = xs.to( device=device, dtype=dtype )
        xA = xA.to( device=device, dtype=dtype )
        xB = xB.to( device=device, dtype=dtype )

        # Forward and compute the loss
        xs_predicted = regression_model( xA, xB, s )
        loss = loss_fcn( xs_predicted, xs )

        # Update the network weights.
        loss.backward()
        grad_norm = getGradientNorm( regression_model )
        optimizer.step()

        # Bookkeeping
        epoch_loss += float( loss.detach().item() )
        n_batches += 1
        counter.append((1.0*batch_idx)/len(train_loader) + epoch)
        losses.append(loss.item())
        grad_norms.append(grad_norm)
    scheduler.step()
    epoch_loss /= n_batches
    print('Train Epoch: {} \tLoss: {:.6f} \t Gradient Norm {:.6f} \t Learning Rate {:.2E}'.format(  epoch, epoch_loss, grad_norm, current_lr ))

    regression_model.eval()
    with pt.no_grad():
        for batch_idx, (s, xs, xA, xB) in enumerate( test_loader ):
            # Move to the gpu
            s = s.to( device=device, dtype=dtype )
            xs = xs.to( device=device, dtype=dtype )
            xA = xA.to( device=device, dtype=dtype )
            xB = xB.to( device=device, dtype=dtype )

            # Forward and compute the loss
            xs_predicted = regression_model( xA, xB, s )
            validation_loss = loss_fcn( xs_predicted, xs )

            # Bookkeeping
            validation_counter.append((1.0*batch_idx)/len(test_loader) + epoch)
            validation_losses.append( validation_loss.item() )

        if validation_loss.item() < best_loss:
            print('Storing best regression model')
            pt.save( regression_model.state_dict(), "./models/regressor.pth")
            best_loss = validation_loss.item()

    print('Validation Loss {:.6f}'.format( validation_loss.item() ))

# Plot the loss and grad norm
plt.semilogy(counter, losses, label='Losses', alpha=0.5)
plt.semilogy(counter, grad_norms, label='Gradient Norms', alpha=0.5)
plt.semilogy(validation_counter, validation_losses, label='Validation Losses', alpha=0.5)
plt.xlabel('Epoch')
plt.legend()
plt.show()