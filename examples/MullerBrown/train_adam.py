import os
import torch as pt
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt

from chemdm.TrajectoryDataset import TrajectoryDataset
from chemdm.ScoreNetwork import ScoreNetwork
from chemdm.ScoreLoss import ScoreLoss
from chemdm.util import getGradientNorm

# Set global device and dtype, except for the dataloader
dtype = pt.float32
device = pt.device("mps")

# Load the full dataset
B = 1024
data_folder = os.path.abspath( './data' )
train_dataset = TrajectoryDataset( data_folder, "train" )
train_loader = DataLoader(train_dataset, B, shuffle=True)
test_dataset = TrajectoryDataset( data_folder, "valid" )
test_loader = DataLoader(test_dataset, len(test_dataset))

# Build the complicated FiLM Scoring Network
n_grid = 100
n_embeddings = 8
hidden_layers = [64, 64, 64]
film_hidden_layers = [64, 64, 64]
score_model = ScoreNetwork( n_embeddings, hidden_layers, film_hidden_layers ).to( device=device )
n_params = sum( p.numel() for p in score_model.parameters() if p.requires_grad )
print(f"Total trainable parameters: {n_params:,}")

# Optimizer and learning rate scheduler
lr = 2e-4
n_epochs = 5_000
optimizer = optim.Adam(score_model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=n_epochs,
    eta_min=1e-6
)

# The score loss function
loss_fcn = ScoreLoss( device, dtype )

counter = []
losses = []
grad_norms = []
test_counter = []
test_losses = []

best_loss = float("inf")
for epoch in range(n_epochs):
    current_lr = optimizer.param_groups[0]["lr"]

    score_model.train()
    for batch_idx, (s, u0, xA, xB) in enumerate(train_loader):
        optimizer.zero_grad( set_to_none=True )

        loss = loss_fcn( u0, xA, xB, s )
        loss.backward()
        grad_norm = getGradientNorm( score_model )
        optimizer.step()

        counter.append((1.0*batch_idx)/len(train_loader) + epoch)
        losses.append(loss.item())
        grad_norms.append(grad_norm)
    scheduler.step()
    print('Train Epoch: {} \tLoss: {:.6f} \t Gradient Norm {:.6f} \t Learning Rate {:.2E}'.format(  epoch, loss.item(), grad_norm, current_lr ))

    score_model.eval()
    with pt.no_grad():
        for batch_idx, (s, u0, xA, xB) in enumerate(train_loader):
            test_loss = loss_fcn( u0, xA, xB, s )
            test_counter.append((1.0*batch_idx)/len(test_loader) + epoch)
            test_losses.append(test_loss.item())

        if test_loss.item() < best_loss:
            pt.save(score_model.state_dict(), "./models/porous_score_model.pth")
            best_loss = test_loss.item()

    print('Test Loss {:.6f}'.format( test_loss.item() ))

# Save the final network weights on file    
pt.save(score_model.state_dict(), "./models/porous_score_model.pth")

# Plot the loss and grad norm
plt.semilogy(counter, losses, label='Losses', alpha=0.5)
plt.semilogy(counter, grad_norms, label='Gradient Norms', alpha=0.5)
plt.semilogy(test_counter, test_losses, label='Test Losses', alpha=0.5)
plt.xlabel('Epoch')
plt.legend()
plt.show()