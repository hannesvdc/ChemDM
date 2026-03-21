import torch as pt
from torch.optim import Adam
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from TrajectoryDataset import TrajectoryDataset
from chemdm.Trajectory import Trajectory
from chemdm.MolecularEmbeddingNetwork import MolecularEmbeddingNetwork
from chemdm.TransitionPathNetwork import TransitionPathNetwork
from chemdm.util import getGradientNorm

from typing import List

# Load (a reference to) the data
def collate_identity(batch):
    return batch

def main():
    B = 128
    train_dataset = TrajectoryDataset( "train"  )
    train_loader = DataLoader(
        train_dataset,
        batch_size=B,
        shuffle=True,
        collate_fn=collate_identity,
    )
    valid_dataset = TrajectoryDataset( "valid" )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=B,
        shuffle=True,
        collate_fn=collate_identity,
    )

    # Global molecular information
    d_cutoff = 5.0 # Amstrong

    # Construct the neural network architecture
    embedding_state_size = 32
    embedding_message_size = 32
    n_embedding_layers = 3
    xA_embedding = MolecularEmbeddingNetwork(embedding_state_size, d_cutoff, n_embedding_layers, embedding_message_size)
    xB_embedding = MolecularEmbeddingNetwork(embedding_state_size, d_cutoff, n_embedding_layers, embedding_message_size)
    n_tp_layers = 3
    tp_message_size = 32
    tp_network = TransitionPathNetwork( xA_embedding, xB_embedding, d_cutoff, n_tp_layers, tp_message_size )
    print( 'Number of Trainable Parameters: ', sum( [p.numel() for p in tp_network.parameters() if p.requires_grad]) )

    # Build the optimizer
    lr = 1e-4
    optimizer = Adam( tp_network.parameters(), lr, amsgrad=True )
    step_size = 5000
    scheduler = pt.optim.lr_scheduler.StepLR( optimizer, step_size=step_size, gamma=0.1)

    # A simple MSE loss as a start
    def loss_fcn( x : pt.Tensor,
                xs : pt.Tensor ) -> pt.Tensor:
        assert x.shape == xs.shape, f"`x` and `xs` must have the same (unbatched) shape, but got {x.shape} and {xs.shape}."
        return pt.mean( (x - xs)**2 ) # average over N and 3

    # Move to the GPU
    device = pt.device( "cpu" )
    dtype = pt.float32
    train_dataset.to( device=device, dtype=dtype )
    valid_dataset.to( device=device, dtype=dtype )
    tp_network.to( device=device, dtype=dtype )

    # General Batch evaluation function
    def evaluate_batch( batch_input : List[Trajectory] ) -> pt.Tensor:
        loss = 0.0
        for idx in range(len(batch_input)):
            trajectory = batch_input[idx]

            # Pick a random point on the trajectory (not batching `s` yet)
            s = trajectory.s
            s_idx = int(pt.randint(0, s.numel(), (1,)))

            # Evaluate the network
            xA = trajectory.xA
            xB = trajectory.xB
            GA = trajectory.GA
            GB = trajectory.GB
            Z = trajectory.Z
            xs = tp_network(Z, xA, xB, GA, GB, s[s_idx])

            # Compute the loss
            x = trajectory.x[s_idx,:,:]
            idx_loss = loss_fcn( x, xs )
            loss += idx_loss
        loss = loss / len(batch_input)
        return loss # type: ignore

    train_counter = []
    train_losses = []
    train_grads = []
    def train( epoch : int ) -> float :
        tp_network.train()

        # Generate random batching indices
        n_batches = len(train_loader)
        batch_idx = 0
        epoch_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad( set_to_none=True )

            # Evalute the loss
            batch_idx += 1
            loss = evaluate_batch( batch )
            epoch_loss += float( loss.item() )

            # Make an optmizer step
            loss.backward()
            grad_norm = getGradientNorm( tp_network )
            pt.nn.utils.clip_grad_norm_( tp_network.parameters(), 1.0 )
            optimizer.step()

            # Print some information
            epoch_idx = epoch + (batch_idx+1.0) / n_batches
            train_counter.append(epoch_idx)
            train_losses.append(loss.item())
            train_grads.append(grad_norm)
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{}] \tLoss: {:.6f} \t Gradient Norm {:.6f} \t Learning Rate {:.2E}'
                    .format( epoch, batch_idx, n_batches, loss.item(), grad_norm, optimizer.param_groups[-1]["lr"] ))
        return epoch_loss / n_batches

    valid_counter = []
    valid_losses = []
    @pt.no_grad()   
    def validate( epoch : int ) -> float:
        tp_network.eval()

        # Generate random batching indices
        n_batches = len(valid_loader)
        batch_idx = 0
        epoch_loss = 0.0
        for batch in valid_loader:
            batch_idx += 1

            # Evalute the loss
            loss = evaluate_batch( batch )
            epoch_loss += float( loss.item() )

            # Print some information
            epoch_idx = epoch + (batch_idx+1.0) / n_batches
            valid_counter.append(epoch_idx)
            valid_losses.append(loss.item())
            if batch_idx % 10 == 0:
                print('Vaidation Epoch: {} [{}/{}] \tLoss: {:.6f}'
                    .format( epoch, batch_idx, n_batches, loss.item() ))
        return epoch_loss / n_batches

    # The simplest of training loops for now.            
    n_epochs = 3 * step_size
    try:
        for epoch in range(n_epochs):
            train_loss = train( epoch )
            print( "Train Epoch {} \tTotal Loss: {}\n".format(epoch, train_loss) )
            valid_loss = validate( epoch )
            print( "Validation Epoch {} \tTotal Loss: {}\n".format(epoch, valid_loss) )
            scheduler.step()
    except KeyboardInterrupt:
        print('Aborting Training due to KeyboardInterrupt')

    # Plot the loss and grad norm
    plt.semilogy( train_counter, train_losses, label='Losses', alpha=0.5)
    plt.semilogy( train_counter, train_grads, label='Gradient Norms', alpha=0.5)
    plt.semilogy( valid_counter, valid_losses, label='Validation Losses', alpha=0.5)
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()