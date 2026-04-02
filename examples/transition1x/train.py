import json
import random
import numpy as np
import torch as pt
from torch.optim import Adam, AdamW
import matplotlib.pyplot as plt

import itertools
from torch.utils.data import DataLoader

import wandb

from chemdm.MoleculeGraph import MoleculeGraph, batchMolecules
from chemdm.TransitionPathDataset import TransitionPathDataset, Trajectory
from chemdm.MoleculeGraph import BatchedMoleculeGraph
from chemdm.MolecularEmbeddingNetwork import MolecularEmbeddingGNN
from chemdm.TransitionPathNetwork import TransitionPathGNN
from chemdm.util import getGradientNorm

from typing import List, Tuple

def collate_molecules(batch : List[List[Trajectory]]
                     ) -> Tuple[BatchedMoleculeGraph, BatchedMoleculeGraph, pt.Tensor, pt.Tensor]:
    """
    Batching molecules and random points on the path.
    """
    trajectories = list(itertools.chain.from_iterable( batch )) # squash the nested list of trajectories

    # Sample random points on the trajectory for each molecule
    xA_molecules = []
    xB_molecules = []
    s_list = []
    x_list = []
    for trajectory in trajectories:
        xA = MoleculeGraph( trajectory.Z, trajectory.xA, trajectory.GA )
        xA_molecules.append( xA )
        xB = MoleculeGraph( trajectory.Z, trajectory.xB, trajectory.GB )
        xB_molecules.append( xB )

        s_idx = random.randint( 0, len(trajectory.s)-1 )
        s_list.append( trajectory.s[s_idx] * pt.ones_like(trajectory.Z) )
        x_list.append( trajectory.x[s_idx,:,:] )
    s = pt.cat( s_list ) # (N_atoms,)
    x_ref = pt.cat( x_list, dim=0 ) # (N_atoms,3)

    xA = batchMolecules( xA_molecules )
    xB = batchMolecules( xB_molecules )
    return xA, xB, s, x_ref

def main():
    with open( './data_config.json', "r" ) as f:
        data_config = json.load( f )
    data_directory = data_config["data_folder"]
    device_name = data_config["device"]
    setup_wandb = data_config.get("setup_wandb", True)
    exp_name = data_config.get( "name", "")

    # Start a new wandb run to track this script.
    lr = 1e-4
    n_epochs = 5000
    if setup_wandb:
        run = wandb.init(
            # Set the wandb entity where your project will be logged (generally your team name).
            entity="hannesvdc-open-numerics",
            # Set the wandb project where this run will be logged.
            project="transition1x",
            # Track hyperparameters and run metadata.
            config={
                "learning_rate": lr,
                "architecture": "GNN",
                "dataset": "transition1x",
                "epochs": n_epochs,
            },
        )

    B = 8
    train_dataset = TransitionPathDataset( "train", data_directory )
    train_loader = DataLoader(
        train_dataset,
        batch_size=B,
        shuffle=True,
        num_workers=8,
        collate_fn=collate_molecules,
        pin_memory=True,
        persistent_workers=True, 
        prefetch_factor=2,
    )
    valid_dataset = TransitionPathDataset( "val", data_directory )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=B,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_molecules,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    # Global molecular information
    d_cutoff = 12.0 # Angstrom

    # Construct the neural network architecture
    embedding_state_size = 64
    embedding_message_size = 64
    n_embedding_layers = 10
    xA_embedding = MolecularEmbeddingGNN(embedding_state_size, embedding_message_size, n_embedding_layers, d_cutoff)
    xB_embedding = MolecularEmbeddingGNN(embedding_state_size, embedding_message_size, n_embedding_layers, d_cutoff)
    n_tp_layers = 10
    tp_message_size = 64
    tp_network = TransitionPathGNN( xA_embedding, xB_embedding, tp_message_size, n_tp_layers, d_cutoff )
    print( 'Number of Trainable Parameters: ', sum( [p.numel() for p in tp_network.parameters() if p.requires_grad]) )

    # Build the optimizer
    weight_decay = 1e-2
    optimizer = AdamW( tp_network.parameters(), lr, weight_decay=weight_decay, amsgrad=True )

    # A simple MSE loss as a start
    def loss_fcn( x : pt.Tensor,
                xs : pt.Tensor ) -> pt.Tensor:
        assert x.shape == xs.shape, f"`x` and `xs` must have the same (unbatched) shape, but got {x.shape} and {xs.shape}."
        return pt.mean( (x - xs)**2 ) # average over N and 3

    # Move to the GPU
    device = pt.device( device_name )
    dtype = pt.float32
    tp_network.to( device=device, dtype=dtype )

    # General Batch evaluation function
    def evaluate_batch( xA : BatchedMoleculeGraph,
                        xB : BatchedMoleculeGraph,
                        s : pt.Tensor,
                        x_ref : pt.Tensor ) -> pt.Tensor:
        xs = tp_network( xA, xB, s )
        loss = loss_fcn( x_ref, xs )
        return loss

    train_counter = []
    train_losses = []
    train_grads = []
    def train( epoch : int ) -> Tuple[float,float] :
        tp_network.train()

        # Generate random batching indices
        n_batches = len(train_loader)
        epoch_loss = 0.0
        for batch_idx, (xA, xB, s, x_ref) in enumerate( train_loader ):
            optimizer.zero_grad( set_to_none=True )

            xA = xA.to( device=device, dtype=dtype )
            xB = xB.to( device=device, dtype=dtype )
            s = s.to( device=device, dtype=dtype )
            x_ref = x_ref.to( device=device, dtype=dtype )

            # Evalute the loss
            loss = evaluate_batch( xA, xB, s, x_ref )
            epoch_loss += float( loss.item() )

            # Make sure to clean up any memory
            del xA, xB, s, x_ref

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
            if (batch_idx+1) % 10 == 0:
                print('Train Epoch: {} [{}/{}] \tLoss: {:.6f} \t Gradient Norm {:.6f} \t Learning Rate {:.2E}'
                    .format( epoch, batch_idx+1, n_batches, loss.item(), grad_norm, optimizer.param_groups[-1]["lr"] ), flush=True)
        return epoch_loss / n_batches, grad_norm

    valid_counter = []
    valid_losses = []
    @pt.no_grad()   
    def validate( epoch : int ) -> float:
        tp_network.eval()

        # Generate random batching indices
        n_batches = len(valid_loader)
        epoch_loss = 0.0
        for batch_idx, (xA, xB, s, x_ref) in enumerate( valid_loader ):

            xA = xA.to( device=device, dtype=dtype )
            xB = xB.to( device=device, dtype=dtype  )
            s = s.to( device=device, dtype=dtype )
            x_ref = x_ref.to( device=device, dtype=dtype )

            # Evalute the loss
            loss = evaluate_batch( xA, xB, s, x_ref )
            epoch_loss += float( loss.item() )

            # Make sure to clean up any memory
            del xA, xB, s, x_ref

            # Print some information
            epoch_idx = epoch + (batch_idx+1.0) / n_batches
            valid_counter.append(epoch_idx)
            valid_losses.append(loss.item())
            if (batch_idx+1) % 10 == 0:
                print('Vaidation Epoch: {} [{}/{}] \tLoss: {:.6f}'
                    .format( epoch, batch_idx+1, n_batches, loss.item() ), flush=True)
        return epoch_loss / n_batches

    # The simplest of training loops for now.            
    best_val_loss = float("inf")
    try:
        for epoch in range(n_epochs):
            train_loss, train_grad = train( epoch )
            print( "Train Epoch {} \tTotal Loss: {}\n".format(epoch, train_loss), flush=True )
            valid_loss = validate( epoch )
            print( "Validation Epoch {} \tTotal Loss: {}\n".format(epoch, valid_loss), flush=True )
            if valid_loss < best_val_loss:
                print('Saving best model')
                best_val_loss = valid_loss
                pt.save( tp_network.state_dict(), f"./models/{exp_name}best_gnn.pth" )

            # Log to weights & biases
            if setup_wandb:
                run.log({"epoch": epoch, "train_loss": train_loss, 
                         "train_grad": train_grad, "valid_loss" : valid_loss, 
                         "best_val_loss" : best_val_loss})

            if epoch % 10 == 0:
                pt.save( tp_network.state_dict(), f"./models/{exp_name}_gnn.pth" )
                pt.save( optimizer.state_dict(), f"./models/{exp_name}_optimizer.pth" )
    except KeyboardInterrupt:
        print('Aborting Training due to KeyboardInterrupt')
    finally:
        if setup_wandb:
            run.finish()

    # Store training convergence
    np.save( f"./models/{exp_name}_train_convergence.npy", np.vstack( (np.array(train_counter), np.array(train_losses), np.array(train_grads)) ) )
    np.save( f"./models/{exp_name}_valid_convergence.npy", np.vstack( (np.array(valid_counter), np.array(valid_losses) ) ) )

    # Plot the loss and grad norm
    plt.semilogy( train_counter, train_losses, label='Losses', alpha=0.5)
    plt.semilogy( train_counter, train_grads, label='Gradient Norms', alpha=0.5)
    plt.semilogy( valid_counter, valid_losses, label='Validation Losses', alpha=0.5)
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
