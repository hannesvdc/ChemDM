import numpy as np
import torch as pt
from torch.optim import AdamW
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

import wandb
from pathlib import Path
import json
import argparse

from chemdm.TransitionPathDataset import TransitionPathDataset
from chemdm.MoleculeGraph import BatchedMoleculeGraph
from chemdm.MolecularEmbeddingNetwork import MolecularEmbeddingGNN
from chemdm.TransitionPathE3NN import TransitionPathE3NN
from chemdm.util import getGradientNorm, perCoordinateRMSE, collate_molecules

from typing import Tuple

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        dest='name',
        help="Experiment name. Results are stored in experiments/<name>/.",
    )
    return parser.parse_args()

def make_experiment_dir(exp_name: str, root: str = "./experiments") -> Path:
    exp_dir = Path(root) / exp_name

    print('Storing results in', exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir

def main( exp_name : str ):
    with open( './data_config.json', "r" ) as f:
        data_config = json.load( f )
    data_directory = data_config["data_folder"]
    device_name = data_config["device"]
    setup_wandb = data_config.get("setup_wandb", True)

    B = 1
    train_dataset = TransitionPathDataset( "train", data_directory )
    pin_memory = device_name.startswith("cuda")
    train_loader = DataLoader(
        train_dataset,
        batch_size=B,
        shuffle=True,
        num_workers=8,
        collate_fn=collate_molecules,
        pin_memory=pin_memory,
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
        pin_memory=pin_memory,
        persistent_workers=True,
        prefetch_factor=2,
    )

    # Global molecular information
    d_cutoff = 6.0      # I would start smaller for e3nn
    n_rbf = 10

    # Endpoint embedding networks
    embedding_state_size = 64
    embedding_message_size = 64
    n_embedding_layers = 5
    xA_embedding = MolecularEmbeddingGNN( embedding_state_size, embedding_message_size, n_embedding_layers, d_cutoff )
    xB_embedding = MolecularEmbeddingGNN( embedding_state_size, embedding_message_size, n_embedding_layers, d_cutoff )

    # E3NN transition-path network
    irreps_node_str = "64x0e + 16x1o + 8x1e"
    n_e3_layers = 10
    tp_network = TransitionPathE3NN(
        xA_embedding_network=xA_embedding,
        xB_embedding_network=xB_embedding,
        irreps_node_str=irreps_node_str,
        n_layers=n_e3_layers,
        d_cutoff=d_cutoff,
        n_freq=8,
        n_rbf=n_rbf,   # include this if your constructor has it
    )
    n_params = sum(p.numel() for p in tp_network.parameters() if p.requires_grad)
    print( "Number of Trainable Parameters: ", n_params )

    # Build the optimizer
    lr = 1e-3
    n_epochs = 5000
    weight_decay = 1e-4
    optimizer = AdamW( tp_network.parameters(), lr, weight_decay=weight_decay, amsgrad=True )

    # Everything has been set up, now log the config and initialize weights&biases
    experiment_config = {
        "architecture": "TransitionPathE3GNN",
        "experiment_name": exp_name,
        "data_directory": data_directory,
        "device": device_name,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "batch_size": B,
        "epochs": n_epochs,
        "irreps_node": irreps_node_str,
        "n_e3_layers": n_e3_layers,
        "d_cutoff": d_cutoff,
        "n_rbf": n_rbf,
        "embedding_state_size": embedding_state_size,
        "embedding_message_size": embedding_message_size,
        "n_embedding_layers": n_embedding_layers,
        "n_trainable_parameters": n_params,
    }
    exp_dir = make_experiment_dir( exp_name )
    with open(exp_dir / "config.json", "w") as f:
        json.dump(experiment_config, f, indent=2)
    if setup_wandb:
        run = wandb.init(
            # Set the wandb entity where your project will be logged (generally your team name).
            entity="hannesvdc-open-numerics",
            # Set the wandb project where this run will be logged.
            project="transition1x_e3nn",
            # Set the name
            name=exp_name,
            # Track hyperparameters and run metadata.
            config=experiment_config
        )

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
        loss = loss_fcn( x_ref, xs.x )
        return loss

    @pt.no_grad()
    def evaluate_rmse( loader, max_batches : int = 20 ) -> float:
        """Evaluate per-coordinate RMSE over a limited number of batches."""
        tp_network.eval()
        rmse_sum = 0.0
        n_batches = min( max_batches, len(loader) )
        for batch_idx, (xA, xB, s, x_ref) in enumerate( loader ):
            if batch_idx >= max_batches:
                break
            xA    = xA.to( device=device, dtype=dtype )
            xB    = xB.to( device=device, dtype=dtype )
            s     = s.to( device=device, dtype=dtype )
            x_ref = x_ref.to( device=device, dtype=dtype )

            xs = tp_network( xA, xB, s )
            rmse_sum += perCoordinateRMSE( x_ref, xs.x )
        return rmse_sum / n_batches

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
            if (batch_idx+1) % 1 == 0:
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

            # Print some information
            epoch_idx = epoch + (batch_idx+1.0) / n_batches
            valid_counter.append(epoch_idx)
            valid_losses.append(loss.item())
            if (batch_idx+1) % 10 == 0:
                print('Validation Epoch: {} [{}/{}] \tLoss: {:.6f}'
                    .format( epoch, batch_idx+1, n_batches, loss.item() ), flush=True)
        return epoch_loss / n_batches

    # The simplest of training loops for now.            
    best_val_loss = float("inf")
    try:
        for epoch in range(n_epochs):
            train_loss, train_grad = train( epoch )
            print( "Train Epoch {} \tTotal Loss: {}\n".format(epoch, train_loss), flush=True )
            valid_loss = validate( epoch )
            train_rmse = evaluate_rmse( train_loader )
            valid_rmse = evaluate_rmse( valid_loader )
            print( "Validation Epoch {} \tTotal Loss: {} \tTrain RMSE: {:.6f} \tValid RMSE: {:.6f}\n"
                   .format(epoch, valid_loss, train_rmse, valid_rmse), flush=True )
            if valid_loss < best_val_loss:
                print('Saving best model')
                best_val_loss = valid_loss
                pt.save( tp_network.state_dict(), exp_dir / "best_gnn.pth" )

            # Log to weights & biases
            if setup_wandb:
                run.log({"epoch": epoch, "train_loss": train_loss,
                         "train_grad": train_grad, "valid_loss" : valid_loss,
                         "best_val_loss" : best_val_loss,
                         "train_rmse": train_rmse, "valid_rmse": valid_rmse})

            if epoch % 10 == 0:
                pt.save( tp_network.state_dict(), exp_dir / "gnn.pth" )
                pt.save( optimizer.state_dict(), exp_dir / "optimizer.pth" )
    except KeyboardInterrupt:
        print('Aborting Training due to KeyboardInterrupt')
    finally:
        if setup_wandb:
            run.finish()

    # Store training convergence
    np.save( exp_dir / "train_convergence.npy", np.vstack( (np.array(train_counter), np.array(train_losses), np.array(train_grads)) ) )
    np.save( exp_dir / "valid_convergence.npy", np.vstack( (np.array(valid_counter), np.array(valid_losses) ) ) )

    # Plot the loss and grad norm
    plt.semilogy( train_counter, train_losses, label='Losses', alpha=0.5)
    plt.semilogy( train_counter, train_grads, label='Gradient Norms', alpha=0.5)
    plt.semilogy( valid_counter, valid_losses, label='Validation Losses', alpha=0.5)
    plt.xlabel('Epoch')
    plt.legend()
    plt.tight_layout()
    plt.savefig(exp_dir / "convergence.png", dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == '__main__':
    args = parse_args( )
    main( args.name )
