import json
import random
import numpy as np
import torch as pt
from torch.optim import AdamW
import matplotlib.pyplot as plt

import itertools
from torch.utils.data import DataLoader

import wandb

from chemdm.MoleculeGraph import MoleculeGraph, batchMolecules, BatchedMoleculeGraph
from chemdm.Trajectory import Trajectory, enforceCOM
from chemdm.TransitionPathDataset import TransitionPathDataset
from chemdm.MolecularEmbeddingNetwork import MolecularEmbeddingGNN
from chemdm.TransitionPathDiffusionNetwork import TransitionPathDiffusionGNN
from chemdm.DDPMSchedule import DDPMSchedule
from chemdm.util import getGradientNorm

from typing import List, Tuple


def collate_molecules( batch : List[List[Trajectory]]
                     ) -> Tuple[BatchedMoleculeGraph, BatchedMoleculeGraph, pt.Tensor, pt.Tensor]:
    """
    Collate trajectories into batched molecule graphs and sample one random
    point along each trajectory.
    """
    trajectories = list( itertools.chain.from_iterable(batch) )

    xA_molecules = []
    xB_molecules = []
    s_list  = []
    x_list  = []
    for trajectory in trajectories:
        trajectory = enforceCOM( trajectory )

        xA = MoleculeGraph( trajectory.Z, trajectory.xA, trajectory.GA )
        xA_molecules.append( xA )
        xB = MoleculeGraph( trajectory.Z, trajectory.xB, trajectory.GB )
        xB_molecules.append( xB )

        s_idx = random.randint( 0, len(trajectory.s) - 1 )
        s_list.append( trajectory.s[s_idx] * pt.ones_like(trajectory.Z) )
        x_list.append( trajectory.x[s_idx, :, :] )

    s = pt.cat( s_list )
    x_ref = pt.cat( x_list, dim=0 )
    xA = batchMolecules( xA_molecules )
    xB = batchMolecules( xB_molecules )
    return xA, xB, s, x_ref


def main():
    with open( "./data_config.json", "r" ) as config_file:
        data_config = json.load( config_file )
    data_directory = data_config["data_folder"]
    device_name    = data_config["device"]
    setup_wandb    = data_config.get( "setup_wandb", True )
    exp_name       = data_config.get( "name", "" )

    lr = 1e-4
    n_epochs = 5000
    weight_decay = 1e-3
    B = 1
    d_cutoff = 12.0       # Angstrom

    # Diffusion
    T_diffusion = 1000       # number of diffusion timesteps
    schedule  = "cosine"

    # Network
    embedding_state_size  = 64
    embedding_message_size = 64
    n_embedding_layers = 10
    n_tp_layers = 10
    tp_message_size = 64

    if setup_wandb:
        wandb.init(
            entity  = "hannesvdc-open-numerics",
            project = "transition1x-diffusion",
            config  = {
                "learning_rate": lr, "epochs": n_epochs, "weight_decay": weight_decay,
                "T_diffusion": T_diffusion, "schedule": schedule,
                "d_cutoff": d_cutoff,
                "embedding_state_size": embedding_state_size,
                "n_embedding_layers": n_embedding_layers,
                "n_tp_layers": n_tp_layers,
            },
        )

    # datasets
    train_dataset = TransitionPathDataset( "train", data_directory )
    train_loader  = DataLoader(
        train_dataset, batch_size=B, shuffle=True,
        num_workers=8, collate_fn=collate_molecules,
        pin_memory=True, persistent_workers=True, prefetch_factor=2,
    )
    valid_dataset = TransitionPathDataset( "val", data_directory )
    valid_loader  = DataLoader(
        valid_dataset, batch_size=B, shuffle=False,
        num_workers=4, collate_fn=collate_molecules,
        pin_memory=True, persistent_workers=True, prefetch_factor=2,
    )

    # networks
    diffusion_schedule = DDPMSchedule( T=T_diffusion, schedule=schedule )
    xA_embedding = MolecularEmbeddingGNN(
        embedding_state_size, embedding_message_size, n_embedding_layers, d_cutoff
    )
    xB_embedding = MolecularEmbeddingGNN(
        embedding_state_size, embedding_message_size, n_embedding_layers, d_cutoff
    )
    network = TransitionPathDiffusionGNN(
        xA_embedding, xB_embedding, tp_message_size, n_tp_layers, d_cutoff
    )
    print( "Number of trainable parameters:",
           sum( p.numel() for p in network.parameters() if p.requires_grad ) )

    # optimizer
    optimizer = AdamW( network.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True )

    # Move to the GPU
    device = pt.device( device_name )
    dtype  = pt.float32
    network.to( device=device, dtype=dtype )
    diffusion_schedule.to( device=device )

    # loss: mse between predicted and actual x0
    def loss_fcn( x_0 : pt.Tensor, 
                 x_0_pred : pt.Tensor ) -> pt.Tensor:
        return pt.mean( (x_0 - x_0_pred) ** 2 )

    #  training helpers
    def sample_timesteps( molecule_id : pt.Tensor ) -> pt.Tensor:
        """Sample one diffusion timestep per molecule, expand to per-atom."""
        n_molecules = molecule_id.max().item() + 1
        t_per_mol = pt.randint( 0, T_diffusion, (n_molecules,), device=molecule_id.device )
        return t_per_mol[molecule_id]

    def evaluate_batch( xA : BatchedMoleculeGraph, 
                        xB : BatchedMoleculeGraph,
                        s : pt.Tensor, 
                        x_ref : pt.Tensor ) -> pt.Tensor:
        """Forward-noise x_ref, predict x_0, return loss."""
        t_int = sample_timesteps( xA.molecule_id )
        x_t, _noise = diffusion_schedule.q_sample( x_ref, t_int )

        t_normalized = t_int.float() / (T_diffusion - 1)
        x_0_pred = network( x_t, xA, xB, s, t_normalized )
        return loss_fcn( x_ref, x_0_pred.x )

    #  training loop
    train_counter, train_losses, train_grads = [], [], []
    def train( epoch : int ) -> Tuple[float, float]:
        network.train()

        # Generate random batching indices
        n_batches  = len(train_loader)
        epoch_loss = 0.0
        grad_norm  = 0.0
        for batch_idx, (xA, xB, s, x_ref) in enumerate( train_loader ):
            optimizer.zero_grad( set_to_none=True )

            xA = xA.to( device=device, dtype=dtype )
            xB = xB.to( device=device, dtype=dtype )
            s = s.to( device=device, dtype=dtype )
            x_ref = x_ref.to( device=device, dtype=dtype )

            # Evalute the loss
            loss = evaluate_batch( xA, xB, s, x_ref )
            epoch_loss += loss.item()

            # Make sure to clean up any memory
            del xA, xB, s, x_ref

            # Make an optimizer step
            loss.backward()
            grad_norm = getGradientNorm( network )
            pt.nn.utils.clip_grad_norm_( network.parameters(), 1.0 )
            optimizer.step()

            # Print some information
            epoch_idx = epoch + (batch_idx + 1.0) / n_batches
            train_counter.append( epoch_idx )
            train_losses.append( loss.item() )
            train_grads.append( grad_norm )
            if (batch_idx + 1) % 10 == 0:
                print( f"Train Epoch: {epoch} [{batch_idx+1}/{n_batches}] "
                       f"\tLoss: {loss.item():.6f} \tGrad: {grad_norm:.6f} "
                       f"\tLR: {optimizer.param_groups[-1]['lr']:.2E}", flush=True )
        return epoch_loss / n_batches, grad_norm

    valid_counter, valid_losses = [], []
    @pt.no_grad()
    def validate( epoch : int ) -> float:
        network.eval()

        # Generate random batching indices
        n_batches  = len(valid_loader)
        epoch_loss = 0.0
        for batch_idx, (xA, xB, s, x_ref) in enumerate( valid_loader ):
            xA = xA.to( device=device, dtype=dtype )
            xB = xB.to( device=device, dtype=dtype )
            s = s.to( device=device, dtype=dtype )
            x_ref = x_ref.to( device=device, dtype=dtype )

            # Evaluate the loss
            loss = evaluate_batch( xA, xB, s, x_ref )
            epoch_loss += loss.item()

            # Make sure to clean up any memory
            del xA, xB, s, x_ref

            epoch_idx = epoch + (batch_idx + 1.0) / n_batches
            valid_counter.append( epoch_idx )
            valid_losses.append( loss.item() )
            if (batch_idx + 1) % 10 == 0:
                print( f"Validation Epoch: {epoch} [{batch_idx+1}/{n_batches}] "
                       f"\tLoss: {loss.item():.6f}", flush=True )
        return epoch_loss / n_batches

    # The simplest of training loops for now.  
    best_val_loss = float("inf")
    try:
        for epoch in range( n_epochs ):
            train_loss, train_grad = train( epoch )
            print( f"Train Epoch {epoch} \tTotal Loss: {train_loss}\n", flush=True )

            valid_loss = validate( epoch )
            print( f"Validation Epoch {epoch} \tTotal Loss: {valid_loss}\n", flush=True )

            if valid_loss < best_val_loss:
                print( "Saving best model" )
                best_val_loss = valid_loss
                pt.save( network.state_dict(), f"./models/{exp_name}best_diffusion_gnn.pth" )

            if setup_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_grad": train_grad,
                    "valid_loss": valid_loss,
                    "best_val_loss": best_val_loss,
                })

            if epoch % 10 == 0:
                pt.save( network.state_dict(), f"./models/{exp_name}_diffusion_gnn.pth" )
                pt.save( optimizer.state_dict(), f"./models/{exp_name}_diffusion_optimizer.pth" )
    except KeyboardInterrupt:
        print( "Aborting training due to KeyboardInterrupt" )
    finally:
        if setup_wandb:
            wandb.finish()

    # save curves
    np.save( f"./models/{exp_name}_diffusion_train_convergence.npy",
             np.vstack( (np.array(train_counter), np.array(train_losses), np.array(train_grads)) ) )
    np.save( f"./models/{exp_name}_diffusion_valid_convergence.npy",
             np.vstack( (np.array(valid_counter), np.array(valid_losses)) ) )

    plt.semilogy( train_counter, train_losses, label="Train Loss", alpha=0.5 )
    plt.semilogy( train_counter, train_grads,  label="Gradient Norm", alpha=0.5 )
    plt.semilogy( valid_counter, valid_losses,  label="Validation Loss", alpha=0.5 )
    plt.xlabel( "Epoch" )
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
