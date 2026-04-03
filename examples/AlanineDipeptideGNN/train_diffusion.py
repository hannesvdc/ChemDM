import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import json
import random
import numpy as np
import torch as pt
from torch.optim import AdamW
import matplotlib.pyplot as plt

from pathlib import Path
from torch.utils.data import DataLoader

from chemdm.MoleculeGraph import BatchedMoleculeGraph, MoleculeGraph, batchMolecules
from chemdm.Trajectory import Trajectory
from chemdm.MolecularEmbeddingNetwork import MolecularEmbeddingGNN
from chemdm.TransitionPathDiffusionNetwork import TransitionPathDiffusionGNN
from chemdm.DDPMSchedule import DDPMSchedule
from chemdm.util import getGradientNorm
from TrajectoryDataset import TrajectoryDataset

from typing import List, Tuple


def collate_molecules( batch : List[Trajectory]
                     ) -> Tuple[BatchedMoleculeGraph, BatchedMoleculeGraph, pt.Tensor, pt.Tensor]:
    """
    Collate trajectories into batched molecule graphs and sample one random
    point along each trajectory.
    """
    xA_molecules = []
    xB_molecules = []
    s_list = []
    x_list = []
    for trajectory in batch:
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


def generateTrainValidMask( n : int, train_fraction : float ) -> pt.Tensor:
    perm = pt.randperm( n )
    n_train = int( train_fraction * n )
    train_mask = pt.zeros( n, dtype=pt.bool )
    train_mask[perm[:n_train]] = True
    return train_mask


def main():
    with open( "./data_config.json", "r" ) as config_file:
        data_config = json.load( config_file )
    data_directory = data_config["data_folder"]
    device_name = data_config["device"]
    exp_name = data_config.get( "name", "" )

    # hyperparameteres
    lr = 1e-3
    n_epochs = 4000
    weight_decay = 1e-3
    B = 128

    # Note: alanine dipeptide coordinates from OpenMM are in nm (not Angstrom).
    # The cutoff is therefore also in nm. The molecule is ~1 nm across.
    d_cutoff = 1.0 # nm

    # Diffusion
    T_diffusion = 1000
    schedule = "cosine"

    # Network
    embedding_state_size = 32
    embedding_message_size = 32
    n_embedding_layers = 5
    n_tp_layers = 5
    tp_message_size = 32

    # Datasets
    train_dataset = TrajectoryDataset( outdir=Path(data_directory) )
    valid_dataset = TrajectoryDataset( outdir=Path(data_directory) )
    train_mask = generateTrainValidMask( len(train_dataset), 0.9 )
    valid_mask = ~train_mask
    train_dataset.apply_mask( train_mask )
    valid_dataset.apply_mask( valid_mask )
    train_loader = DataLoader(
        train_dataset, batch_size=B, shuffle=True,
        collate_fn=collate_molecules,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=B, shuffle=False,
        collate_fn=collate_molecules,
    )

    # Networks
    diffusion_schedule = DDPMSchedule( T=T_diffusion, schedule=schedule )
    xA_embedding = MolecularEmbeddingGNN( embedding_state_size, embedding_message_size, n_embedding_layers, d_cutoff )
    xB_embedding = MolecularEmbeddingGNN( embedding_state_size, embedding_message_size, n_embedding_layers, d_cutoff )
    network = TransitionPathDiffusionGNN( xA_embedding, xB_embedding, tp_message_size, n_tp_layers, d_cutoff )
    print( "Number of trainable parameters:", sum( p.numel() for p in network.parameters() if p.requires_grad ) )

    # Optimizer
    optimizer = AdamW( network.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True )
    step_size = 1000
    scheduler = pt.optim.lr_scheduler.StepLR( optimizer, step_size=step_size, gamma=0.1 )

    device = pt.device( device_name )
    dtype  = pt.float32
    network.to( device=device, dtype=dtype )
    diffusion_schedule.to( device=device )

    # loss
    def loss_fcn( x_0 : pt.Tensor, 
                  x_0_pred : pt.Tensor,
                ) -> pt.Tensor:
        return pt.mean( (x_0 - x_0_pred) ** 2 )

    # training helpers
    def sample_timesteps( molecule_id : pt.Tensor ) -> pt.Tensor:
        """Sample one diffusion timestep per molecule, expand to per-atom."""
        n_molecules = molecule_id.max().item() + 1
        t_per_mol = pt.randint( 0, T_diffusion, (n_molecules,), device=molecule_id.device )
        return t_per_mol[molecule_id]

    def evaluate_batch( xA, xB, s, x_ref ):
        """Forward-noise x_ref, predict x_0, return loss."""
        t_int = sample_timesteps( xA.molecule_id )
        x_t, _noise = diffusion_schedule.q_sample( x_ref, t_int )

        t_normalized = t_int.float() / (T_diffusion - 1)
        x_0_pred = network( x_t, xA, xB, s, t_normalized )
        return loss_fcn( x_ref, x_0_pred.x )

    # Simple Training loop
    train_counter, train_losses, train_grads = [], [], []
    def train( epoch : int ) -> Tuple[float, float]:
        network.train()

        n_batches  = len(train_loader)
        epoch_loss = 0.0
        grad_norm  = 0.0
        for batch_idx, (xA, xB, s, x_ref) in enumerate( train_loader ):
            optimizer.zero_grad( set_to_none=True )

            # Move to the GPU
            xA = xA.to( device=device, dtype=dtype )
            xB = xB.to( device=device, dtype=dtype )
            s = s.to( device=device, dtype=dtype )
            x_ref = x_ref.to( device=device, dtype=dtype )

            # Evaluate the loss
            loss = evaluate_batch( xA, xB, s, x_ref )
            epoch_loss += loss.item()

            # Free up memory
            del xA, xB, s, x_ref

            # Gradient step
            loss.backward()
            grad_norm = getGradientNorm( network )
            pt.nn.utils.clip_grad_norm_( network.parameters(), 1.0 )
            optimizer.step()

            # Logging
            epoch_idx = epoch + (batch_idx + 1.0) / n_batches
            train_counter.append( epoch_idx )
            train_losses.append( loss.item() )
            train_grads.append( grad_norm )
            if (batch_idx + 1) % 100 == 0:
                print( f"Train Epoch: {epoch} [{batch_idx+1}/{n_batches}] "
                       f"\tLoss: {loss.item():.6f} \tGrad: {grad_norm:.6f} "
                       f"\tLR: {optimizer.param_groups[-1]['lr']:.2E}", flush=True )
        return epoch_loss / n_batches, grad_norm

    valid_counter, valid_losses = [], []

    @pt.no_grad()
    def validate( epoch : int ) -> float:
        network.eval()

        n_batches  = len(valid_loader)
        epoch_loss = 0.0
        for batch_idx, (xA, xB, s, x_ref) in enumerate( valid_loader ):
            xA = xA.to( device=device, dtype=dtype )
            xB = xB.to( device=device, dtype=dtype )
            s = s.to( device=device, dtype=dtype )
            x_ref = x_ref.to( device=device, dtype=dtype )

            loss = evaluate_batch( xA, xB, s, x_ref )
            epoch_loss += loss.item()

            del xA, xB, s, x_ref

            epoch_idx = epoch + (batch_idx + 1.0) / n_batches
            valid_counter.append( epoch_idx )
            valid_losses.append( loss.item() )
            if (batch_idx + 1) % 100 == 0:
                print( f"Validation Epoch: {epoch} [{batch_idx+1}/{n_batches}] "
                       f"\tLoss: {loss.item():.6f}", flush=True )
        return epoch_loss / n_batches

    # Main training loop
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

            scheduler.step()

            if epoch % 10 == 0:
                pt.save( network.state_dict(),  f"./models/{exp_name}_diffusion_gnn.pth" )
                pt.save( optimizer.state_dict(), f"./models/{exp_name}_diffusion_optimizer.pth" )
    except KeyboardInterrupt:
        print( "Aborting training due to KeyboardInterrupt" )

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
