import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import random
import numpy as np
import torch as pt
from torch.optim import AdamW
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from chemdm.MoleculeGraph import BatchedMoleculeGraph, MoleculeGraph, batchMolecules
from TrajectoryDataset import TrajectoryDataset
from chemdm.Trajectory import Trajectory, alignToReactant
from chemdm.MolecularEmbeddingNetwork import MolecularEmbeddingGNN
from chemdm.FiLMTransitionPathGNN import FiLMTransitionPathGNN
from chemdm.util import getGradientNorm, perCoordinateRMSE

from typing import List, Tuple


def collate_molecules(batch : List[Trajectory]
                     ) -> Tuple[BatchedMoleculeGraph, BatchedMoleculeGraph, pt.Tensor, pt.Tensor]:
    xA_molecules = []
    xB_molecules = []
    s_list = []
    x_list = []
    for trajectory in batch:
        trajectory = alignToReactant( trajectory )
        n_s = len( trajectory.s )
        
        xA_molecules.extend( [MoleculeGraph( trajectory.Z, trajectory.xA, trajectory.GA ) for _ in range( n_s )])
        xB_molecules.extend( [MoleculeGraph( trajectory.Z, trajectory.xB, trajectory.GB ) for _ in range( n_s )])

        # xA = MoleculeGraph( trajectory.Z, trajectory.xA, trajectory.GA )
        # xA_molecules.append( xA )
        # xB = MoleculeGraph( trajectory.Z, trajectory.xB, trajectory.GB )
        # xB_molecules.append( xB )

        # s_idx = random.randint( 0, len(trajectory.s)-1 )
        s_vec = trajectory.s[:,None] * pt.ones( (1, len(trajectory.Z)) )
        s_list.append( pt.flatten(s_vec) )
        # s_list.append( trajectory.s[s_idx] * pt.ones_like(trajectory.Z) )
        # x_list.append( trajectory.x[s_idx,:,:] )
        x_list.append( pt.flatten(trajectory.x, start_dim=0, end_dim=1 ) ) # (n_s * natoms, 3)
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
    B = 128
    train_dataset = TrajectoryDataset()
    valid_dataset = TrajectoryDataset()
    train_mask = generateTrainValidMask( len(train_dataset), 0.9 )
    valid_mask = ~train_mask
    train_dataset.apply_mask( train_mask )
    valid_dataset.apply_mask( valid_mask )
    train_loader = DataLoader( train_dataset, batch_size=B, shuffle=True, collate_fn=collate_molecules )
    valid_loader = DataLoader( valid_dataset, batch_size=B, shuffle=True, collate_fn=collate_molecules, )

    device = pt.device( "mps" )
    dtype = pt.float32

    # Note: alanine dipeptide coordinates from OpenMM are in nm (not Angstrom).
    d_cutoff = 1.0  # nm

    # Network architecture
    embedding_state_size = 64
    embedding_message_size = 64
    n_embedding_layers = 7 # increased over deterministic model because fewer atoms can reach each other.
    n_tp_layers = 7
    tp_message_size = 64

    xA_embedding = MolecularEmbeddingGNN( embedding_state_size, embedding_message_size, n_embedding_layers, d_cutoff )
    xB_embedding = MolecularEmbeddingGNN( embedding_state_size, embedding_message_size, n_embedding_layers, d_cutoff )
    tp_network = FiLMTransitionPathGNN( xA_embedding, xB_embedding, tp_message_size, n_tp_layers, d_cutoff )
    print( 'Number of Trainable Parameters: ', sum( p.numel() for p in tp_network.parameters() if p.requires_grad ) )

    tp_network.to( dtype=dtype, device=device )

    # Optimizer
    lr = 1e-3
    weight_decay = 1e-3
    optimizer = AdamW( tp_network.parameters(), lr, weight_decay=weight_decay, amsgrad=True )
    step_size = 1000
    scheduler = pt.optim.lr_scheduler.StepLR( optimizer, step_size=step_size, gamma=0.1 )

    def loss_fcn( x : pt.Tensor, xs : pt.Tensor ) -> pt.Tensor:
        assert x.shape == xs.shape
        return pt.mean( (x - xs)**2 )

    def evaluate_batch( xA : BatchedMoleculeGraph,
                        xB : BatchedMoleculeGraph,
                        s : pt.Tensor,
                        x_ref : pt.Tensor ) -> pt.Tensor:
        xs = tp_network( xA, xB, s )
        return loss_fcn( x_ref, xs.x )

    @pt.no_grad()
    def evaluate_rmse( loader, max_batches : int = 20 ) -> float:
        tp_network.eval()
        rmse_sum = 0.0
        n_batches = min( max_batches, len(loader) )
        for batch_idx, (xA, xB, s, x_ref) in enumerate( loader ):
            if batch_idx >= max_batches:
                break
            xA = xA.to( device=device, dtype=dtype )
            xB = xB.to( device=device, dtype=dtype )
            s = s.to( device=device, dtype=dtype )
            x_ref = x_ref.to( device=device, dtype=dtype )

            xs = tp_network( xA, xB, s )
            rmse_sum += perCoordinateRMSE( x_ref, xs.x )
        return rmse_sum / n_batches

    train_counter, train_losses, train_grads = [], [], []
    def train( epoch : int ) -> float:
        tp_network.train()
        n_batches = len(train_loader)
        epoch_loss = 0.0
        for batch_idx, (xA, xB, s, x_ref) in enumerate( train_loader ):
            optimizer.zero_grad( set_to_none=True )

            xA = xA.to( device=device, dtype=dtype )
            xB = xB.to( device=device, dtype=dtype )
            s = s.to( device=device, dtype=dtype )
            x_ref = x_ref.to( device=device, dtype=dtype )

            loss = evaluate_batch( xA, xB, s, x_ref )
            epoch_loss += float( loss.item() )

            loss.backward()
            grad_norm = getGradientNorm( tp_network )
            pt.nn.utils.clip_grad_norm_( tp_network.parameters(), 1.0 )
            optimizer.step()

            epoch_idx = epoch + (batch_idx + 1.0) / n_batches
            train_counter.append( epoch_idx )
            train_losses.append( loss.item() )
            train_grads.append( grad_norm )
            if (batch_idx + 1) % 100 == 0:
                print( f'Train Epoch: {epoch} [{batch_idx+1}/{n_batches}] '
                       f'\tLoss: {loss.item():.6f} \tGrad: {grad_norm:.6f} '
                       f'\tLR: {optimizer.param_groups[-1]["lr"]:.2E}' )
        return epoch_loss / n_batches

    valid_counter, valid_losses = [], []
    @pt.no_grad()
    def validate( epoch : int ) -> float:
        tp_network.eval()
        n_batches = len(valid_loader)
        epoch_loss = 0.0
        for batch_idx, (xA, xB, s, x_ref) in enumerate( valid_loader ):
            xA = xA.to( device=device, dtype=dtype )
            xB = xB.to( device=device, dtype=dtype )
            s = s.to( device=device, dtype=dtype )
            x_ref = x_ref.to( device=device, dtype=dtype )

            loss = evaluate_batch( xA, xB, s, x_ref )
            epoch_loss += float( loss.item() )

            epoch_idx = epoch + (batch_idx + 1.0) / n_batches
            valid_counter.append( epoch_idx )
            valid_losses.append( loss.item() )
            if (batch_idx + 1) % 100 == 0:
                print( f'Validation Epoch: {epoch} [{batch_idx+1}/{n_batches}] '
                       f'\tLoss: {loss.item():.6f}' )
        return epoch_loss / n_batches

    n_epochs = 4 * step_size
    best_val_loss = pt.inf
    try:
        for epoch in range( n_epochs ):
            train_loss = train( epoch )
            print( f"Train Epoch {epoch} \tTotal Loss: {train_loss}\n" )
            valid_loss = validate( epoch )
            train_rmse = evaluate_rmse( train_loader )
            valid_rmse = evaluate_rmse( valid_loader )
            print( f"Validation Epoch {epoch} \tTotal Loss: {valid_loss} "
                   f"\tTrain RMSE: {train_rmse:.6f} \tValid RMSE: {valid_rmse:.6f}\n" )
            if valid_loss < best_val_loss:
                print( 'Saving best model' )
                best_val_loss = valid_loss
                pt.save( tp_network.state_dict(), './models/film_gnn.pth' )
            scheduler.step()
    except KeyboardInterrupt:
        print( 'Aborting Training due to KeyboardInterrupt' )

    # Save convergence data
    np.save( "./models/film_train_convergence.npy",
             np.vstack( (np.array(train_counter), np.array(train_losses), np.array(train_grads)) ) )
    np.save( "./models/film_valid_convergence.npy",
             np.vstack( (np.array(valid_counter), np.array(valid_losses)) ) )

    # Plot
    plt.semilogy( train_counter, train_losses, label='Losses', alpha=0.5 )
    plt.semilogy( train_counter, train_grads, label='Gradient Norms', alpha=0.5 )
    plt.semilogy( valid_counter, valid_losses, label='Validation Losses', alpha=0.5 )
    plt.xlabel( 'Epoch' )
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
