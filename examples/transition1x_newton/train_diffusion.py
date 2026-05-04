import math
import numpy as np
import torch as pt
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

import wandb
from pathlib import Path
import json
import argparse
import traceback

from chemdm.TransitionPathDataset import TransitionPathDataset
from chemdm.MoleculeGraph import BatchedMoleculeGraph
from chemdm.MolecularEmbeddingNetwork import MolecularEmbeddingGNN
from chemdm.ResidualDiffusionE3NN import ResidualDiffusionE3NN
from chemdm.util import getGradientNorm, collate_molecules

from loadNewtonModel import loadNewtonModel

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
    root = data_config.get( "store_root" )

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

    # Load the pre-trained Newton TP model. Ensure it's frozen.
    device = pt.device( device_name )
    dtype = pt.float32
    tp_network = loadNewtonModel( root, device, dtype )
    tp_network.eval()
    for p in tp_network.parameters():
        p.requires_grad_(False)

    # Global molecular information
    d_cutoff = 5.0
    n_rbf = 10

    # Endpoint embedding networks
    embedding_state_size = 64
    embedding_message_size = 64
    n_embedding_layers = 5
    xA_embedding = MolecularEmbeddingGNN( embedding_state_size, embedding_message_size, n_embedding_layers, d_cutoff )
    xB_embedding = MolecularEmbeddingGNN( embedding_state_size, embedding_message_size, n_embedding_layers, d_cutoff )

    # E3NN transition-path network
    irreps_node_str = "48x0e + 16x1o + 16x1e + 8x2e"
    n_denoising_steps = 3
    residual_scale = 1.0
    diffusion_network = ResidualDiffusionE3NN( xA_embedding, 
                                               xB_embedding, 
                                               irreps_node_str, 
                                               n_denoising_steps, 
                                               d_cutoff, 
                                               n_arclength_freq=8, 
                                               n_rbf=n_rbf,
                                               residual_scale=residual_scale )
    diffusion_network.to( device=device, dtype=dtype )
    n_params = sum(p.numel() for p in diffusion_network.parameters() if p.requires_grad)
    print( "Number of Trainable Parameters: ", n_params )

    # Build the optimizer and scheduler
    lr_min = 1e-6
    lr_max = 1e-3
    n_epochs = 1000
    warmup_epochs = 25
    weight_decay = 1e-2

    optimizer = AdamW( diffusion_network.parameters(), lr=lr_max, weight_decay=weight_decay, amsgrad=True )
    warmup_scheduler = LinearLR( optimizer, start_factor=lr_min / lr_max, end_factor=1.0, total_iters=warmup_epochs )
    cosine_scheduler = CosineAnnealingLR( optimizer, T_max=n_epochs - warmup_epochs, eta_min=lr_min )
    scheduler = SequentialLR( optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs], )

    # Diffusion model - Cosine schedule (Nichol & Dhariwal, 2021)
    T = 100
    cosine_s = 0.008
    steps = pt.linspace(0, T, T + 1, dtype=pt.float64)
    alpha_bar_full = pt.cos( (steps / T + cosine_s) / (1.0 + cosine_s) * (math.pi / 2.0) ) ** 2
    alpha_bar_full = alpha_bar_full / alpha_bar_full[0]

    # T entries for diffusion indices t = 0, ..., T-1.
    # alpha_bar[0] is nearly clean, alpha_bar[-1] is nearly pure noise.
    alpha_bar = alpha_bar_full[1:].to(device=device,dtype=dtype)
    sqrt_alpha_bar = pt.sqrt(alpha_bar)
    sqrt_one_minus_alpha_bar = pt.sqrt(1.0 - alpha_bar)

    # Everything has been set up, now log the config and initialize weights&biases
    experiment_config = {
        "architecture": "DiffusionE3NN",
        "experiment_name": exp_name,
        "data_directory": data_directory,
        "device": device_name,
        "weight_decay": weight_decay,
        "batch_size": B,
        "epochs": n_epochs,
        "irreps_node": irreps_node_str,
        "n_denoising_steps": n_denoising_steps,
        "d_cutoff": d_cutoff,
        "n_rbf": n_rbf,
        "embedding_state_size": embedding_state_size,
        "embedding_message_size": embedding_message_size,
        "n_embedding_layers": n_embedding_layers,
        "n_trainable_parameters": n_params,
        "T": T,
        "cosine_s": cosine_s,
        "residual_scale": residual_scale,
    }
    exp_dir = make_experiment_dir( exp_name, root=root )
    with open(exp_dir / "config.json", "w") as f:
        json.dump(experiment_config, f, indent=2)
    if setup_wandb:
        run = wandb.init(
            # Set the wandb entity where your project will be logged (generally your team name).
            entity="hannesvdc-open-numerics",
            # Set the wandb project where this run will be logged.
            project="newton_diffusion",
            # Set the name
            name=exp_name,
            # Track hyperparameters and run metadata.
            config=experiment_config
        )

    # Average MSE error per molecule
    def molecule_balanced_mean( atom_values: pt.Tensor,  molecule_id: pt.Tensor ) -> pt.Tensor:
        molecule_id = molecule_id.to(device=atom_values.device).long()

        # If molecule_id is 0,1,...,B-1 this is enough.
        n_molecules = int(molecule_id.max().item()) + 1
        sums = pt.zeros(n_molecules, device=atom_values.device, dtype=atom_values.dtype)
        counts = pt.zeros(n_molecules, device=atom_values.device, dtype=atom_values.dtype)

        sums.index_add_(0, molecule_id, atom_values)
        counts.index_add_(0, molecule_id, pt.ones_like(atom_values))

        per_molecule_mean = sums / counts.clamp_min(1.0)
        return per_molecule_mean.mean()
    
    # Diffusion loss
    def per_molecule_mse_loss( eps_pred: pt.Tensor, eps_target: pt.Tensor, molecule_id: pt.Tensor, ) -> pt.Tensor:
        assert eps_pred.shape == eps_target.shape
        assert eps_pred.ndim == 2 and eps_pred.shape[1] == 3
        assert molecule_id.shape[0] == eps_pred.shape[0]

        atom_sq_error = ((eps_pred - eps_target) ** 2).sum(dim=-1)  # (N_total,)
        return molecule_balanced_mean( atom_sq_error, molecule_id )

    def single_state_loss( x_ref : pt.Tensor, x_model : pt.Tensor, molecule_id : pt.Tensor ) -> pt.Tensor:
        atom_sq_error = ((x_model - x_ref) ** 2).sum(dim=-1)  # (N_total,)
        return molecule_balanced_mean( atom_sq_error, molecule_id )

    # General Batch evaluation function
    def evaluate_diffusion_batch( xA: BatchedMoleculeGraph,
                                  xB: BatchedMoleculeGraph,
                                  s: pt.Tensor,
                                  t: pt.Tensor,
                                  x_ref: pt.Tensor,
                                ) -> tuple[pt.Tensor, pt.Tensor]:
        """
        Evaluate one diffusion-training batch.

        `t` is expected to be integer diffusion indices of shape (B,), one per molecule.
        The model receives normalized atomwise t in [0, 1].
        """

        # Evaluate the deterministic model (frozen)
        with pt.no_grad():
            x_newton, _ = tp_network(xA, xB, s)
        x_base = x_newton.x  # (N_total, 3)

        # Prepare Diffusion Input
        c0 = (x_ref - x_base) / residual_scale  # (N_total, 3)
        eps = pt.randn_like(c0)

        # Diffusion input
        sqrt_ab = sqrt_alpha_bar[t][:, None]
        sqrt_omab = sqrt_one_minus_alpha_bar[t][:, None]
        c_t = sqrt_ab * c0 + sqrt_omab * eps

        # Predict noise. Normalized diffusion time for the neural network.
        t_atom = t.to(dtype=x_ref.dtype) / float(T - 1)
        eps_pred = diffusion_network( xA=xA, xB=xB, s=s, x_base=x_newton, c_t=c_t, t=t_atom )

        # Compute the molecule-balanced noise-prediction loss.
        loss = per_molecule_mse_loss( eps_pred, eps, molecule_id=xA.molecule_id )

        # Diagnostic: clean residual reconstruction error from predicted eps.
        with pt.no_grad():
            sqrt_ab_safe = sqrt_ab.clamp_min(1e-3)
            c0_pred = (c_t - sqrt_omab * eps_pred) / sqrt_ab_safe
            gate = 4.0 * s[:, None] * (1.0 - s[:, None])
            x_pred = x_base + gate * residual_scale * c0_pred

            final_state_loss = single_state_loss( x_ref, x_pred, xA.molecule_id.long(), )

        return loss, final_state_loss
    
    # Main Training Loop
    train_counter = []
    train_losses = []
    train_grads = []
    def train( epoch : int ) -> Tuple[float,float, float]:
        diffusion_network.train()

        # Generate random batching indices
        n_batches = len(train_loader)
        epoch_loss = 0.0
        n_success = 0
        final_state_epoch_loss = 0.0
        for batch_idx, (xA, xB, s, x_ref) in enumerate( train_loader ):
            optimizer.zero_grad( set_to_none=True )

            # Gather the training data
            xA = xA.to( device=device, dtype=dtype )
            xB = xB.to( device=device, dtype=dtype )
            s = s.to( device=device, dtype=dtype )
            x_ref = x_ref.to( device=device, dtype=dtype )

            # Sample diffusion time
            with pt.no_grad():
                n_molecules = len( pt.unique(xA.molecule_id) )
                t_mol = pt.randint( 0, T, (n_molecules,), device=x_ref.device )
                t_atom = t_mol[ xA.molecule_id.long() ]

            # Evalute the loss
            try:
                loss, final_state_loss = evaluate_diffusion_batch( xA, xB, s, t_atom, x_ref )
            except:
                print(traceback.format_exc())
                print('Blowup during batch evaluation. Continuing.')
                continue
            epoch_loss += float( loss.item() )
            final_state_epoch_loss += float( final_state_loss.item() )
            n_success += 1

            # Make an optmizer step
            loss.backward()
            grad_norm = getGradientNorm( diffusion_network )
            pt.nn.utils.clip_grad_norm_( diffusion_network.parameters(), 1.0 )
            optimizer.step()

            # Print some information
            epoch_idx = epoch + (batch_idx+1.0) / n_batches
            train_counter.append(epoch_idx)
            train_losses.append(loss.item())
            train_grads.append(grad_norm)
            if (batch_idx+1) % 1 == 0:
                print('Train Epoch: {} [{}/{}] \tLoss: {:.6f} \t Gradient Norm {:.6f} \t Learning Rate {:.2E}'
                    .format( epoch, batch_idx+1, n_batches, loss.item(), grad_norm, optimizer.param_groups[-1]["lr"] ), flush=True)
                
        denom = max(n_success, 1)
        return epoch_loss / denom, grad_norm, final_state_epoch_loss / denom

    valid_counter = []
    valid_losses = []
    valid_final_losses = []
    @pt.no_grad()
    def validate(epoch: int) -> tuple[float, float]:
        diffusion_network.eval()

        n_batches = len(valid_loader)
        epoch_loss = 0.0
        final_state_epoch_loss = 0.0
        n_success = 0
        for batch_idx, (xA, xB, s, x_ref) in enumerate(valid_loader):

            xA = xA.to(device=device, dtype=dtype)
            xB = xB.to(device=device, dtype=dtype)
            s = s.to(device=device, dtype=dtype)
            x_ref = x_ref.to(device=device, dtype=dtype)

            try:
                molecule_id = xA.molecule_id.long()
                n_molecules = int(molecule_id.max().item()) + 1

                # One integer diffusion timestep per molecule.
                t_mol = pt.randint(0, T, (n_molecules,), device=x_ref.device)
                t_atom = t_mol[molecule_id.long()]

                loss, final_state_loss = evaluate_diffusion_batch( xA, xB, s, t_atom, x_ref )
            except Exception:
                print(traceback.format_exc())
                print("Blowup during validation batch evaluation. Continuing.")
                continue

            epoch_loss += float(loss.item())
            final_state_epoch_loss += float(final_state_loss.item())
            n_success += 1
            epoch_idx = epoch + (batch_idx + 1.0) / n_batches
            valid_counter.append(epoch_idx)
            valid_losses.append(loss.item())
            valid_final_losses.append(final_state_loss.item())

            if (batch_idx + 1) % 10 == 0:
                print( "Validation Epoch: {} [{}/{}] \tDiffusion Loss: {:.6f} "
                       "\tReconstruction Loss: {:.6f}".format( epoch, batch_idx + 1, n_batches, loss.item(), final_state_loss.item(), ),
                    flush=True, )

        denom = max(n_success, 1)
        return epoch_loss / denom, final_state_epoch_loss / denom

    # The simplest of training loops for now.            
    best_val_loss = float("inf")
    try:
        for epoch in range(n_epochs):
            train_loss, train_grad, train_res_loss = train( epoch )
            print( "Train Epoch {} \tTotal Loss: {}\n".format(epoch, train_loss), flush=True )

            valid_loss, valid_res_loss = validate( epoch )
            print( "Validation Epoch {} \tTotal Loss: {}" .format(epoch, valid_loss), flush=True )
            if valid_loss < best_val_loss:
                print('Saving best model')
                best_val_loss = valid_loss
                pt.save( diffusion_network.state_dict(), exp_dir / "best_diffusion_gnn.pth" )

            # Log to weights & biases
            if setup_wandb:
                run.log({"epoch": epoch, "train_loss": train_loss,
                         "train_grad": train_grad, "valid_loss" : valid_loss,
                         "best_val_loss" : best_val_loss,
                         "train_final_state_loss" : train_res_loss,
                         "valid_final_state_loss" : valid_res_loss,
                         "lr": optimizer.param_groups[0]["lr"]})

            if epoch % 10 == 0:
                pt.save( diffusion_network.state_dict(), exp_dir / "diffusion_gnn.pth" )
                pt.save( optimizer.state_dict(), exp_dir / "diffusion_optimizer.pth" )

            scheduler.step()

    except KeyboardInterrupt:
        print('Aborting Training due to KeyboardInterrupt')
    finally:
        if setup_wandb:
            run.finish()

    # Store training convergence
    np.save( exp_dir / "diffusion_train_convergence.npy", np.vstack( (np.array(train_counter), np.array(train_losses), np.array(train_grads)) ) )
    np.save( exp_dir / "diffusion_valid_convergence.npy", np.vstack( (np.array(valid_counter), np.array(valid_losses) ) ) )

    # Plot the loss and grad norm
    plt.semilogy( train_counter, train_losses, label='Losses', alpha=0.5)
    plt.semilogy( train_counter, train_grads, label='Gradient Norms', alpha=0.5)
    plt.semilogy( valid_counter, valid_losses, label='Validation Losses', alpha=0.5)
    plt.xlabel('Epoch')
    plt.legend()
    plt.tight_layout()
    plt.savefig(exp_dir / "diffusion_convergence.png", dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == '__main__':
    args = parse_args( )
    main( args.name )
