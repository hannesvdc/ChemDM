"""
Training script for the torsional-diffusion score model on GEOM-QM9.

Pipeline (one training step):

    1. Pull a batch from the preparsed TorsionalDataset (collate_torsional
       gives us flat batched tensors + atom_batch / bond_batch indices).
    2. Sample t ~ U(0, 1) per molecule, then σ = σ(t) per molecule, then
       broadcast σ to each rotatable bond via `bond_batch`.
    3. Sample ε ~ N(0, I_m) and form Δτ = σ_bond · ε.
    4. Build x_t = apply_torsion_update(x_0, …, Δτ). Bond lengths and
       bond angles are invariant under this operation.
    5. Build the spatial graph from x_t with cutoff = 5 Å.
    6. Forward through the score network → δτ_pred ∈ R^{m_total}.
    7. Compute the closed-form target ∇log p_{t|0}(τ_t | τ_0)
       = wrapped_normal_score(Δτ, σ_bond).
    8. Karras-style DSM loss with σ² weighting (see notes in `loss_fn`):
           L = mean over bonds( σ_bond² · ‖δτ_pred - target‖² )
       Backprop, Adam step.

Reads `DEVICE` from .env (e.g. DEVICE=mps), default mps. Runs a fixed
number of epochs, prints train loss every `LOG_EVERY` steps, and reports
val loss every epoch.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch as pt
import wandb
from torch.utils.data import DataLoader
from dotenv import load_dotenv

from dataset import TorsionalDataset, collate_torsional
from score_network import TorsionalScoreNetwork
from diffusion import sigma_schedule, wrapped_normal_score

from chemdm.geometry import apply_torsion_update
from chemdm.MoleculeGraph import findAllNeighbors

# Training setup
BATCH_SIZE   = 16
N_EPOCHS     = 5
LR           = 3.0e-4
CUTOFF       = 5.0
LOG_EVERY    = 100              # steps between train-loss prints
NUM_WORKERS  = 0                # DataLoader workers; >0 needs collator pickle-ability
DTYPE        = pt.float32       # float dtype for both model and per-batch tensors

# Weights & Biases
WANDB_ENTITY  = "hannesvdc-open-numerics"
WANDB_PROJECT = "torsional_diffusion"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name", type=str, required=True, dest="name",
        help="Experiment name. Checkpoints go to ./checkpoints/<name>/.",
    )
    return parser.parse_args()


# Loss
def loss_fn( delta_tau_pred: pt.Tensor, # (m,)  model output
             delta_tau: pt.Tensor,      # (m,)  Δτ used to perturb x_0
             sigma_bond: pt.Tensor,     # (m,)  per-bond σ
           ) -> pt.Tensor:
    """
    Denoising score-matching loss with σ²-weighting.

    The target is the closed-form wrapped-normal score; weighting by σ²
    cancels the 1/σ² magnitude of the score and balances the gradient
    contribution across all noise levels of the schedule. Equivalent up
    to a constant to the Karras 'σ·score' parameterisation with a unit-
    weighted MSE.
    """
    target = wrapped_normal_score( delta_tau, sigma_bond )
    return ( sigma_bond ** 2 * (delta_tau_pred - target) ** 2 ).mean()


# One training / evaluation step
def perturb_and_score( model: TorsionalScoreNetwork, batch: dict, device: pt.device ) -> pt.Tensor:
    """
    The shared core of training and validation. Given a clean batch from
    the dataset, sample a diffusion time, perturb torsions, forward through
    the model, and return the DSM loss.
    """
    # Move the batch onto the compute device. BatchedMoleculeGraph.to() handles
    # Z (long, device only) and x (float, device + dtype) consistently.
    mol = batch["mol"].to( device=device, dtype=DTYPE )
    bonds         = batch["bonds"].to( device=device )
    side_atom_idx = batch["side_atom_idx"].to( device=device )
    side_bond_idx = batch["side_bond_idx"].to( device=device )
    bond_batch    = batch["bond_batch"].to( device=device )

    B = int( bond_batch.max().item() ) + 1            # number of molecules in batch
    m = bonds.shape[0]                                # total rotatable bonds in batch

    # Diffusion time + per-bond σ.
    t = pt.rand( B, device=device )
    sigma = sigma_schedule(t)                         # (B,)
    sigma_bond = sigma[bond_batch]                    # (m,)

    # Noise increment Δτ for conditional wrapped normal
    eps = pt.randn( m, device=device )
    delta_tau = sigma_bond * eps                      # (m,)

    # Bridge τ → x: apply Δτ to x_0 to get x_t.
    x_t = apply_torsion_update( mol.x, bonds, side_atom_idx, side_bond_idx, delta_tau )

    # Build the union neighbor graph from current x_t: covalent bonds ∪ atom
    # pairs within CUTOFF. The is_bond flag tells the trunk which is which.
    # KDTree handles batched separation natively via the molecule_id trick.
    mol_t = mol.copyWithNewPositions( x_t )
    neighbors_E2, is_bond = findAllNeighbors( mol_t, CUTOFF )
    neighbors = neighbors_E2.T.contiguous()       # (2, E)
    is_bond   = is_bond.to( dtype=DTYPE )         # float for edge context

    # Score model forward. This is not the backward diffusion, just training the model.
    # The model takes t directly; σ stays in scope for the loss weighting only.
    delta_tau_pred = model(
        mol=mol_t, t=t,
        neighbors=neighbors, is_bond=is_bond,
        bonds=bonds, bond_batch=bond_batch,
    )

    return loss_fn( delta_tau_pred, delta_tau, sigma_bond )


# Main
def main( exp_name: str ) -> None:
    load_dotenv()
    with open( "./data_config.json", "r" ) as f:
        data_config = json.load( f )

    qm9_dir     = Path( data_config["data_folder"] )
    parsed_dir  = qm9_dir.parent / "parsed"
    store_root  = data_config.get( "store_root", "./checkpoints" )
    ckpt_dir    = Path( store_root ) / exp_name
    ckpt_dir.mkdir( parents=True, exist_ok=True )

    device_str  = data_config.get( "device", os.environ.get("DEVICE", "mps") )
    device      = pt.device( device_str )
    setup_wandb = data_config.get( "setup_wandb", True )

    print(f"experiment:  {exp_name}")
    print(f"device:      {device}")
    print(f"checkpoints: {ckpt_dir}")

    # Data
    train_ds = TorsionalDataset( parsed_dir / "train.pt" )
    val_ds = TorsionalDataset( parsed_dir / "val.pt" )
    print(f"train: {len(train_ds):,} conformers   val: {len(val_ds):,} conformers")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_torsional, num_workers=NUM_WORKERS,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_torsional, num_workers=NUM_WORKERS,
        drop_last=False,
    )

    # Model + optimiser
    model = TorsionalScoreNetwork().to( device=device, dtype=DTYPE )
    n_params = sum( p.numel() for p in model.parameters() )
    print(f"model parameters: {n_params:,}")

    # Regular Adam optimizer. Might add weight decay later.
    optimizer = pt.optim.Adam( model.parameters(), lr=LR )

    # Config snapshot + optional Weights&Biases logging.
    experiment_config = {
        "architecture":           "TorsionalScoreNetwork",
        "experiment_name":        exp_name,
        "device":                 device_str,
        "batch_size":             BATCH_SIZE,
        "n_epochs":                N_EPOCHS,
        "lr":                     LR,
        "cutoff":                 CUTOFF,
        "n_trainable_parameters": n_params,
    }
    with open( ckpt_dir / "config.json", "w" ) as f:
        json.dump( experiment_config, f, indent=2 )

    run = None
    if setup_wandb:
        run = wandb.init(
            entity  = WANDB_ENTITY,
            project = WANDB_PROJECT,
            name    = exp_name,
            config  = experiment_config,
        )

    # Training
    best_val_loss = float("inf")
    for epoch in range(1, N_EPOCHS + 1):
        model.train()

        epoch_t0 = time.time()
        running_loss = 0.0
        n_steps  = 0
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad( set_to_none=True )

            loss = perturb_and_score( model, batch, device )

            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            n_steps += 1
            if step % LOG_EVERY == 0:
                avg = running_loss / max(n_steps, 1)
                print(
                    f"  Epoch {epoch}  step {step:>5d}/{len(train_loader):<5d}  "
                    f"train_loss = {loss.item():.4f}  (running avg {avg:.4f})"
                )

        train_loss = running_loss / n_steps
        epoch_dt = time.time() - epoch_t0

        # Validation
        model.eval()

        val_loss = 0.0
        n_val    = 0
        with pt.no_grad():
            for batch in val_loader:
                val_loss += float( perturb_and_score(model, batch, device).item() )
                n_val += 1
        val_loss /= max(n_val, 1)

        print(
            f"[epoch {epoch:>2d}]  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"({epoch_dt:.1f}s)"
        )

        # Checkpointing. `latest.pt` is overwritten every epoch and carries
        # enough state to resume (model + optimizer + epoch). `best.pt`
        # tracks the best validation loss seen so far and stores just the
        # model state_dict — that's the one to load for sampling.
        pt.save(
            {
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch":     epoch,
                "train_loss": train_loss,
                "val_loss":  val_loss,
            },
            ckpt_dir / "latest.pt",
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            pt.save( model.state_dict(), ckpt_dir / "best.pt" )
            print( f"  -> new best val_loss ({val_loss:.4f}); saved best.pt" )

        if setup_wandb and run is not None:
            run.log({
                "epoch":         epoch,
                "train_loss":    train_loss,
                "val_loss":      val_loss,
                "best_val_loss": best_val_loss,
                "epoch_time":    epoch_dt,
                "lr":            optimizer.param_groups[0]["lr"],
            })


if __name__ == "__main__":
    args = parse_args()
    main( args.name )
