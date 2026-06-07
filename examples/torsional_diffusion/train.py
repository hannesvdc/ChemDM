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

import os
import time
from pathlib import Path

import torch as pt
from torch.utils.data import DataLoader
from dotenv import load_dotenv

from dataset import TorsionalDataset, collate_torsional
from score_network import TorsionalScoreNetwork
from diffusion import sigma_schedule, wrapped_normal_score, radius_graph

from chemdm.geometry import apply_torsion_update

# Training setup
BATCH_SIZE   = 32
N_EPOCHS     = 5
LR           = 3.0e-4
CUTOFF       = 5.0
LOG_EVERY    = 50          # steps between train-loss prints
NUM_WORKERS  = 0           # DataLoader workers; >0 needs collator pickle-ability


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
def perturb_and_score( model:  TorsionalScoreNetwork, batch:  dict[str, pt.Tensor], device: pt.device ) -> pt.Tensor:
    """
    The shared core of training and validation. Given a clean batch from
    the dataset, sample a diffusion time, perturb torsions, forward through
    the model, and return the DSM loss.
    """
    # Move the batch onto the compute device once, in place.
    for k, v in batch.items():
        batch[k] = v.to( device=device, non_blocking=True )

    B = int( batch["bond_batch"].max().item() ) + 1   # number of molecules in batch
    m = batch["bonds"].shape[0]                       # total rotatable bonds in batch

    # Diffusion time + per-bond σ.
    t = pt.rand( B, device=device )
    sigma = sigma_schedule(t)  # (B,)  one σ per molecule, B=32
    sigma_bond = sigma[batch["bond_batch"]]  # (m,) e.g. [0, 0, 0, 1, 1, 2, 2, 2, 2, ...] given mol_0 | mol_1 | mol_2 | ...

    # Noise increment Δτ for conditional wrapped normal
    eps = pt.randn(m, device=device)
    delta_tau = sigma_bond * eps                      # (m,)

    # Bridge τ → x: apply Δτ to x_0 to get x_t.
    x_t = apply_torsion_update( batch["x"], batch["bonds"], batch["side_atom_idx"], batch["side_bond_idx"], delta_tau )

    # Build the spatial graph from current x_t (must be rebuilt — atoms moved).
    edge_index = radius_graph( x_t, batch["atom_batch"], cutoff=CUTOFF )

    # Score model forward. This is not the backward diffusion, just training the model.
    # The model takes t directly; σ stays in scope for the loss weighting only.
    delta_tau_pred = model(
        Z=batch["Z"], x=x_t, t=t,
        edge_index=edge_index, bonds=batch["bonds"],
        atom_batch=batch["atom_batch"], bond_batch=batch["bond_batch"],
    )

    return loss_fn( delta_tau_pred, delta_tau, sigma_bond )


# Main
def main() -> None:
    load_dotenv()

    qm9_dir = Path( os.environ["QM9_FOLDER"] )
    parsed_dir = qm9_dir.parent / "parsed"
    ckpt_dir   = Path( "./checkpoints" )
    ckpt_dir.mkdir( parents=True, exist_ok=True )

    device_str = os.environ.get("DEVICE", "mps")
    device = pt.device( device_str )
    print(f"device: {device}")
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
    model = TorsionalScoreNetwork().to(device)
    n_params = sum( p.numel() for p in model.parameters() )
    print(f"model parameters: {n_params:,}")

    # Regular Adam optimizer. Might add weight decay later.
    optimizer = pt.optim.Adam( model.parameters(), lr=LR )

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


if __name__ == "__main__":
    main()
