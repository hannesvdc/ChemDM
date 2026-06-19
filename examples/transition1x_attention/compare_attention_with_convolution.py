"""
Head-to-head comparison of the two transition-path models:

  * attention   -> EquivariantTransformer (this directory, SE(3)-Transformer
                   Q/K/V attention), checkpoint experiments/best_gnn.pth
  * convolution -> NewtonE3NN (chemdm, GAT-style convolutional refinement),
                   checkpoint ../transition1x_newton/experiments/best_gnn.pth

Both models are scored on the *same* reactions of a single dataset
(processed_unique, read from this example's data_config.json), so every
reaction yields a matched (rmsd_attention, rmsd_convolution) pair. The matched
structure is what gives the comparison its statistical power: we report the
paired-difference distribution, a Wilcoxon signed-rank test, the per-reaction
win rate, per-layer convergence, an RMSD-vs-arclength profile, and a
reaction-feature breakdown of which architecture wins where.

Configure the run by editing SPLITS below; inference always runs fresh.

Note on leakage: the attention checkpoint was trained on processed_unique, but
the convolution checkpoint was (per its data_config) trained on `processed`.
Evaluating both on processed_unique may therefore let the convolution model see
reactions it trained on. We surface this with a warning; interpret the
convolution numbers as an optimistic bound until it is retrained on
processed_unique.
"""

import sys
import json
from pathlib import Path

import numpy as np
import torch as pt
import matplotlib.pyplot as plt
from chemdm.NewtonE3NN import NewtonE3NN
from EquivariantTransformer import EquivariantTransformer
from scipy.stats import wilcoxon

EXAMPLES = Path( __file__ ).resolve().parent.parent
sys.path.append( str(EXAMPLES) )
ATTENTION_STORE = EXAMPLES / "transition1x_attention" / "experiments"
CONVOLUTION_STORE = EXAMPLES / "transition1x_newton" / "experiments"

from transition1x_attention.test import loadAttentionModel, evaluateML, evaluateMoleculeErrors, _reaction_features
from transition1x_newton.loadNewtonModel import loadNewtonModel
from chemdm.TransitionPathDataset import TransitionPathDataset
from chemdm.util import formula_from_Z

# Splits to compare on. train is the largest (~bulk of ~9.9k reactions), so
# including it makes the run noticeably longer on CPU.
SPLITS = [ "train", "val", "test" ]

# Common arclength grid for the RMSD-vs-path profile (endpoints included).
PROFILE_GRID = np.linspace(0.0, 1.0, 21)


def _per_image_rmsd( final_state: pt.Tensor, x_ref: pt.Tensor ) -> np.ndarray:
    """Per-atom RMSD (A) of each image in the final-layer prediction."""
    assert x_ref.ndim == 3 and x_ref.shape[2] == 3, f"`x_ref` must be shape (n_images, mol_size, 3) but got {x_ref.shape}."

    se = pt.sum( (final_state - x_ref) ** 2, dim=2 )   # (n_images, mol_size)
    mse = pt.mean( se, dim=1 )                          # (n_images,)
    return pt.sqrt( mse ).cpu().numpy()


def evaluate_pair( att_model : EquivariantTransformer, 
                   conv_model : NewtonE3NN, 
                   dataset, 
                   device : pt.device, 
                   n_layers : int):
    """
    Run both models over every reaction in `dataset`, once.

    Returns
      err_att, err_conv : (n_reactions, n_layers) per-layer MSE matrices
      prof_att, prof_conv : (len(PROFILE_GRID),) mean final-layer RMSD vs arclength
    """
    n = len( dataset )
    err_att = np.zeros( (n, n_layers) )
    err_conv = np.zeros( (n, n_layers) )
    prof_att_acc = np.zeros_like(PROFILE_GRID)
    prof_conv_acc = np.zeros_like(PROFILE_GRID)

    for i in range(n):
        if i % 100 == 0:
            print(f"  reaction {i} / {n}")
        traj = dataset[i][-1]

        Z = traj.Z.to(device=device, dtype=pt.int)
        xA = traj.xA.to(device=device, dtype=pt.float32)
        Ga = traj.GA.to(device=device, dtype=pt.int)
        xB = traj.xB.to(device=device, dtype=pt.float32)
        Gb = traj.GB.to(device=device, dtype=pt.int)
        s = traj.s.to(device=device, dtype=pt.float32)
        x_ref = traj.x.to(device=device, dtype=pt.float32)

        # Evalute both models using the same function. Jeuj!
        _, states_a = evaluateML( att_model, s, Z, xA, xB, Ga, Gb ) # type: ignore
        _, states_c = evaluateML( conv_model, s, Z, xA, xB, Ga, Gb )

        # Also re-use, Jeuj!
        err_att[i, :] = evaluateMoleculeErrors( states_a, x_ref )
        err_conv[i, :] = evaluateMoleculeErrors( states_c, x_ref )

        # RMSD-vs-arclength: interpolate this reaction's per-image final-layer
        # RMSD onto the common grid, then accumulate the mean across reactions.
        s_np = s.cpu().numpy()
        order = np.argsort(s_np)
        s_sorted = s_np[order]
        prof_att_acc += np.interp( PROFILE_GRID, s_sorted, _per_image_rmsd(states_a[-1], x_ref)[order] )
        prof_conv_acc += np.interp( PROFILE_GRID, s_sorted, _per_image_rmsd(states_c[-1], x_ref)[order] )

    return err_att, err_conv, prof_att_acc / n, prof_conv_acc / n


def paired_report( rmsd_att : np.ndarray, 
                   rmsd_conv : np.ndarray ):
    """
    Print the marginal and paired summary statistics.
    
    Both input arrays must be one-dimensional (per-reaction rmsd)
    """
    print("\n=== Final-layer per-reaction RMSD (A) ===")
    for name, r in ( ("attention", rmsd_att), ("convolution", rmsd_conv) ):
        p = np.percentile(r, [50, 90, 95, 99])
        print(
            f"  {name:<12s} n={len(r):>6d}  mean={r.mean():.3f}  median={p[0]:.3f}  "
            f"p90={p[1]:.3f}  p95={p[2]:.3f}  p99={p[3]:.3f}  max={r.max():.3f}"
        )

    # Paired: negative diff => attention has the lower (better) RMSD.
    diff = rmsd_att - rmsd_conv
    att_wins = int( np.sum(diff < 0) )
    conv_wins = int( np.sum(diff > 0) )
    ties = int( np.sum(diff == 0) )
    n = len(diff)

    print("\n=== Paired comparison (attention - convolution) ===")
    print(f"  attention better : {att_wins:>6d}  ({100 * att_wins / n:5.1f} %)")
    print(f"  convolution better: {conv_wins:>6d}  ({100 * conv_wins / n:5.1f} %)")
    print(f"  exact ties        : {ties:>6d}")
    print(f"  mean   diff       : {diff.mean():+.4f} A")
    print(f"  median diff       : {np.median(diff):+.4f} A")

    try:
        stat, pval = wilcoxon( rmsd_att, rmsd_conv )
        verdict = "attention" if np.median(diff) < 0 else "convolution"
        print(f"  Wilcoxon signed-rank: W={stat:.1f}  p={pval:.2e}  "
              f"(median favours {verdict})")
    except ValueError as e:
        print(f"  Wilcoxon signed-rank: skipped ({e})")


def feature_breakdown(rmsd_att : np.ndarray, 
                      rmsd_conv : np.ndarray, 
                      features):
    """Median RMSD per model and attention win-rate, stratified by reaction type."""
    discrete = [
        "n_broken", "n_formed", "n_reactive_atoms", "n_reactive_h",
        "has_forming_HX", "has_breaking_HX", "has_ring_change", "has_NOS_at_center",
    ]
    print("\n=== Win-rate by reaction feature ===")
    header = (f"  {'feature':<22s} {'value':>6s} {'n':>5s} "
              f"{'med_att':>8s} {'med_conv':>8s} {'att_win%':>8s}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for feat in discrete:
        values = np.array([f[feat] for f in features])
        for v in sorted(set(values.tolist())):
            mask = values == v
            ra, rc = rmsd_att[mask], rmsd_conv[mask]
            v_str = ("yes" if v else "no") if isinstance(v, (bool, np.bool_)) else str(v)
            win = 100 * np.mean(ra < rc)
            print(f"  {feat:<22s} {v_str:>6s} {mask.sum():>5d} "
                  f"{np.median(ra):>8.3f} {np.median(rc):>8.3f} {win:>7.1f}%")


def make_plots( rmsd_att, rmsd_conv, err_att, err_conv, prof_att, prof_conv, split ):
    diff = rmsd_att - rmsd_conv
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # (a) Overlaid marginal RMSD distributions.
    ax = axes[0, 0]
    bins = np.linspace(0, max(rmsd_att.max(), rmsd_conv.max()), 60)
    ax.hist(rmsd_conv, bins=bins, alpha=0.55, label="convolution", color="tab:orange")
    ax.hist(rmsd_att, bins=bins, alpha=0.55, label="attention", color="tab:blue")
    ax.set_xlabel("per-reaction RMSD [A]")
    ax.set_ylabel("# reactions")
    ax.set_title("Final-layer RMSD distributions")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # (b) Paired difference distribution (the headline view).
    ax = axes[0, 1]
    lim = np.percentile(np.abs(diff), 99)
    ax.hist(np.clip(diff, -lim, lim), bins=60, color="tab:green", edgecolor="black")
    ax.axvline(0.0, color="black", lw=1)
    ax.set_xlabel("RMSD(attention) - RMSD(convolution) [A]")
    ax.set_ylabel("# reactions")
    ax.set_title(f"Paired difference (<0 = attention better, median={np.median(diff):+.3f})")
    ax.grid(axis="y", alpha=0.3)

    # (c) Per-reaction scatter with the y = x reference line.
    ax = axes[1, 0]
    hi = max(rmsd_att.max(), rmsd_conv.max())
    lo = max(min(rmsd_att.min(), rmsd_conv.min()), 1e-3)
    ax.scatter(rmsd_conv, rmsd_att, s=6, alpha=0.3)
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("convolution RMSD [A]")
    ax.set_ylabel("attention RMSD [A]")
    ax.set_title("Per-reaction RMSD (points below line: attention better)")
    ax.grid(alpha=0.3)

    # (d) Per-layer convergence (relative to the initial interpolation) and the
    #     RMSD-vs-arclength profile, on twin axes.
    ax = axes[1, 1]
    rel_att = err_att / err_att[:, 0:1]
    rel_conv = err_conv / err_conv[:, 0:1]
    layers = np.arange(err_att.shape[1])
    ax.semilogy(layers, rel_att.mean(0), "o-", color="tab:blue", label="attention (layer MSE)")
    ax.semilogy(layers, rel_conv.mean(0), "s-", color="tab:orange", label="convolution (layer MSE)")
    ax.set_xlabel("refinement layer")
    ax.set_ylabel("mean relative MSE")
    ax.set_title("Convergence over layers  &  RMSD vs arclength (inset axis)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    ax2 = ax.twiny()
    ax2.plot(PROFILE_GRID, prof_att, color="tab:blue", ls=":", label="attention (vs s)")
    ax2.plot(PROFILE_GRID, prof_conv, color="tab:orange", ls=":", label="convolution (vs s)")
    ax2.set_xlabel("arclength s")

    fig.suptitle(f"Attention vs convolution — {split} split (n={len(rmsd_att)})")
    fig.tight_layout()


def main():
    with open( EXAMPLES / "transition1x_attention" / "data_config.json", "r") as f:
        data_config = json.load(f)
    data_directory = data_config["data_folder"]   # processed_unique
    print(f"Common evaluation dataset: {data_directory}")
    if "processed_unique" not in str(data_directory):
        print("WARNING: data_folder is not processed_unique — check data_config.json")

    # CPU only: test.py documents an MPS memory overflow during evaluation.
    device = pt.device( "cpu" )
    dtype = pt.float32

    print("Loading models...")
    att_model = loadAttentionModel( ATTENTION_STORE, device, dtype )
    conv_model = loadNewtonModel( str(CONVOLUTION_STORE), device, dtype )
    print("WARNING: convolution checkpoint was trained on `processed`; "
          "its numbers on processed_unique may be optimistic (train leakage).")
    n_layers = att_model.n_refinement_steps + 1

    for split in SPLITS:
        print( f"\n################  {split}  ################" )
        dataset = TransitionPathDataset(split, data_directory)

        err_att, err_conv, prof_att, prof_conv = evaluate_pair(
            att_model, conv_model, dataset, device, n_layers
        )

        # MSE is mean over (image, atom) of summed-squared-xyz, so sqrt is the
        # standard per-atom RMSD in Angstrom.
        rmsd_att = np.sqrt(np.maximum(err_att[:, -1], 0.0))
        rmsd_conv = np.sqrt(np.maximum(err_conv[:, -1], 0.0))

        # Reaction features are CPU set-ops; computed from the dataset (fast).
        features = [_reaction_features(dataset[i][-1]) for i in range(len(dataset))]

        paired_report( rmsd_att, rmsd_conv )
        feature_breakdown( rmsd_att, rmsd_conv, features )

        # Worst attention reactions relative to convolution.
        diff = rmsd_att - rmsd_conv
        print("\n=== 5 reactions where attention loses most to convolution ===")
        for idx in np.argsort(-diff)[:5]:
            traj = dataset[int(idx)][-1]
            print(f"  idx={int(idx):>5d}  att={rmsd_att[idx]:.3f}  conv={rmsd_conv[idx]:.3f}  "
                  f"diff={diff[idx]:+.3f}  {formula_from_Z(traj.Z)}  "
                  f"file={dataset.file_names[int(idx)]}")

        make_plots(rmsd_att, rmsd_conv, err_att, err_conv, prof_att, prof_conv, split)

    plt.show()


if __name__ == "__main__":
    main()
