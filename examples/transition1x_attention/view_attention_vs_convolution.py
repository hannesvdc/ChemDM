"""
Visual, single-reaction comparison of the two transition-path models.

Pick one reaction from the transition1x dataset (SPLIT + REACTION_INDEX below)
and play BOTH predicted paths as one synchronized ChemistryViewer movie:
attention on the left, convolution on the right, advancing through the same
arclength together. This is a quick way to eyeball the smoothness of each
generated path side by side.

It also reports, for each path, the max perpendicular force to the path (the
component of the true xTB force orthogonal to the path tangent — what NEB
actually drives to zero). A smaller value means the predicted path sits closer
to a force-balanced minimum-energy path, i.e. a better NEB starting point.

Single window, single `view_movie` call: pywebview's event loop is one-shot and
blocking, so two separate movies can't both be launched from one process. We
sidestep that by drawing both molecules in every frame (the convolution copy
shifted along +x), which also gives genuinely synchronized playback. That same
one-shot constraint is why this shows a single reaction per run rather than
re-prompting in a loop.

The reaction is chosen interactively at startup (split + index or filename);
there are no CLI args.
"""

import sys
import json
from pathlib import Path

import numpy as np
import torch as pt

from chemviewer import view_movie
from chemdm.TBLitePotential import TBLitePotential
from chemdm.nebXtbDirect import evaluate_path, neb_force, neb_force_metrics
from chemdm.Constants import KJ_MOL_TO_EV

EXAMPLES = Path(__file__).resolve().parent.parent
sys.path.append(str(EXAMPLES))
ATTENTION_STORE = EXAMPLES / "transition1x_attention" / "experiments"
CONVOLUTION_STORE = EXAMPLES / "transition1x_newton" / "experiments"

from transition1x_attention.test import loadAttentionModel, evaluateML
from transition1x_newton.loadNewtonModel import loadNewtonModel
from chemdm.TransitionPathDataset import TransitionPathDataset
from chemdm.util import formula_from_Z

# --- Display / physics knobs -------------------------------------------------
FPS = 10                # movie playback speed
GAP = 3.0               # Angstrom gap between the two molecules in the frame
SPRING_K = 0.0          # spring constant for neb_force; F_perp is independent of it

# A candidate bond is drawn in a frame only when the two atoms are within
# BOND_SCALE * (sum of covalent radii). This lets breaking/forming bonds appear
# and disappear as the reaction proceeds instead of rendering as long sticks.
BOND_SCALE = 1.3
COVALENT_RADII = {1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57,
                  15: 1.07, 16: 1.05, 17: 1.02}  # Cordero et al., Angstrom


def ask_reaction(data_directory):
    """Prompt for a split and a reaction (index or filename). Returns (split, dataset, index)."""
    valid = ("train", "val", "test")
    split = input(f"Split {valid} [test]: ").strip().lower() or "test"
    while split not in valid:
        split = input(f"  enter one of {valid} [test]: ").strip().lower() or "test"

    dataset = TransitionPathDataset(split, data_directory)
    n = len(dataset)
    names = list(dataset.file_names)

    while True:
        raw = input(f"{split} has {n} reactions. Index [0..{n-1}] or filename: ").strip()
        if raw in names:
            return split, dataset, names.index(raw)
        if raw + ".pkl" in names:
            return split, dataset, names.index(raw + ".pkl")
        try:
            idx = int(raw)
        except ValueError:
            print("  not an integer or a known filename — try again.")
            continue
        if 0 <= idx < n:
            return split, dataset, idx
        print(f"  index out of range (0..{n-1}).")


def predict_path(model, traj, device) -> np.ndarray:
    """Final predicted path for one reaction, shape (n_images, n_atoms, 3), Angstrom."""
    Z = traj.Z.to(device=device, dtype=pt.int)
    xA = traj.xA.to(device=device, dtype=pt.float32)
    Ga = traj.GA.to(device=device, dtype=pt.int)
    xB = traj.xB.to(device=device, dtype=pt.float32)
    Gb = traj.GB.to(device=device, dtype=pt.int)
    s = traj.s.to(device=device, dtype=pt.float32)
    x, _ = evaluateML(model, s, Z, xA, xB, Ga, Gb)
    return x.cpu().numpy().astype(float)


def perp_force_report(Z_np: np.ndarray, path: np.ndarray, name: str) -> dict:
    """Max (and RMS) perpendicular force to the path, via xTB. Units: kJ/mol/A."""
    xtb = TBLitePotential(Z=Z_np)
    E, F = evaluate_path(xtb, path)                       # kJ/mol, kJ/mol/A
    _, F_perp = neb_force(path, E, F, SPRING_K)           # (M-2, n_atoms, 3)

    perp_atom_norms = np.linalg.norm(F_perp, axis=-1)     # (M-2, n_atoms)
    max_perp = float(perp_atom_norms.max())
    worst_image = int(perp_atom_norms.max(axis=1).argmax()) + 1   # +1: endpoints excluded
    metrics = neb_force_metrics(path, E, F, SPRING_K)

    print(f"\n  {name}")
    print(f"    max perpendicular force : {max_perp:8.3f} kJ/mol/A   "
          f"({max_perp * KJ_MOL_TO_EV:.4f} eV/A)   at interior image {worst_image}")
    print(f"    max per-image RMS perp  : {metrics['max_force_rms']:8.3f} kJ/mol/A")
    print(f"    barrier (rel. to start) : {metrics['barrier_kJ_mol']:8.3f} kJ/mol")
    return {"max_perp": max_perp, "F_perp": F_perp, "metrics": metrics}


def frame_bonds(Z_np, x):
    """Perceive connectivity for one geometry: every atom pair within covalent range.

    Independent of GA/GB, so transient bonds present in neither the reactant nor
    product graph still appear if the geometry calls for them.
    """
    n = len(Z_np)
    radii = np.array([COVALENT_RADII.get(int(z), 0.77) for z in Z_np])
    drawn = []
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(x[i] - x[j]) <= BOND_SCALE * (radii[i] + radii[j]):
                drawn.append((i, j))
    return drawn


def side_by_side_frames(Z_np, path_left, path_right, gap):
    """Build movie frames with both molecules per frame: left as-is, right shifted +x.

    Connectivity is perceived per frame (and per side) from interatomic distances,
    so bonds break/form as the reaction proceeds instead of drawing stretched
    sticks across space.
    """
    n_atoms = len(Z_np)
    both = np.concatenate([path_left, path_right], axis=0)
    offset = np.array([(both[..., 0].max() - both[..., 0].min()) + gap, 0.0, 0.0])
    Z_pair = np.concatenate([Z_np, Z_np])

    frames = []
    for xl, xr in zip(path_left, path_right):
        bl = frame_bonds(Z_np, xl)
        br = frame_bonds(Z_np, xr)
        bonds_pair = [tuple(b) for b in bl] + [(i + n_atoms, j + n_atoms) for i, j in br]
        x_pair = np.concatenate([xl, xr + offset], axis=0)
        frames.append((Z_pair, x_pair, bonds_pair))
    return frames


def main():
    with open(EXAMPLES / "transition1x_attention" / "data_config.json", "r") as f:
        data_directory = json.load(f)["data_folder"]

    device, dtype = pt.device("cpu"), pt.float32
    split, dataset, index = ask_reaction(data_directory)
    traj = dataset[index][-1]

    Z_np = traj.Z.cpu().numpy().astype(int)
    formula = formula_from_Z(traj.Z)
    n_images = int(traj.s.numel())
    print(f"\nreaction: {formula}  (N={len(Z_np)} atoms, {n_images} images)  "
          f"{split}[{index}]  file={dataset.file_names[index]}")

    print("Loading models...")
    att_model = loadAttentionModel(ATTENTION_STORE, device, dtype)
    conv_model = loadNewtonModel(str(CONVOLUTION_STORE), device, dtype)

    path_att = predict_path(att_model, traj, device)
    path_conv = predict_path(conv_model, traj, device)

    print("\nMax perpendicular force to the predicted path (lower = better NEB start):")
    perp_force_report(Z_np, path_att, "attention  (left) ")
    perp_force_report(Z_np, path_conv, "convolution (right)")

    frames = side_by_side_frames(Z_np, path_att, path_conv, GAP)

    print("\nOpening viewer: attention (left)  vs  convolution (right). "
          "Close the window to exit.")
    view_movie(
        frames,
        title=f"{formula}  —  attention (left)  vs  convolution (right)",
        fps=FPS,
    )


if __name__ == "__main__":
    main()
