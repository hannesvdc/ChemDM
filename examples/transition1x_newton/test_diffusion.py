import math
import json
from pathlib import Path

import torch as pt
from torch.utils.data import DataLoader

from chemdm.TransitionPathDataset import TransitionPathDataset
from chemdm.MolecularEmbeddingNetwork import MolecularEmbeddingGNN
from chemdm.ResidualDiffusionE3NN import ResidualDiffusionE3NN
from chemdm.util import collate_molecules
from loadNewtonModel import loadNewtonModel
from chemdm.MoleculeGraph import Molecule

@pt.no_grad()
def cosine_alpha_bar(T, device, dtype):
    s = 0.008
    steps = pt.linspace(0, T, T + 1, dtype=pt.float64, device=device)
    f = pt.cos((steps / T + s) / (1.0 + s) * (math.pi / 2.0)) ** 2
    f = f / f[0]
    return f[1:].to(dtype=dtype)

@pt.no_grad()
def rmse(x, y):
    return pt.sqrt(pt.mean((x - y) ** 2)).item()

@pt.no_grad()
def ddim_step(c_t : pt.Tensor, eps_pred : pt.Tensor, t, t_prev, alpha_bar):
    alpha_t = alpha_bar[t]
    c0_pred = (c_t - pt.sqrt(1.0 - alpha_t) * eps_pred) / pt.sqrt(alpha_t).clamp_min(1e-8)

    if t_prev < 0:
        alpha_prev = pt.tensor(1.0, device=c_t.device, dtype=c_t.dtype)
    else:
        alpha_prev = alpha_bar[t_prev]

    c_prev = pt.sqrt(alpha_prev) * c0_pred + pt.sqrt(1.0 - alpha_prev) * eps_pred
    return c_prev


@pt.no_grad()
def sample_path(diffusion_model : pt.nn.Module, 
                xA : Molecule, 
                xB : Molecule, 
                s : pt.Tensor, 
                x_newton : Molecule, 
                alpha_bar : pt.Tensor, 
                residual_scale : float, 
                T : int, 
                n_steps : int = 20):
    x_base = x_newton.x
    c_t = pt.randn_like(x_base)

    times = pt.linspace(T - 1, 0, n_steps, device=x_base.device).round().long()
    times = pt.unique_consecutive(times)

    for i, t in enumerate(times):
        t_prev = times[i + 1] if i + 1 < len(times) else pt.tensor(-1, device=x_base.device)
        t_atom = pt.full((len(xA.Z),), int(t.item()), device=x_base.device, dtype=pt.long)
        t_norm = t_atom.to(dtype=x_base.dtype) / float(T - 1)

        eps_pred = diffusion_model(xA, xB, s, x_newton, c_t, t_norm)
        c_t = ddim_step(c_t, eps_pred, int(t.item()), int(t_prev.item()), alpha_bar)

    gate = 4.0 * s[:, None] * (1.0 - s[:, None])
    return x_base + gate * residual_scale * c_t


# def write_xyz(filename, Z, x_path):
#     symbols = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 15: "P", 16: "S", 17: "Cl"}
#     Z = Z.detach().cpu().numpy()
#     x_path = x_path.detach().cpu().numpy()

#     with open(filename, "w") as f:
#         for k, x in enumerate(x_path):
#             f.write(f"{len(Z)}\n")
#             f.write(f"image {k}\n")
#             for z, r in zip(Z, x):
#                 f.write(f"{symbols.get(int(z), str(int(z)))} {r[0]:.8f} {r[1]:.8f} {r[2]:.8f}\n")


def main():
    with open("./data_config.json", "r") as f:
        cfg = json.load(f)

    device = pt.device(cfg["device"])
    dtype = pt.float32
    root = cfg.get("store_root")
    data_dir = cfg["data_folder"]

    # Settings must match training
    T = 100
    residual_scale = 1.0
    diffusion_ckpt = Path(root) / "best_diffusion_gnn.pth"
    # out_dir = Path(root) / "diffusion_sampling_test"
    # out_dir.mkdir(parents=True, exist_ok=True)

    newton_model = loadNewtonModel(root, device, dtype)
    newton_model.eval()

    d_cutoff = 5.0
    xA_emb = MolecularEmbeddingGNN(64, 64, 5, d_cutoff)
    xB_emb = MolecularEmbeddingGNN(64, 64, 5, d_cutoff)
    diffusion_model = ResidualDiffusionE3NN(
        xA_emb,
        xB_emb,
        "48x0e + 16x1o + 16x1e + 8x2e",
        3,
        d_cutoff,
        n_arclength_freq=8,
        n_rbf=10,
        residual_scale=residual_scale,
    ).to(device=device, dtype=dtype)
    diffusion_model.load_state_dict(pt.load(diffusion_ckpt, map_location=device, weights_only=True))
    diffusion_model.eval()

    alpha_bar = cosine_alpha_bar(T, device, dtype)

    # Transition1x dataset
    dataset = TransitionPathDataset( "train", data_dir )
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_molecules)

    n_reactions = 5
    n_samples = 4
    for reaction_idx, (xA, xB, s, x_ref) in enumerate(loader):
        if reaction_idx >= n_reactions:
            break

        xA = xA.to(device=device, dtype=dtype)
        xB = xB.to(device=device, dtype=dtype)
        s = s.to(device=device, dtype=dtype).flatten()
        x_ref = x_ref.to(device=device, dtype=dtype)

        x_newton, _ = newton_model(xA, xB, s)

        newton_rmse = rmse(x_newton.x, x_ref)

        best_x = None
        best_rmse = float("inf")

        for _ in range(n_samples):
            x_sample = sample_path(diffusion_model, xA, xB, s, x_newton, alpha_bar, residual_scale, T)
            sample_rmse = rmse(x_sample, x_ref)

            if sample_rmse < best_rmse:
                best_rmse = sample_rmse
                best_x = x_sample

        print(f"Reaction {reaction_idx}: Newton RMSE={newton_rmse:.4f}, best diffusion RMSE={best_rmse:.4f}")

        # Save best sampled path
        mol_size = int((xA.molecule_id == xA.molecule_id[0]).sum().item())
        n_images = len(xA.Z) // mol_size

        #write_xyz(
        #    out_dir / f"reaction_{reaction_idx:03d}_diffusion.xyz",
        #    xA.Z[:mol_size],
        #    best_x.reshape(n_images, mol_size, 3),
        #)


if __name__ == "__main__":
    main()