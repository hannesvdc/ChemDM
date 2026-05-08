import json

import torch as pt
from torch.utils.data import DataLoader

from chemdm.TransitionPathDataset import TransitionPathDataset
from chemdm.util import collate_molecules
from loadNewtonModel import loadNewtonModel
from loadDiffusionModel import loadDiffusionModel
from sample_path import sample_path


@pt.no_grad()
def rmse(x, y):
    return pt.sqrt(pt.mean((x - y) ** 2)).item()


def main():
    with open("./data_config.json", "r") as f:
        cfg = json.load(f)

    device = pt.device(cfg["device"])
    dtype = pt.float32
    root = cfg.get("store_root")
    data_dir = cfg["data_folder"]

    # Settings must match training
    T = 100
    residual_scale = 0.15

    newton_model = loadNewtonModel( root, device, dtype )
    newton_model.eval()
    diffusion_model = loadDiffusionModel( root, device, dtype )
    diffusion_model.eval()

    # Transition1x dataset
    dataset = TransitionPathDataset( "test", data_dir )
    loader = DataLoader( dataset, batch_size=1, shuffle=True, collate_fn=collate_molecules )

    n_reactions = 5
    n_samples = 20
    for reaction_idx, (xA, xB, s, x_ref) in enumerate(loader):
        if reaction_idx >= n_reactions:
            break

        xA = xA.to(device=device, dtype=dtype)
        xB = xB.to(device=device, dtype=dtype)
        s = s.to(device=device, dtype=dtype).flatten()
        x_ref = x_ref.to(device=device, dtype=dtype)

        x_newton, _ = newton_model(xA, xB, s)

        newton_rmse = rmse(x_newton.x, x_ref)

        best_rmse = float("inf")

        for _ in range(n_samples):
            x_sample = sample_path(diffusion_model, xA, xB, s, x_newton, residual_scale, T)
            sample_rmse = rmse(x_sample, x_ref)
            print( sample_rmse)

            if sample_rmse < best_rmse:
                best_rmse = sample_rmse

        print(f"Reaction {reaction_idx}: Newton RMSE={newton_rmse:.4f}, best diffusion RMSE={best_rmse:.4f}")


if __name__ == "__main__":
    main()