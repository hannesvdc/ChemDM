from pathlib import Path
import json
import numpy as np
import torch as pt
import matplotlib.pyplot as plt

from system import build_alanine_dipeptide_simulation, compute_phi_psi_from_xyz
from endpoint_selection import BasinCircle

from chemdm.NEBCartesian import OpenMMEnergyForceEvaluator, computeMEP_openmm_cartesian

def load_representatives(outdir: Path, basin_name: str, max_n: int | None = None) -> np.ndarray:
    """
    Load representative coordinates for one basin from:
        outputs/{basin_name}_representatives.npy

    Returns
    -------
    xyz : (n_reps, n_atoms, 3) numpy array in nm
    """
    path = outdir / f"{basin_name}_representatives.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing representative file: {path}")

    xyz = np.load(path)
    if max_n is not None:
        xyz = xyz[:max_n]
    return xyz


def save_neb_result(
    save_path: Path,
    x0: pt.Tensor,
    x_opt: pt.Tensor,
    F_opt: float,
    meta: dict,
):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        save_path,
        x0=x0.detach().cpu().numpy(),
        x_opt=x_opt.detach().cpu().numpy(),
        F_opt=np.array(F_opt, dtype=np.float64),
        metadata_json=np.array(json.dumps(meta)),
    )

def run_neb_family(
    outdir: Path,
    basin_A: str,
    basin_B: str,
    *,
    max_reps_A: int = 15,
    max_reps_B: int = 15,
    N_images_minus_1: int = 16,
    k: float = 5.0,
    n_steps: int = 2000,
    lr: float = 1e-3,
    device: str = "cpu",
):
    """
    Compute NEB trajectories for all representative pairs between basin_A and basin_B.

    Save all forward and reverse paths into one single .npz file.
    """

    reps_A = load_representatives(outdir, basin_A, max_n=max_reps_A)
    reps_B = load_representatives(outdir, basin_B, max_n=max_reps_B)

    nA = reps_A.shape[0]
    nB = reps_B.shape[0]

    print(f"Loaded {nA} representatives for {basin_A}")
    print(f"Loaded {nB} representatives for {basin_B}")
    print(f"Computing {nA * nB} NEBs and storing both directions in one file")

    simulation, topology, positions = build_alanine_dipeptide_simulation(
        temperature=300.0,
        friction=1.0,
        timestep_fs=2.0,
        platform_name="CPU",
        minimize=False,
    )
    evaluator = OpenMMEnergyForceEvaluator(simulation)

    x0_all = []
    xopt_all = []
    Fopt_all = []

    start_basin_all = []
    end_basin_all = []
    start_rep_all = []
    end_rep_all = []
    direction_all = []

    for i in range(nA):
        for j in range(nB):
            print(f"[{i+1:02d}/{nA}] x [{j+1:02d}/{nB}]   {basin_A}_{i:02d} <-> {basin_B}_{j:02d}")

            xA_xyz = pt.tensor(reps_A[i], dtype=pt.float64, device=device)
            xB_xyz = pt.tensor(reps_B[j], dtype=pt.float64, device=device)

            x0, x_opt, F_opt = computeMEP_openmm_cartesian(
                evaluator=evaluator,
                xA_xyz=xA_xyz,
                xB_xyz=xB_xyz,
                N=N_images_minus_1,
                k=k,
                n_steps=n_steps,
                lr=lr,
                verbose=False,
                generate_initial_path=None,
                project_band=True,
            )

            x0_np = x0.detach().cpu().numpy()
            xopt_np = x_opt.detach().cpu().numpy()

            # forward
            x0_all.append(x0_np)
            xopt_all.append(xopt_np)
            Fopt_all.append(F_opt)
            start_basin_all.append(basin_A)
            end_basin_all.append(basin_B)
            start_rep_all.append(i)
            end_rep_all.append(j)
            direction_all.append("forward")

            # reverse
            x0_all.append(x0_np[::-1].copy())
            xopt_all.append(xopt_np[::-1].copy())
            Fopt_all.append(F_opt)
            start_basin_all.append(basin_B)
            end_basin_all.append(basin_A)
            start_rep_all.append(j)
            end_rep_all.append(i)
            direction_all.append("reverse_from_forward")

    x0_all = np.stack(x0_all, axis=0)
    xopt_all = np.stack(xopt_all, axis=0)
    Fopt_all = np.asarray(Fopt_all, dtype=np.float64)

    start_basin_all = np.asarray(start_basin_all, dtype=object)
    end_basin_all = np.asarray(end_basin_all, dtype=object)
    start_rep_all = np.asarray(start_rep_all, dtype=np.int32)
    end_rep_all = np.asarray(end_rep_all, dtype=np.int32)
    direction_all = np.asarray(direction_all, dtype=object)

    save_path = outdir / f"{basin_A}__{basin_B}__neb_dataset.npz"
    np.savez(
        save_path,
        x0=x0_all,
        x_opt=xopt_all,
        F_opt=Fopt_all,
        start_basin=start_basin_all,
        end_basin=end_basin_all,
        start_rep=start_rep_all,
        end_rep=end_rep_all,
        direction=direction_all,
        basin_A=np.array(basin_A, dtype=object),
        basin_B=np.array(basin_B, dtype=object),
        k=np.array(k, dtype=np.float64),
        n_steps=np.array(n_steps, dtype=np.int32),
        lr=np.array(lr, dtype=np.float64),
    )

    print(f"Saved dataset to {save_path}")
    print("Done.")

    # --------------------------------------------------------
    # Quick plot of all optimized NEB trajectories in (phi, psi)
    # --------------------------------------------------------
    plt.figure(figsize=(7, 6))
    for path_xyz in xopt_all:
        phi_path, psi_path = compute_phi_psi_from_xyz(path_xyz)
        plt.plot(np.degrees(phi_path), np.degrees(psi_path), "-o", ms=3, alpha=0.7)

    for basin in [
        BasinCircle("left_wrap",   phi0=np.deg2rad(-140), psi0=np.deg2rad(160),  radius=np.deg2rad(35)),
        BasinCircle("left_center", phi0=np.deg2rad(-70),  psi0=np.deg2rad(-50),  radius=np.deg2rad(30)),
        BasinCircle("right_lower", phi0=np.deg2rad(50),   psi0=np.deg2rad(-100), radius=np.deg2rad(30)),
        BasinCircle("right_upper", phi0=np.deg2rad(50),   psi0=np.deg2rad(50),   radius=np.deg2rad(30)),
    ]:
        circ = plt.Circle((np.degrees(basin.phi0), np.degrees(basin.psi0)),# type: ignore
                          np.degrees(basin.radius), fill=False, ls="--", lw=1.2, color="black")
        plt.gca().add_patch(circ)

    plt.xlim(-180, 180)
    plt.ylim(-180, 180)
    plt.xlabel(r"$\phi$ [deg]")
    plt.ylabel(r"$\psi$ [deg]")
    plt.tight_layout()
    plt.savefig(outdir / f"{basin_A}__{basin_B}__neb_paths.png", dpi=200)
    plt.close()

if __name__ == "__main__":
    run_neb_family(
        outdir=Path("outputs"),
        basin_A="right_lower",
        basin_B="right_upper",
        max_reps_A=15,
        max_reps_B=15,
        N_images_minus_1=100,
        k=500.0,
        n_steps=2000,
        lr=1e-3,
        device="cpu",
    )