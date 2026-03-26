from pathlib import Path
import json
import numpy as np
import torch as pt

from system import build_alanine_dipeptide_simulation

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


def run_neb_family_from_file(
    rep_file: Path,
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
    outdir: Path = Path("outputs/neb"),
):
    """
    Compute NEB trajectories for all representative pairs between basin_A and basin_B.

    Both directions are saved explicitly:
        A_i -> B_j
        B_j -> A_i
    """

    reps_A = load_representatives(rep_file, basin_A, max_n=max_reps_A)
    reps_B = load_representatives(rep_file, basin_B, max_n=max_reps_B)

    nA = reps_A.shape[0]
    nB = reps_B.shape[0]

    family_dir = outdir / f"{basin_A}__{basin_B}"
    family_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {nA} representatives for {basin_A}")
    print(f"Loaded {nB} representatives for {basin_B}")
    print(f"Computing {nA * nB} NEBs and saving both directions")

    # Build one OpenMM simulation and evaluator
    simulation, topology, positions = build_alanine_dipeptide_simulation(
        temperature=300.0,
        friction=1.0,
        timestep_fs=2.0,
        platform_name="CPU",
        minimize=False,
    )
    evaluator = OpenMMEnergyForceEvaluator(simulation)

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

            # Save forward direction explicitly
            meta_fwd = {
                "start_basin": basin_A,
                "end_basin": basin_B,
                "start_rep": i,
                "end_rep": j,
                "direction": "forward",
                "N_images": int(x_opt.shape[0]),
                "k": float(k),
                "n_steps": int(n_steps),
                "lr": float(lr),
                "F_opt": float(F_opt),
            }
            save_neb_result(
                family_dir / f"{basin_A}_{i:02d}__{basin_B}_{j:02d}.npz",
                x0=x0,
                x_opt=x_opt,
                F_opt=F_opt,
                meta=meta_fwd,
            )

            # Save reverse direction explicitly
            x0_rev = pt.flip(x0, dims=[0])
            x_opt_rev = pt.flip(x_opt, dims=[0])

            meta_rev = {
                "start_basin": basin_B,
                "end_basin": basin_A,
                "start_rep": j,
                "end_rep": i,
                "direction": "reverse_from_forward",
                "N_images": int(x_opt_rev.shape[0]),
                "k": float(k),
                "n_steps": int(n_steps),
                "lr": float(lr),
                "F_opt": float(F_opt),
            }
            save_neb_result(
                family_dir / f"{basin_B}_{j:02d}__{basin_A}_{i:02d}.npz",
                x0=x0_rev,
                x_opt=x_opt_rev,
                F_opt=F_opt,
                meta=meta_rev,
            )

    print("Done.")

if __name__ == "__main__":
    run_neb_family_from_file(
        rep_file=Path("outputs/endpoint_representatives.npz"),
        basin_A="right_lower",
        basin_B="right_upper",
        max_reps_A=15,
        max_reps_B=15,
        N_images_minus_1=16,
        k=5.0,
        n_steps=2000,
        lr=1e-3,
        device="cpu",
        outdir=Path("outputs/neb"),
    )