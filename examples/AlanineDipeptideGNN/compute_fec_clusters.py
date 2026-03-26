from pathlib import Path
import numpy as np
import mdtraj as md
import matplotlib.pyplot as plt

from endpoint_selection import BasinCircle, select_basin_representatives

def wrap_to_pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi

def load_alanine_dipeptide_dataset(
    outdir: Path,
    torsion_angles: np.ndarray,
    burnin_fraction: float = 0.2,
    phi_atoms=(4, 6, 8, 14),
    psi_atoms=(6, 8, 14, 16),
):
    """
    Load all seeded alanine-dipeptide trajectories, discard per-trajectory burn-in,
    compute phi/psi, and merge everything into one dataset.

    Returns
    -------
    xyz : (N, n_atoms, 3)
    phi : (N,)
    psi : (N,)
    seed_labels : (N, 2)
        Initial (phi0, psi0) seed angles in degrees for each frame.
    """
    xyz_list = []
    phi_list = []
    psi_list = []
    seed_label_list = []

    phi_atoms = np.array([phi_atoms], dtype=int)
    psi_atoms = np.array([psi_atoms], dtype=int)

    top_path = outdir / "topology.pdb"

    jobs = [(phi0, psi0) for phi0 in torsion_angles for psi0 in torsion_angles]

    for phi0, psi0 in jobs:
        traj_path = outdir / f"traj_phi={phi0:+06.1f}_psi={psi0:+06.1f}.dcd"
        if not traj_path.exists():
            print(f"Skipping missing file: {traj_path.name}")
            continue

        try:
            traj = md.load(str(traj_path), top=str(top_path))
        except:
            continue

        n_burn = int(burnin_fraction * traj.n_frames)
        traj = traj[n_burn:]

        if traj.n_frames == 0:
            print(f"Skipping empty post-burnin trajectory: {traj_path.name}")
            continue

        phi = md.compute_dihedrals(traj, phi_atoms)[:, 0]
        psi = md.compute_dihedrals(traj, psi_atoms)[:, 0]

        phi = wrap_to_pi(phi)
        psi = wrap_to_pi(psi)

        xyz_list.append(traj.xyz)
        phi_list.append(phi)
        psi_list.append(psi)

        labels = np.tile(np.array([[phi0, psi0]], dtype=float), (traj.n_frames, 1))
        seed_label_list.append(labels)

        print(f"Loaded {traj_path.name}: {traj.n_frames} frames after burn-in")

    if len(xyz_list) == 0:
        raise RuntimeError("No trajectories were loaded.")

    xyz = np.concatenate(xyz_list, axis=0)
    phi = np.concatenate(phi_list, axis=0)
    psi = np.concatenate(psi_list, axis=0)
    seed_labels = np.concatenate(seed_label_list, axis=0)

    return xyz, phi, psi, seed_labels

def compute_clustering():

    outdir = Path("outputs")
    torsion_angles = np.linspace(-180, 180, 10, endpoint=False)

    xyz, phi, psi, seed_labels = load_alanine_dipeptide_dataset(
        outdir=outdir,
        torsion_angles=torsion_angles,
        burnin_fraction=0.2,
    )

    basins = [
        BasinCircle("left_wrap",     phi0=np.deg2rad(-140), psi0=np.deg2rad(160),  radius=np.deg2rad(35)),
        BasinCircle("left_center",   phi0=np.deg2rad(-70),  psi0=np.deg2rad(-50),  radius=np.deg2rad(30)),
        BasinCircle("right_lower",   phi0=np.deg2rad(50),   psi0=np.deg2rad(-100), radius=np.deg2rad(30)),
        BasinCircle("right_upper",   phi0=np.deg2rad(50),   psi0=np.deg2rad(50),   radius=np.deg2rad(30)),
    ]

    results = select_basin_representatives(
        xyz=xyz,
        phi=phi,
        psi=psi,
        basins=basins,
        energies=None,
        atom_indices=None,
        rmsd_threshold=0.03,
        min_cluster_size=3,
        max_reps_per_basin=15,
        representative_mode="medoid",
        max_candidates_per_basin=2000,
        rng_seed=0,
    )

    for basin_name, info in results.items():
        print(f"{basin_name}:")
        print("  frames:", len(info["frame_indices"]))
        print("  representatives:", info["representative_indices"])

        rep_idx = info["representative_indices"]
        np.save( f"outputs/{basin_name}_representatives.npy", xyz[rep_idx,:,:])

    # labels are the same in every basin result entry, so just grab them once
    labels = next(iter(results.values()))["labels"]

    plt.figure(figsize=(8, 7))

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]

    # Plot each basin in a different color
    for b_idx, basin in enumerate(basins):
        frame_idx = np.where(labels == b_idx)[0]
        if len(frame_idx) == 0:
            continue

        plt.scatter(
            np.degrees(phi[frame_idx]),
            np.degrees(psi[frame_idx]),
            s=5,
            alpha=0.35,
            color=colors[b_idx % len(colors)],
            label=basin.name,
        )

    # Optional: plot representatives on top in black-edged markers
    markers = ["o", "s", "^", "D", "P", "X"]
    for b_idx, basin in enumerate(basins):
        rep_idx = results[basin.name]["representative_indices"]
        if len(rep_idx) == 0:
            continue

        plt.scatter(
            np.degrees(phi[rep_idx]),
            np.degrees(psi[rep_idx]),
            s=80,
            color=colors[b_idx % len(colors)],
            marker=markers[b_idx % len(markers)],
            edgecolors="black",
            linewidths=1.0,
        )

    # Optional: show unassigned points in light gray
    unassigned_idx = np.where(labels == -1)[0]
    if len(unassigned_idx) > 0:
        plt.scatter(
            np.degrees(phi[unassigned_idx]),
            np.degrees(psi[unassigned_idx]),
            s=3,
            alpha=0.15,
            color="lightgray",
            label="unassigned",
        )

    plt.xlim(-180, 180)
    plt.ylim(-180, 180)
    plt.xlabel(r"$\phi$ [deg]")
    plt.ylabel(r"$\psi$ [deg]")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    compute_clustering()