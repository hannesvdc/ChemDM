import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict

def wrap_to_pi(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def angular_difference(a: np.ndarray, b: float) -> np.ndarray:
    """
    Smallest signed angular difference a - b, wrapped to [-pi, pi).
    """
    return wrap_to_pi(a - b)

@dataclass
class BasinCircle:
    name: str
    phi0: float      # radians
    psi0: float      # radians
    radius: float    # radians

def assign_basins_circles(
    phi: np.ndarray,
    psi: np.ndarray,
    basins: List[BasinCircle],
) -> np.ndarray:
    """
    Assign each (phi, psi) point to the first basin circle that contains it.
    Returns:
        labels: shape (n_frames,), integer basin index or -1 if none
    """
    n = len(phi)
    labels = -np.ones(n, dtype=int)

    for b_idx, basin in enumerate(basins):
        dphi = angular_difference(phi, basin.phi0)
        dpsi = angular_difference(psi, basin.psi0)
        dist = np.sqrt(dphi**2 + dpsi**2)

        mask = (dist <= basin.radius) & (labels == -1)
        labels[mask] = b_idx

    return labels


# ============================================================
# Cartesian alignment / RMSD
# ============================================================

def kabsch_align(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    Align P onto Q using Kabsch.
    P, Q: (n_atoms, 3)
    Returns aligned P.
    """
    Pc = P - P.mean(axis=0, keepdims=True)
    Qc = Q - Q.mean(axis=0, keepdims=True)

    C = Pc.T @ Qc
    V, S, Wt = np.linalg.svd(C)

    # Reflection correction
    d = np.sign(np.linalg.det(V @ Wt))
    D = np.diag([1.0, 1.0, d])

    R = V @ D @ Wt
    P_aligned = Pc @ R

    return P_aligned

def rmsd_aligned(P: np.ndarray, Q: np.ndarray) -> float:
    """
    RMSD after aligning P to Q.
    """
    P_aligned = kabsch_align(P, Q)
    Qc = Q - Q.mean(axis=0, keepdims=True)
    diff = P_aligned - Qc
    return np.sqrt(np.mean(np.sum(diff**2, axis=1)))

def pairwise_rmsd_matrix(xyz: np.ndarray) -> np.ndarray:
    """
    xyz: (n_frames, n_atoms, 3)
    Returns:
        D: (n_frames, n_frames) RMSD matrix
    """
    n = xyz.shape[0]
    D = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            d = rmsd_aligned(xyz[i], xyz[j])
            D[i, j] = d
            D[j, i] = d

    return D

# ============================================================
# Greedy RMSD clustering
# ============================================================

def greedy_rmsd_clustering(
    D: np.ndarray,
    threshold: float,
) -> List[List[int]]:
    """
    Greedy clustering based on RMSD threshold.

    Procedure:
      - take first unassigned frame as cluster center
      - assign all frames within threshold to that cluster
      - repeat

    D: pairwise RMSD matrix
    threshold: RMSD cutoff (same length units as xyz; for OpenMM/MDTraj xyz this is nm)

    Returns:
        list of clusters, each cluster is a list of frame indices
    """
    n = D.shape[0]
    unassigned = set(range(n))
    clusters = []

    while unassigned:
        seed = min(unassigned)
        cluster = [j for j in unassigned if D[seed, j] <= threshold]
        for j in cluster:
            unassigned.remove(j)
        clusters.append(cluster)

    return clusters


def medoid_of_cluster(D: np.ndarray, cluster: List[int]) -> int:
    """
    Return the medoid index of a cluster.
    """
    if len(cluster) == 1:
        return cluster[0]

    subD = D[np.ix_(cluster, cluster)]
    total = np.sum(subD, axis=1)
    return cluster[int(np.argmin(total))]


def representative_of_cluster(
    D: np.ndarray,
    cluster: List[int],
    energies: Optional[np.ndarray] = None,
    mode: str = "medoid",
) -> int:
    """
    Choose representative of a cluster.

    mode:
      - 'medoid'      : structural medoid
      - 'lowest_energy': member with lowest supplied energy
    """
    if mode == "medoid" or energies is None:
        return medoid_of_cluster(D, cluster)
    elif mode == "lowest_energy":
        subE = energies[cluster]
        return cluster[int(np.argmin(subE))]
    else:
        raise ValueError(f"Unknown representative mode: {mode}")


# ============================================================
# Endpoint selection pipeline
# ============================================================

def select_basin_representatives(
    xyz: np.ndarray,
    phi: np.ndarray,
    psi: np.ndarray,
    basins: List[BasinCircle],
    energies: Optional[np.ndarray] = None,
    atom_indices: Optional[np.ndarray] = None,
    rmsd_threshold: float = 0.03,
    min_cluster_size: int = 1,
    max_reps_per_basin: Optional[int] = None,
    representative_mode: str = "medoid",
    max_candidates_per_basin: Optional[int] = 500,
    rng_seed: int = 0,
) -> Dict[str, Dict]:
    """
    Select representative endpoint conformers in each basin.

    Parameters
    ----------
    xyz : (n_frames, n_atoms, 3)
        Cartesian coordinates, ideally already minimized or locally relaxed.
    phi, psi : (n_frames,)
        Torsion angles in radians.
    basins : list[BasinCircle]
        Basin definitions in torsion space.
    energies : optional (n_frames,)
        Optional energies per frame. Useful for lowest-energy representative selection.
    atom_indices : optional array of atom indices
        If provided, RMSD clustering uses only these atoms.
    rmsd_threshold : float
        RMSD clustering threshold in nm.
    min_cluster_size : int
        Discard clusters smaller than this.
    max_reps_per_basin : optional int
        Keep only the largest N clusters per basin.
    representative_mode : str
        'medoid' or 'lowest_energy'
    max_candidates_per_basin : optional int
        Randomly subsample candidates within each basin before RMSD matrix construction.
    rng_seed : int
        Random seed for reproducibility.

    Returns
    -------
    result : dict
        result[basin_name] contains:
          - frame_indices
          - representative_indices
          - clusters
          - labels
    """
    rng = np.random.default_rng(rng_seed)
    labels = assign_basins_circles(phi, psi, basins)

    if atom_indices is None:
        xyz_used = xyz
    else:
        xyz_used = xyz[:, atom_indices, :]

    results = {}

    for b_idx, basin in enumerate(basins):
        frame_indices = np.where(labels == b_idx)[0]

        if len(frame_indices) == 0:
            results[basin.name] = {
                "frame_indices": np.array([], dtype=int),
                "representative_indices": np.array([], dtype=int),
                "clusters": [],
                "labels": labels,
            }
            continue

        # Optional subsampling for computational tractability
        if max_candidates_per_basin is not None and len(frame_indices) > max_candidates_per_basin:
            frame_indices = rng.choice(frame_indices, size=max_candidates_per_basin, replace=False)
            frame_indices = np.sort(frame_indices)

        xyz_b = xyz_used[frame_indices]

        D = pairwise_rmsd_matrix(xyz_b)
        clusters_local = greedy_rmsd_clustering(D, threshold=rmsd_threshold)

        # Filter by cluster size
        clusters_local = [c for c in clusters_local if len(c) >= min_cluster_size]

        # Sort by size descending
        clusters_local = sorted(clusters_local, key=len, reverse=True)

        if max_reps_per_basin is not None:
            clusters_local = clusters_local[:max_reps_per_basin]

        # Convert local cluster indices back to global frame indices
        clusters_global = [[frame_indices[i] for i in c] for c in clusters_local]

        rep_indices = []
        for c_local, c_global in zip(clusters_local, clusters_global):
            if energies is None:
                rep_local = representative_of_cluster(D, c_local, energies=None, mode="medoid")
            else:
                rep_local = representative_of_cluster(
                    D, c_local, energies=energies[frame_indices], mode=representative_mode
                )
            rep_global = frame_indices[rep_local]
            rep_indices.append(rep_global)

        results[basin.name] = {
            "frame_indices": frame_indices,
            "representative_indices": np.array(rep_indices, dtype=int),
            "clusters": clusters_global,
            "labels": labels,
        }

    return results