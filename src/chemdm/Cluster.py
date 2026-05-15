import numpy as np

from chemdm.geometry import kabsch_align_numpy

def rmsd_clustering( Z : np.ndarray, 
                    conformers : list[np.ndarray], 
                    rmsd_tol : float = 0.5
                    ) -> tuple[list[np.ndarray], list[int], list[int]]:
    if len(conformers) == 0:
        return conformers, list(), list()
    idx = (Z != 1)
    
    initial_conformer = conformers[0] - np.mean( conformers[0], axis=0, keepdims=True )
    optimal_conformers = [ initial_conformer ]
    indices = [ 0 ]
    cluster_size = [ 1 ]
    for n in range(1, len(conformers)):
        x_conf = conformers[n] - np.mean( conformers[n], axis=0, keepdims=True )

        is_new = True
        for k in range( len(optimal_conformers) ):
            reference_conformer = optimal_conformers[k]

            # Center and Kabsch align
            x_conf_aligned = kabsch_align_numpy( x_conf, reference_conformer, Z )

            # Compute the per-atom RMSD. Ignore hydrogens
            rmsd = np.sqrt(np.mean(np.sum( (x_conf_aligned[idx,:] - reference_conformer[idx,:])**2, axis=1 ) ))
            
            if rmsd <= rmsd_tol:
                is_new = False
                cluster_size[ k ] += 1
                break

        if is_new:
            optimal_conformers.append( x_conf )
            indices.append( n )
            cluster_size.append( 1 )

    return optimal_conformers, indices, cluster_size

def post_relaxation_rmsd_clustering( Z: np.ndarray,
                                     conformers: list[np.ndarray],
                                     energies: list[float],
                                     forces: list[float],
                                     cluster_sizes: list[int],
                                     rmsd_tol: float = 0.5,
    ) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, list[int], list[int]]:
    """
    Greedy post-relaxation RMSD clustering. Runtime complexity is O( N log N ) with N the number of conformers.
    Representatives are chosen to be the lowest-energy conformer in each RMSD cluster.

    Parameters
    ----------
    Z:
        Atomic numbers, shape (n_atoms,).
    conformers:
        Relaxed conformers, each shape (n_atoms, 3).
    energies:
        Relaxed conformer energies, shape (n_conformers,).
    forces:
        Relaxed conformer force norms, shape (n_conformers,).
    cluster_sizes:
        Number of original/generated conformers represented by each relaxed conformer.
        If there was no pre-clustering, this should be all ones.
    rmsd_tol:
        Heavy-atom RMSD clustering radius in Angstrom.

    Returns
    -------
    optimal_conformers:
        Lowest-energy representative of each cluster.
    optimal_energies:
        Energy of each representative.
    optimal_forces:
        Force norm of each representative.
    representative_indices:
        Original input index of each representative.
    final_cluster_sizes:
        Total number of conformers assigned to each final cluster.
    """

    n_confs = len(conformers)
    if n_confs == 0:
        return [], np.array([]), np.array([]), [], []

    energies = np.asarray(energies, dtype=float)
    forces = np.asarray(forces, dtype=float)
    cluster_sizes = list(cluster_sizes)
    assert len(energies) == n_confs
    assert len(forces) == n_confs
    assert len(cluster_sizes) == n_confs

    Z = np.asarray(Z)
    heavy = Z != 1

    # Sort by energy so each new cluster center is the lowest-energy
    # unassigned conformer in that RMSD basin.
    order = np.argsort(energies)
    optimal_conformers: list[np.ndarray] = []
    optimal_energies: list[float] = []
    optimal_forces: list[float] = []
    representative_indices: list[int] = []
    final_cluster_sizes: list[int] = []

    for idx_in_original in order:
        x_conf = conformers[idx_in_original]
        x_conf = x_conf - np.mean(x_conf, axis=0, keepdims=True)

        assigned = False
        for k, reference_conformer in enumerate(optimal_conformers):
            x_conf_aligned = kabsch_align_numpy( x_conf, reference_conformer, Z, )

            diff = x_conf_aligned[heavy, :] - reference_conformer[heavy, :]
            rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
            if rmsd <= rmsd_tol:
                assigned = True
                final_cluster_sizes[k] += cluster_sizes[idx_in_original]
                break

        if not assigned:
            optimal_conformers.append(x_conf)
            optimal_energies.append(float(energies[idx_in_original]))
            optimal_forces.append(float(forces[idx_in_original]))
            representative_indices.append(int(idx_in_original))
            final_cluster_sizes.append(int(cluster_sizes[idx_in_original]))

    return optimal_conformers, np.asarray(optimal_energies), np.asarray(optimal_forces), representative_indices, final_cluster_sizes,