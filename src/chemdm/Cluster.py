import numpy as np
from rdkit import Chem

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


def _symmetry_corrected_heavy_rmsd( a_heavy: np.ndarray,
                                    b_heavy: np.ndarray,
                                    permutations: list[np.ndarray],
                                    allow_reflection: bool = False ) -> float:
    """Minimum heavy-atom RMSD between two conformers over the molecule's graph
    automorphisms (each `permutations[i]` a permutation of the heavy atoms, identity
    included). `a_heavy`, `b_heavy` are the heavy-atom coordinates, shape (H, 3).

    Alignment uses `kabsch_align_numpy`, a PROPER rotation. With `allow_reflection`
    False, a permutation that would only superimpose the two under a mirror leaves the
    RMSD high -- enantiomers / diastereomers are never merged (chirality kept). Set
    `allow_reflection` True ONLY for an achiral molecule: reflecting one axis and then
    letting the proper Kabsch rotate composes to the best improper fit, so genuine
    conformational enantiomers (mirror-image, energy-degenerate) are recognised as one.
    """
    signs = (1.0, -1.0) if allow_reflection else (1.0,)
    best = np.inf
    for s in signs:
        a_s = a_heavy.copy(); a_s[:, 0] *= s                     # reflect one axis; proper Kabsch supplies the rotation
        for perm in permutations:
            b_perm = b_heavy[perm]
            a_aligned = kabsch_align_numpy( a_s, b_perm )        # Z=None -> fit on all (heavy) atoms
            r = float( np.sqrt( np.mean( np.sum( (a_aligned - b_perm) ** 2, axis=1 ) ) ) )
            if r < best:
                best = r
                if best < 1e-3:
                    return best                                  # exact symmetry match; can't beat it
    return best


def _heavy_automorphisms( mol ) -> list[np.ndarray]:
    """
    Heavy-atom graph automorphisms of `mol` as index permutations over the heavy
    atoms (identity included).
    """
    heavy_mol = Chem.RemoveHs( mol )
    return [ np.asarray(p) for p in heavy_mol.GetSubstructMatches(
                heavy_mol, uniquify=False, useChirality=False, maxMatches=10000 ) ]


def post_relaxation_clustering( Z: np.ndarray,
                                conformers: list[np.ndarray],
                                energies: list[float],
                                forces: list[float],
                                cluster_sizes: list[int],
                                rmsd_tol: float = 0.5,
                                mol = None,                       # rdkit Chem.Mol (with Hs); enables symmetry dedup
                                energy_tol: float = 1.0,          # kJ/mol; loose pre-filter
                                symmetry_rmsd_tol: float = 0.1,   # Angstrom; the real identity test
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
    mol:
        Optional RDKit molecule (with hydrogens, same atom order as `conformers`).
        When given, conformers with near-identical energy that coincide under a graph
        automorphism (an atom-index shuffle by molecular symmetry) are merged as
        chemically identical -- e.g. a symmetric ring placing a substituent on an
        equivalent site. The automorphisms are listed lazily, only once some conformers
        actually share an energy, so molecules with all-distinct energies pay nothing.
        None disables the symmetry dedup.
    energy_tol:
        Energy window (kJ/mol) gating the symmetry check (a loose pre-filter).
    symmetry_rmsd_tol:
        Heavy-atom RMSD tolerance (Angstrom) for the automorphism-corrected identity
        test. Alignment is a proper rotation, so chirality is preserved.

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
    assert len(energies) == n_confs
    assert len(forces) == n_confs
    assert len(cluster_sizes) == n_confs

    Z = np.asarray(Z)
    heavy = (Z != 1)

    # Sort by energy so each new cluster center is the lowest-energy
    # unassigned conformer in that RMSD basin.
    order = np.argsort(energies)
    optimal_conformers: list[np.ndarray] = []
    optimal_energies: list[float] = []
    optimal_forces: list[float] = []
    representative_indices: list[int] = []
    final_cluster_sizes: list[int] = []
    permutations: list[np.ndarray] | None = None   # heavy-atom automorphisms, listed lazily (see below)
    reflectable = False                            # is the molecule achiral? set lazily with permutations

    for idx_in_original in order:
        x_conf = conformers[idx_in_original]
        x_conf = x_conf - np.mean(x_conf, axis=0, keepdims=True)
        E = float(energies[idx_in_original])

        assigned = False
        for k, reference_conformer in enumerate(optimal_conformers):
            x_conf_aligned = kabsch_align_numpy( x_conf, reference_conformer, Z, )

            diff = x_conf_aligned[heavy, :] - reference_conformer[heavy, :]
            rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
            if rmsd <= rmsd_tol:
                assigned = True
                final_cluster_sizes[k] += cluster_sizes[idx_in_original]
                break

            # Permutation-symmetry dedup: two relaxed conformers with near-identical
            # energy that coincide under a graph automorphism (an atom-index shuffle by
            # molecular symmetry) are chemically identical. Kabsch alignment keeps
            # it chirality-safe.
            if mol is not None and abs(E - optimal_energies[k]) <= energy_tol:
                if permutations is None:
                    permutations = _heavy_automorphisms( mol )
                    reflectable  = len( Chem.FindPotentialStereo(mol) ) == 0   # no stereo elements => achiral
                if _symmetry_corrected_heavy_rmsd( x_conf[heavy, :], reference_conformer[heavy, :], permutations, allow_reflection=reflectable ) <= symmetry_rmsd_tol:
                    assigned = True
                    final_cluster_sizes[k] += cluster_sizes[idx_in_original]
                    break

        if not assigned:
            optimal_conformers.append(x_conf)
            optimal_energies.append(E)
            optimal_forces.append(float(forces[idx_in_original]))
            representative_indices.append(int(idx_in_original))
            final_cluster_sizes.append(int(cluster_sizes[idx_in_original]))

    return optimal_conformers, np.asarray(optimal_energies), np.asarray(optimal_forces), representative_indices, final_cluster_sizes,