import os
import h5py
import numpy as np
import torch as pt
import pickle

from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

from chemdm.Trajectory import Trajectory

import matplotlib.pyplot as plt

from typing import List, Set

def local_minima_indices( d : np.ndarray,
                          drop_fraction : float = 0.85,
                        ):
    """
    Find local minima that represent a significant drop from the preceding peak.
    """
    candidates = np.where( (d[1:-1] < d[:-2]) & (d[1:-1] <= d[2:]) )[0] + 1

    accepted = []
    prev_min_idx = 0
    for idx in candidates:
        peak = np.max( d[prev_min_idx:idx] ) if idx > prev_min_idx else d[idx]
        if d[idx] < drop_fraction * peak:
            accepted.append( idx )
            prev_min_idx = idx
    return np.array( accepted, dtype=int )

def normalized_arclength(tp: np.ndarray) -> np.ndarray:
    """
    tp: (n_points, n_atoms, 3)
        Discrete path segment including xA and xB.

    Returns
    -------
    s : (n_points,)
        Normalized arclength parameter with s[0] = 0 and s[-1] = 1.
    """
    assert tp.ndim == 3, f"Expected shape (n_points, n_atoms, 3), got {tp.shape}"

    # differences between consecutive path images
    dX = tp[1:,:,:] - tp[:-1,:,:]  # (n_points-1, n_atoms, 3)

    # Euclidean distance in full configuration space R^{3N}
    ds = np.linalg.norm(dX.reshape(dX.shape[0], -1), axis=1)   # (n_points-1,)
    s = np.concatenate([[0.0], np.cumsum(ds)])
    if s[-1] > 0:
        s = s / s[-1]

    return s

def xyz_to_rdkit_mol(atomic_numbers, positions, charge=0):
    """
    atomic_numbers: (n_atoms,)
    positions:      (n_atoms, 3)
    """
    atomic_numbers = np.asarray(atomic_numbers, dtype=int)
    positions = np.asarray(positions, dtype=float)

    mol = Chem.RWMol()
    for z in atomic_numbers:
        mol.AddAtom(Chem.Atom(int(z)))

    conf = Chem.Conformer(len(atomic_numbers))
    for i, (x, y, z) in enumerate(positions):
        conf.SetAtomPosition(i, (float(x), float(y), float(z)))

    mol = mol.GetMol()
    mol.AddConformer(conf)

    # infer connectivity and bond orders from 3D geometry
    rdDetermineBonds.DetermineConnectivity(mol)
    #rdDetermineBonds.DetermineBondOrders(mol, charge=charge)

    return mol

def bond_set(mol):
    return {
        tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx())))
        for b in mol.GetBonds()
    }

MAX_COORD = {
    1: 1,   # H
    6: 4,   # C
    7: 4,   # N, allow 4 to be safe
    8: 2,   # O
}

def toBondStructure( bonds : List ) -> pt.Tensor:
    """
    Convert set-type bond structure to a two-dimensional torch tensor.
    """
    bond_structure = []
    for bond in bonds:
        i,j = bond
        bond_structure.append( [i,j] )
        bond_structure.append( [j,i] )
    return pt.unique( pt.tensor( bond_structure ), dim=0 )

def prune_bonds_by_valence(bonds, x, z):
    bonds = set(tuple(sorted(b)) for b in bonds)
    changed = True

    while changed:
        changed = False

        # build neighbor lists
        neighbors = {i: [] for i in range(len(z))}
        for i, j in bonds:
            d = np.linalg.norm(x[i] - x[j])
            neighbors[i].append((j, d))
            neighbors[j].append((i, d))

        for i in range(len(z)):
            max_deg = MAX_COORD.get(int(z[i]), 10)
            if len(neighbors[i]) > max_deg:
                # remove the longest bond attached to atom i
                j_remove, _ = max(neighbors[i], key=lambda t: t[1])
                bonds.remove(tuple(sorted((i, j_remove))))
                changed = True
                break

    return bonds

# Download the data from huggingface
plot_trajectories = False
data_directory = "/Users/hannesvdc/transition1x/"
store_directory = data_directory + "processed/"
with h5py.File( os.path.join(data_directory, "Transition1x.h5"), "r") as f:
    for evaltype in ['train', 'test', 'val']:
        data = f[evaltype]
        molecules = list(data.keys())

        # Count total reactions for progress display
        total_reactions = sum( len(data[mol].keys()) for mol in molecules )

        storage_counter = 0
        reaction_counter = -1
        for molecule in molecules:
            print('molecule', molecule)
            molecule_data = data[molecule]
            reactions = molecule_data.keys()

            for reaction in reactions:
                reaction_counter += 1

                reaction_data = molecule_data[reaction]
                Z = reaction_data['atomic_numbers'][:]
                positions = reaction_data['positions']
                product = reaction_data['product']
                reactant = reaction_data['reactant']
                ts = reaction_data['transition_state']
                
                positions_data = np.array(positions[:])
                energies = np.array(reaction_data['wB97x_6-31G(d).energy'][:])
                forces = np.array(reaction_data['wB97x_6-31G(d).forces'][:])
                
                # Do some essential checks
                assert (reactant['atomic_numbers'][:] == product['atomic_numbers'][:]).all()
                xA = np.asarray( reactant['positions'][:][0] )
                assert np.allclose( xA, positions_data[0,:,:])
                xB = np.asarray( product['positions'][:][0] )
                matches = np.array([np.allclose(frame, xA, rtol=1e-3, atol=1e-3) for frame in positions_data])
                assert np.sum(matches) == 1

                # Infer chemical bonds from positions
                molA = xyz_to_rdkit_mol(Z, xA, charge=0)
                bonds_A = bond_set( molA )
                pruned_bonds_A = prune_bonds_by_valence( bonds_A, xA, Z )
                molB = xyz_to_rdkit_mol(Z, xB, charge=0)
                bonds_B = bond_set( molB )
                pruned_bonds_B = prune_bonds_by_valence( bonds_B, xB, Z )

                # This positions array contains many NEB paths, but the segments are monotonic. 
                # Find the indices of points closest to xA, they are the initial points along the trajectory.
                distance_from_reactant = np.linalg.norm( (positions_data - xA[np.newaxis,:,:]).reshape(positions_data.shape[0], -1), axis=1)
                distance_from_product = np.linalg.norm( (positions_data - xB[np.newaxis,:,:]).reshape(positions_data.shape[0], -1), axis=1)
                indices = local_minima_indices( distance_from_reactant )
                indices = np.insert( indices, 0, [1] )
                indices = indices[-10:] # only keep the last 10 trajectories
                contains_long_path = (max(indices[1:] - indices[0:-1]) > 15) or ((len(distance_from_reactant) - indices[-1]) > 15)
                non_overlapping = (np.max(distance_from_product[np.min(indices):]) < np.min(distance_from_reactant[np.min(indices)]))

                # Flag reactions where any split point is far from xA
                max_split_distance = np.max( distance_from_reactant[indices] )
                split_distance_threshold = 1.0
                # print(storage_counter, contains_long_path, non_overlapping, np.max(distance_from_product[np.min(indices):]), np.min(distance_from_reactant[np.min(indices):]))
                is_flagged = contains_long_path or non_overlapping #or (max_split_distance > split_distance_threshold)

                if plot_trajectories and is_flagged:
                    fig, axes = plt.subplots( 3, 1, figsize=(10, 10), sharex=True )

                    # Panel 1: distance from xA and xB
                    ax = axes[0]
                    ax.plot( distance_from_reactant, color="tab:blue", lw=0.8, label=r"$\|x - x_A\|$" )
                    ax.plot( distance_from_product, color="tab:orange", lw=0.8, alpha=0.5, label=r"$\|x - x_B\|$" )
                    for idx_val in indices:
                        ax.axvline( idx_val, color="red", ls="--", lw=0.6, alpha=0.7 )
                    ax.scatter( indices, distance_from_reactant[indices], color="red", s=30, zorder=5, label="Split points" )
                    ax.set_ylabel( "Distance (flattened Euclidean)" )
                    ax.set_title( f"[{storage_counter}/{total_reactions}] {evaltype} / {molecule} / {reaction}  "
                                  f"({len(indices)} traj, max split dist: {max_split_distance:.2f})" )
                    ax.legend()

                    # Panel 2: energy per frame
                    ax = axes[1]
                    E_A = reactant['wB97x_6-31G(d).energy'][:][0]
                    ax.plot( (energies - E_A) * 27.211, color="tab:green", lw=0.8 )
                    for idx_val in indices:
                        ax.axvline( idx_val, color="red", ls="--", lw=0.6, alpha=0.7 )
                    ax.set_ylabel( "Energy relative to reactant [eV]" )

                    # Panel 3: max force norm per frame
                    ax = axes[2]
                    max_force_norm = np.max( np.linalg.norm(forces, axis=2), axis=1 )
                    ax.plot( max_force_norm, color="tab:purple", lw=0.8 )
                    for idx_val in indices:
                        ax.axvline( idx_val, color="red", ls="--", lw=0.6, alpha=0.7 )
                    ax.set_ylabel( "Max atomic force norm [Ha/Å]" )
                    ax.set_xlabel( "Frame index" )

                    plt.tight_layout()
                    plt.show()

                #
                if is_flagged:
                    print('Not storing Flagged reaction', reaction_counter)
                    continue

                reaction_trajectories = []
                for t_idx in range( len(indices) ):
                    start_idx = indices[t_idx]
                    end_idx = indices[t_idx+1] if t_idx+1 < len(indices) else positions_data.shape[0]
                    if t_idx == 0:
                        end_idx = end_idx-1
                    tp = positions_data[start_idx:end_idx,:,:]
                    tp = np.concatenate( (xA[np.newaxis,:,:], tp, xB[np.newaxis,:,:]), axis=0)

                    # Make center of mass zero
                    xA = xA - np.mean( xA, axis=0, keepdims=True )
                    xB = xB - np.mean( xB, axis=0, keepdims=True )
                    tp = tp - np.mean( tp, axis=1, keepdims=True )

                    # Compute the normalized arclengths
                    s = normalized_arclength( tp )

                    # Store as a trajectory object for easy loading.
                    trajectory = Trajectory( pt.tensor(Z, dtype=pt.long), 
                                             pt.tensor(xA), 
                                             pt.tensor(xB),
                                             toBondStructure( pruned_bonds_A ),
                                             toBondStructure( pruned_bonds_B ),
                                             pt.tensor(s),
                                             pt.tensor(tp) )
                    reaction_trajectories.append( trajectory )

                # Save the trajectories for this reaction.
                with open( os.path.join(store_directory, f"{evaltype}_reaction_{reaction_counter}.pkl"), "wb") as sf:
                   pickle.dump( reaction_trajectories, sf )
                storage_counter += 1
        print( f"Number of {evaltype} reactions store: {storage_counter} / {reaction_counter}" )
