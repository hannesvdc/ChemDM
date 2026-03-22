import torch as pt
import numpy as np
from scipy.spatial import cKDTree

from typing import Tuple

@pt.no_grad()
def findAllDistanceNeighbors( x: pt.Tensor,
                              cutoff: float
                            ) -> pt.Tensor:
    """
    Find all atoms within the cutoff distance from each other. Returns a tensor
    of shape (n_neighbors, 2) representing the new connections between atoms 
    within a cutoff distance. The return tensor can also include bonds from the molecule, 
    but it is guaranteed to be symmetric and exclude self-bonds.

    Arguments
    ---------
    x : pt.Tensor
        The positions of the "molecule" for which to find all neighbors within the cutoff distance.
        Can be any size (N, k) as long as it is two-dimensional. 
    cutoff : float
        The cutoff distance used.

    Returns
    -------
    edge_index : Tensor of shape (n_neighbors, 2)
        Represents all new edgs between neighbors. Guaranteed symmetric and
        excluding self-edges.
    """
    device = x.device

    # Move batch separation into an extra coordinate so different molecules
    # cannot become neighbors.
    points = x.detach().cpu().numpy()

    # Construct the KDTree and query for neighbors
    tree = cKDTree(points)
    pairs = tree.query_pairs( r=cutoff, p=2.0)
    pairs = np.array(list( pairs ), dtype=np.int64)

    # Convert to a torch tensor of shape (n_neighbor_pairs, 2)
    if pairs.size == 0:
        neighbor_edge_index = pt.empty((0, 2), dtype=pt.long, device=device)
    else:
        # Add both directions
        rev_pairs = pairs[:, [1, 0]]
        all_pairs = np.concatenate([pairs, rev_pairs], axis=0)
        neighbor_edge_index = pt.from_numpy(all_pairs).to(device=device, dtype=pt.long)

    return neighbor_edge_index

@pt.no_grad()
def mergeBondAndDistanceNeighbors( bond_neighbors : pt.Tensor,
                                   distance_neighbors : pt.Tensor
                                 ) -> Tuple[pt.Tensor, pt.Tensor]:
    """
    Merge edges coming from bonds and closeby atoms. Returns a flag indicating
    if the edge was a bond before merging.

    Arguments
    ---------
    bond_neighbors : pt.Tensor
        Pairs of bonded atoms. Must have shape (E,2).
    distance_neighbors : pt.Tensor
        Pairs of neighboring atoms based on some distance. Must be shape (E,2).

    Returns
    -------
    all_neighbors : Tensor of shape (n_edges, 2)
        Unique directed edges.
    is_bond : Tensor of shape (n_edges,)
        1 if the edge is a bond in `molecule`, 0 otherwise.
    """
    assert bond_neighbors.ndim == 2 and bond_neighbors.shape[1] == 2, \
        f"`bond_neighbors` must have shape (E,2) but got {bond_neighbors.shape}"
    assert distance_neighbors.ndim == 2 and distance_neighbors.shape[1] == 2, \
        f"`distance_neighbors` must have shape (E,2) but got {distance_neighbors.shape}"
    device = bond_neighbors.device
    
    all_edges = pt.cat([bond_neighbors, distance_neighbors], dim=0)
    edge_type = pt.cat([
        pt.ones(bond_neighbors.shape[0], dtype=pt.float32, device=device),
        pt.zeros(distance_neighbors.shape[0], dtype=pt.float32, device=device),
    ])
    all_neighbors, inverse = pt.unique(all_edges, dim=0, return_inverse=True)

    # Flag neighbors that were bonds. `edge_type` has shape E1+E2, and if either 
    # of the duplicates was a bond, "amax" will return 1, otherwise 0.
    is_bond = pt.zeros(all_neighbors.shape[0], dtype=pt.float32, device=device)
    is_bond = pt.scatter_reduce( is_bond, 0, inverse, edge_type, reduce="amax", include_self=False )

    return all_neighbors, is_bond
