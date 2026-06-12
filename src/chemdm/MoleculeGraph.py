import torch as pt

import chemdm.graph.algorithms as alg
import chemdm.graph.rings as rings
from chemdm.graph.kdtree import KDTree

from abc import ABC, abstractmethod
from typing import List, Tuple, Self

# Expose this class to the rest of the package.
RingInformation = rings.RingInfo

class Molecule(ABC):
    @property
    @abstractmethod
    def Z(self) -> pt.Tensor: pass

    @property
    @abstractmethod
    def x(self) -> pt.Tensor: pass

    @property
    @abstractmethod
    def edge_index(self) -> pt.Tensor: pass

    @abstractmethod
    def to( self, device : pt.device, dtype : pt.dtype ) -> Self: pass

    @abstractmethod
    def copyWithNewPositions(self, x : pt.Tensor ) -> Self: pass

class MoleculeGraph( Molecule ):

    def __init__( self, Z : pt.Tensor,
                        x : pt.Tensor,
                        bonds : pt.Tensor  ) -> None:
        Z = Z.flatten()
        bonds = bonds.to( dtype=pt.long )
        assert bonds.ndim == 2 and bonds.shape[1] == 2, f"`bonds` must be an integer tensor with shape `(n_bonds,2)` but got {bonds.shape}"
        assert bonds.numel() == 0 or (pt.min(bonds) >= 0 and pt.max(bonds) < len(Z)), f"Bond indices must be nonnegative and cannot exceed `len(Z)`."
        assert x.ndim == 2 and x.shape[1] == 3, f"`x` must have shape (N_atoms, 3) but got {x.shape}"
        self._Z = Z
        self._x = x
        self._edge_index = bonds
    
    @property
    def Z(self): return self._Z

    @property
    def x(self): return self._x

    @property
    def edge_index(self): return self._edge_index

    def to( self, device=pt.device("cpu"), dtype=pt.float64 ) -> Self:
        self._Z = self._Z.to( device=device )
        self._x = self._x.to( device=device, dtype=dtype )
        self._edge_index = self._edge_index.to( device=device )
        return self

    def copyWithNewPositions(self, x: pt.Tensor) -> Self:
        return type(self)( self._Z, x, self._edge_index )



class BatchedMoleculeGraph( Molecule ):

    def __init__(self, molecules: List[Molecule]) -> None:
        """
        Batch a list of Molecules. If an input molecule is already batched, its internal molecule_id structure
        is preserved, but shifted so that all molecule IDs remain globally unique.

        This code has not been sufficiently vectorized and may therefore be slow for huge numbers of molecules.
        """
        assert len(molecules) > 0

        # Concatenate atomic numbers and positions.
        self._Z = pt.cat([mol.Z for mol in molecules], dim=0)
        self._x = pt.cat([mol.x for mol in molecules], dim=0)
        device = self._x.device

        molecule_ids = []
        edge_list = []
        atom_offset = 0
        molecule_offset = 0
        for mol in molecules:
            n_atoms = mol.Z.shape[0]

            # Case 1: mol is already batched and has molecule_id.
            if isinstance( mol, BatchedMoleculeGraph ):
                local_molecule_id = mol.molecule_id.to(device=device).long()
                shifted_molecule_id = local_molecule_id + molecule_offset
                n_local_molecules = len( pt.unique(local_molecule_id) )

            # Case 2: mol is a single MoleculeGraph.
            else:
                shifted_molecule_id = pt.full( (n_atoms,), molecule_offset, dtype=pt.long, device=device, )
                n_local_molecules = 1
            molecule_ids.append( shifted_molecule_id )

            # Shift edge indices by atom offset.
            edge_index = mol.edge_index.to( device=device ).long()
            edge_list.append(edge_index + atom_offset)

            atom_offset += n_atoms
            molecule_offset += n_local_molecules

        self._molecule_id = pt.cat( molecule_ids, dim=0 )
        self._edge_index = pt.cat( edge_list, dim=0 )

    def to( self, device=pt.device("cpu"), dtype=pt.float64 ) -> Self:
        self._Z = self._Z.to( device=device )
        self._x = self._x.to( device=device, dtype=dtype )
        self._edge_index = self._edge_index.to( device=device )
        self._molecule_id = self._molecule_id.to( device=device )
        return self

    @property
    def Z(self): return self._Z

    @property
    def x(self): return self._x

    @property
    def edge_index(self): return self._edge_index

    @property
    def molecule_id(self): return self._molecule_id

    @classmethod
    def fromRawTensors(cls, Z: pt.Tensor, x: pt.Tensor, edge_index: pt.Tensor, molecule_id : pt.Tensor):
        obj = cls.__new__(cls)
        obj._Z = Z
        obj._x = x
        obj._edge_index = edge_index
        obj._molecule_id = molecule_id
        return obj

    def copyWithNewPositions(self, x: pt.Tensor):
        return BatchedMoleculeGraph.fromRawTensors(self._Z, x, self._edge_index, self._molecule_id)


@pt.no_grad()
def batchMolecules(molecules: List[Molecule]) -> BatchedMoleculeGraph:
    """
    Create a batched molecule from a list of single-molecule graphs. 

    Arguments
    ---------
    molecules : List[Molecule]
        List of single-molecule graphs to merge.

    Returns
    -------
    batch_molecule : BatchedMoleculeGraph
        The large merged molecule.
    """
    return BatchedMoleculeGraph( molecules )

@pt.no_grad()
def assertNoCrossMoleculeEdges( edge_index: pt.Tensor, molecule_id: pt.Tensor ) -> None:
    src = edge_index[:, 0].long()
    dst = edge_index[:, 1].long()

    bad = molecule_id[src] != molecule_id[dst]

    if pt.any(bad):
        bad_edges = edge_index[bad][:10].detach().cpu().tolist()
        raise ValueError(
            "Found edges connecting atoms from different molecules. "
            f"Examples: {bad_edges}"
        )

@pt.no_grad()
def unbatchBatchedMolecule( batched_molecule : Molecule ) -> List[ Molecule ]:
    """
    Unwrap a batched molecule into its constituent molecules. 
    """
    # Return the molecule graph itself if the passed molecule is not of batched type.
    if not isinstance(batched_molecule, BatchedMoleculeGraph):
        return [ batched_molecule ]
    assertNoCrossMoleculeEdges( batched_molecule.edge_index, batched_molecule.molecule_id )
    
    device = batched_molecule.x.device
    
    molecules = []
    molecule_id = batched_molecule.molecule_id
    src_global = batched_molecule.edge_index[:, 0].long()
    dst_global = batched_molecule.edge_index[:, 1].long()
    for mol_id_tensor in pt.unique( molecule_id, sorted=True ):
        mol_id = int( mol_id_tensor.item() )

        atoms_in_molecule = pt.nonzero( molecule_id == mol_id, as_tuple=False ).flatten()
        n_atoms = int( atoms_in_molecule.numel() )
        if n_atoms == 0:
            continue

        # Extract atomic information and positions
        local_Z = batched_molecule.Z[atoms_in_molecule]
        local_x = batched_molecule.x[atoms_in_molecule,:]

        # Extract edges
        edge_in_this_mol = ( (molecule_id[src_global] == mol_id) & (molecule_id[dst_global] == mol_id) )
        local_edges_global = batched_molecule.edge_index[edge_in_this_mol] # includes an offset

        global_to_local = pt.full( (batched_molecule.Z.shape[0],), -1, dtype=pt.long, device=device )
        global_to_local[atoms_in_molecule] = pt.arange( n_atoms, device=device )
        local_edges = pt.stack( [ global_to_local[local_edges_global[:, 0]], global_to_local[local_edges_global[:, 1]] ], dim=1 )
        if local_edges.numel() == 0:
            local_edges = pt.empty((0, 2), dtype=pt.long, device=device)

        # Put everything in one MoleculeGraph
        local_molecule = MoleculeGraph( local_Z, local_x, local_edges )
        molecules.append( local_molecule )

    return molecules


@pt.no_grad()
def findAllDistanceNeighbors( molecule: Molecule,
                              cutoff: float
                            ) -> pt.Tensor:
    """
    Find all atoms within the cutoff distance from each other. Returns a tensor
    of shape (n_neighbors, 2) representing the new connections between atoms 
    within a cutoff distance. The return tensor can also include bonds from the molecule, 
    but it is guaranteed to be symmetric and exclude self-bonds.

    Arguments
    ---------
    molecule : Molecule
        The molecule for which to find all neighbors within the cutoff distance. If 
        molecule if of type BatchedMolecule, a fourth position dimension is added
        to ensure that atoms from different original molecules can never be 
        neighbors.
    cutoff : float
        The cutoff distance used.

    Returns
    -------
    edge_index : Tensor of shape (n_neighbors, 2)
        Represents all new edgs between neighbors. Guaranteed symmetric and
        excluding self-edges.
    """

    # Move batch separation into an extra coordinate so different molecules
    # cannot become neighbors.
    if isinstance( molecule, BatchedMoleculeGraph ):
        x = pt.cat([molecule.x, 2.0 * cutoff * molecule.molecule_id[:,None]], dim=1)
    else:
        x = molecule.x

    # Inline function
    neighbor_edge_index = alg.findAllDistanceNeighbors( x, cutoff )

    return neighbor_edge_index


@pt.no_grad()
def findAllDistanceNeighbors_gpu( molecule: Molecule,
                                  cutoff: float
                                ) -> pt.Tensor:
    """
    On-device version of `findAllDistanceNeighbors` using the native KDTree
    in `chemdm.graph.kdtree`. Same contract: shape (n_neighbors, 2), symmetric
    (both directions), excludes self-edges, and (for batched molecules)
    excludes cross-molecule edges via a fourth bias dimension.

    Stays on `molecule.x.device` for the whole call — no CPU round-trip and
    no `pt.unique` / `scatter_reduce`, so the sampler doesn't have to sync.
    """
    if isinstance( molecule, BatchedMoleculeGraph ):
        bias = molecule.molecule_id[:, None].to( dtype=molecule.x.dtype ) * (2.0 * cutoff)
        x = pt.cat( [ molecule.x, bias ], dim=1 )
    else:
        x = molecule.x

    pairs = KDTree( x ).query_pairs( r=cutoff )      # (E, 2), i<j, long
    if pairs.numel() == 0:
        return pt.empty( (0, 2), dtype=pt.long, device=x.device )

    return pt.cat( [ pairs, pairs.flip( dims=(1,) ) ], dim=0 )


@pt.no_grad()
def findAllNeighbors_gpu( molecule : Molecule,
                          d_cutoff : float
                        ) -> Tuple[pt.Tensor, pt.Tensor]:
    """
    On-device version of `findAllNeighbors`. Unique union of bond edges and
    distance-cutoff edges plus a float32 `is_bond` flag.

    Skips `pt.unique(dim=0)` and `scatter_reduce` (both CPU-syncing on MPS)
    by packing (src, dst) into a 1-D int64 key and using sorted searchsorted
    to detect which distance edges are also bonds. Assumes both edge sets are
    internally duplicate-free — true in practice (dataset bond lists and
    `KDTree.query_pairs`-derived symmetric edges are each unique).
    """
    bond_edges = molecule.edge_index.long()
    dist_edges = findAllDistanceNeighbors_gpu( molecule, d_cutoff )
    return _mergeBondAndDistanceNeighbors_gpu(
        bond_edges, dist_edges, n_atoms=int( molecule.Z.shape[0] ),
    )


def _mergeBondAndDistanceNeighbors_gpu(
    bond_edges: pt.Tensor,
    dist_edges: pt.Tensor,
    n_atoms: int,
) -> Tuple[pt.Tensor, pt.Tensor]:
    device  = bond_edges.device
    n_bonds = bond_edges.shape[0]
    n_dist  = dist_edges.shape[0]

    if n_bonds == 0:
        return dist_edges, pt.zeros( n_dist, dtype=pt.float32, device=device )
    if n_dist == 0:
        return bond_edges, pt.ones( n_bonds, dtype=pt.float32, device=device )

    # Pack (src, dst) -> src * stride + dst. stride >= n_atoms keeps the map injective.
    stride = max( n_atoms, 1 )
    bond_keys = bond_edges[:, 0] * stride + bond_edges[:, 1]
    dist_keys = dist_edges[:, 0] * stride + dist_edges[:, 1]

    bond_sorted, _ = pt.sort( bond_keys )
    idx = pt.searchsorted( bond_sorted, dist_keys ).clamp( max=bond_sorted.shape[0] - 1 )
    is_dup = bond_sorted[idx] == dist_keys

    non_dup_dist = dist_edges[ ~is_dup ]
    all_edges = pt.cat( [ bond_edges, non_dup_dist ], dim=0 )
    is_bond = pt.cat([
        pt.ones ( n_bonds,                dtype=pt.float32, device=device ),
        pt.zeros( non_dup_dist.shape[0],  dtype=pt.float32, device=device ),
    ])
    return all_edges, is_bond


@pt.no_grad()
def findAllNeighbors( molecule : Molecule,
                      d_cutoff : float
                    ) -> Tuple[pt.Tensor, pt.Tensor]:
    """
    Return all atoms that are either bonds or within a distance of each other.

    Arguments
    ---------
    molecule : Molecule
    d_cutoff : float
        The cutoff distance used for neighbor calculations.

    Returns
    -------
    all_neighbors : Tensor of shape (n_edges, 2)
        Unique directed edges.
    is_bond : Tensor of shape (n_edges,)
        1 if the edge is a bond in `molecule`, 0 otherwise.
    """

    # Merge the neighbors
    bond_neighbors = molecule.edge_index
    distance_neighbors = findAllDistanceNeighbors( molecule, d_cutoff )
    return alg.mergeBondAndDistanceNeighbors(bond_neighbors, distance_neighbors)


def _unionEdgesWithMembership(
    edge_sets: List[pt.Tensor],
    n_atoms:   int,
) -> Tuple[pt.Tensor, List[pt.Tensor]]:
    """
    Union of N internally-unique edge_index tensors. Returns the unique
    edge list and a per-input boolean membership tensor (True where the
    unique edge belongs to that input set).

    Replaces the `pt.unique(dim=0) + scatter_reduce(amax)` pattern with
    `pt.argsort` + first-occurrence selection + `pt.searchsorted` so the
    work stays on-device with no CPU fallback on MPS.
    """
    assert len( edge_sets ) > 0, "edge_sets must be non-empty"
    device = edge_sets[0].device
    stride = max( n_atoms, 1 )

    edge_sets_long = [ es.long() for es in edge_sets ]
    sorted_keys_per_set: List[pt.Tensor] = []
    for es in edge_sets_long:
        if es.numel() == 0:
            sorted_keys_per_set.append( pt.empty( 0, dtype=pt.long, device=device ) )
        else:
            keys = es[:, 0] * stride + es[:, 1]
            sorted_keys_per_set.append( pt.sort( keys ).values )

    all_edges = pt.cat( edge_sets_long, dim=0 )
    if all_edges.shape[0] == 0:
        empty_mask = pt.empty( 0, dtype=pt.bool, device=device )
        return ( pt.empty( (0, 2), dtype=pt.long, device=device ),
                 [ empty_mask for _ in edge_sets ] )

    all_keys = all_edges[:, 0] * stride + all_edges[:, 1]
    sort_idx = pt.argsort( all_keys )
    keys_s   = all_keys[sort_idx]
    edges_s  = all_edges[sort_idx]

    is_first = pt.cat([
        pt.ones( 1, dtype=pt.bool, device=device ),
        keys_s[1:] != keys_s[:-1],
    ])
    unique_edges = edges_s[is_first]
    unique_keys  = keys_s [is_first]

    memberships: List[pt.Tensor] = []
    for sk in sorted_keys_per_set:
        if sk.numel() == 0:
            memberships.append( pt.zeros( unique_edges.shape[0], dtype=pt.bool, device=device ) )
        else:
            idx = pt.searchsorted( sk, unique_keys ).clamp( max=sk.shape[0] - 1 )
            memberships.append( sk[idx] == unique_keys )

    return unique_edges, memberships


@pt.no_grad()
def findAllNeighborsReactantProduct_gpu(
    moleculeA : Molecule,
    moleculeB : Molecule,
    x         : pt.Tensor,
    d_cutoff  : float,
) -> Tuple[pt.Tensor, pt.Tensor, pt.Tensor]:
    """
    On-device version of `findAllNeighborsReactantProduct`. Same contract:
    union of `bonds_A`, `bonds_B`, and the distance-cutoff edges computed on
    the intermediate coordinates `x`, plus float32 `is_bond_A` / `is_bond_B`
    flags. Uses the on-device KDTree and avoids `pt.unique(dim=0)` /
    `scatter_reduce`.
    """
    assert pt.all( moleculeA.Z == moleculeB.Z ), \
        f"Both molecules must have the same atoms in the same ordering."
    if isinstance( moleculeA, BatchedMoleculeGraph ):
        assert isinstance( moleculeB, BatchedMoleculeGraph ), \
            f"If either molecule is batched, so must the other be."
        assert pt.all( moleculeA.molecule_id == moleculeB.molecule_id ), \
            f"Batched molecule A and B must represent the same batch."

    moleculeX = moleculeA.copyWithNewPositions( x )
    dist_edges = findAllDistanceNeighbors_gpu( moleculeX, d_cutoff )

    bonds_A = moleculeA.edge_index.long()
    bonds_B = moleculeB.edge_index.long()

    unique_edges, memberships = _unionEdgesWithMembership(
        [ bonds_A, bonds_B, dist_edges ],
        n_atoms=int( moleculeA.Z.shape[0] ),
    )
    is_bond_A = memberships[0].to( dtype=pt.float32 )
    is_bond_B = memberships[1].to( dtype=pt.float32 )
    return unique_edges, is_bond_A, is_bond_B


@pt.no_grad()
def findFixedUnionNeighbors_gpu(
    moleculeA : Molecule,
    moleculeB : Molecule,
    d_cutoff  : float,
) -> Tuple[pt.Tensor, pt.Tensor, pt.Tensor]:
    """
    On-device version of `findFixedUnionNeighbors`. Union of `bonds_A`,
    `bonds_B`, distance-cutoff edges at `xA`, and distance-cutoff edges at
    `xB`. Returns float32 `is_bond_A` / `is_bond_B` flags.
    """
    assert pt.all( moleculeA.Z == moleculeB.Z ), \
        f"Both molecules must have the same atoms in the same ordering."
    if isinstance( moleculeA, BatchedMoleculeGraph ):
        assert isinstance( moleculeB, BatchedMoleculeGraph ), \
            f"If either molecule is batched, so must the other be."
        assert pt.all( moleculeA.molecule_id == moleculeB.molecule_id ), \
            f"Batched molecule A and B must represent the same batch."

    bonds_A = moleculeA.edge_index.long()
    bonds_B = moleculeB.edge_index.long()
    dist_A  = findAllDistanceNeighbors_gpu( moleculeA, d_cutoff )
    dist_B  = findAllDistanceNeighbors_gpu( moleculeB, d_cutoff )

    unique_edges, memberships = _unionEdgesWithMembership(
        [ bonds_A, bonds_B, dist_A, dist_B ],
        n_atoms=int( moleculeA.Z.shape[0] ),
    )
    is_bond_A = memberships[0].to( dtype=pt.float32 )
    is_bond_B = memberships[1].to( dtype=pt.float32 )
    return unique_edges, is_bond_A, is_bond_B


@pt.no_grad()
def findAllNeighborsReactantProduct( moleculeA : Molecule,
                                     moleculeB : Molecule,
                                     x : pt.Tensor,
                                     d_cutoff : float
                                   ) -> Tuple[pt.Tensor, pt.Tensor, pt.Tensor]:
    """
    Return all atoms that are either bonds in molecule A, bonds in molecule B,
    or within a distance of each other.

    Arguments
    ---------
    moleculeA : Molecule
        Reactant graph.
    moleculeB : Molecule
        Product graph.
    x : Tensor of shape (N_atoms, 3)
        Intermediate coordinates used for distance-based neighbor calculations.
    d_cutoff : float
        The cutoff distance used for neighbor calculations.

    Returns
    -------
    all_neighbors : Tensor of shape (n_edges, 2)
        Unique directed edges.
    is_bond_A : Tensor of shape (n_edges,)
        1 if the edge is a bond in `moleculeA`, 0 otherwise.
    is_bond_B : Tensor of shape (n_edges,)
        1 if the edge is a bond in `moleculeB`, 0 otherwise.
    """
    assert pt.all( moleculeA.Z == moleculeB.Z ), f"Both molecules must have the same atoms in the same ordering."
    if isinstance( moleculeA, BatchedMoleculeGraph ):
        assert isinstance( moleculeB, BatchedMoleculeGraph), f"If either molecuule is a batched molecule, so must the other be"
        assert pt.all( moleculeA.molecule_id == moleculeB.molecule_id ), f"Batched molecule A and B must represent the same batch."

    device = x.device

    # Build a temperary molecule with the same strucure as A and B but with positions x
    moleculeX = moleculeA.copyWithNewPositions( x )
    distance_neighbors = findAllDistanceNeighbors( moleculeX, d_cutoff )

    # Merge the neighbors
    bond_neighbors_A = moleculeA.edge_index
    bond_neighbors_B = moleculeB.edge_index
    all_edges = pt.cat([bond_neighbors_A, bond_neighbors_B, distance_neighbors], dim=0)

    edge_type_A = pt.cat([
        pt.ones (bond_neighbors_A.shape[0], dtype=pt.float32, device=device),
        pt.zeros(bond_neighbors_B.shape[0], dtype=pt.float32, device=device),
        pt.zeros(distance_neighbors.shape[0], dtype=pt.float32, device=device),
    ])

    edge_type_B = pt.cat([
        pt.zeros(bond_neighbors_A.shape[0], dtype=pt.float32, device=device),
        pt.ones (bond_neighbors_B.shape[0], dtype=pt.float32, device=device),
        pt.zeros(distance_neighbors.shape[0], dtype=pt.float32, device=device),
    ])

    all_neighbors, inverse = pt.unique(all_edges, dim=0, return_inverse=True)

    # Flag neighbors that were bonds in A / B
    is_bond_A = pt.zeros(all_neighbors.shape[0], dtype=pt.float32, device=device)
    is_bond_A = pt.scatter_reduce(is_bond_A, 0, inverse, edge_type_A, reduce="amax", include_self=False)

    is_bond_B = pt.zeros(all_neighbors.shape[0], dtype=pt.float32, device=device)
    is_bond_B = pt.scatter_reduce(is_bond_B, 0, inverse, edge_type_B, reduce="amax", include_self=False)

    return all_neighbors, is_bond_A, is_bond_B


@pt.no_grad()
def findFixedUnionNeighbors( moleculeA : Molecule,
                             moleculeB : Molecule,
                             d_cutoff  : float
                           ) -> Tuple[pt.Tensor, pt.Tensor, pt.Tensor]:
    """
    Fixed neighborhood graph: union of

        bonds in A
      U bonds in B
      U distance neighbors of xA  (within d_cutoff)
      U distance neighbors of xB  (within d_cutoff)

    Distinguishing feature versus `findAllNeighborsReactantProduct`: the
    distance neighbors are taken at the ENDPOINTS only (xA and xB), not at
    a moving intermediate `x`. The resulting edge list + bond flags are
    therefore safe to cache and reuse across every subsequent calculation.

    Arguments
    ---------
    moleculeA : Molecule
        Reactant graph (carries xA and bonds_A).
    moleculeB : Molecule
        Product graph (carries xB and bonds_B). Must have the same atoms in
        the same ordering as moleculeA.
    d_cutoff : float
        Distance cutoff in Å.

    Returns
    -------
    all_neighbors : Tensor of shape (n_edges, 2), unique directed edges.
    is_bond_A     : Tensor of shape (n_edges,), 1 iff edge is a bond in A.
    is_bond_B     : Tensor of shape (n_edges,), 1 iff edge is a bond in B.
    """
    assert pt.all( moleculeA.Z == moleculeB.Z ), \
        f"Both molecules must have the same atoms in the same ordering."
    if isinstance( moleculeA, BatchedMoleculeGraph ):
        assert isinstance( moleculeB, BatchedMoleculeGraph ), \
            f"If either molecule is batched, so must the other be."
        assert pt.all( moleculeA.molecule_id == moleculeB.molecule_id ), \
            f"Batched molecule A and B must represent the same batch."

    device = moleculeA.x.device

    bond_neighbors_A = moleculeA.edge_index
    bond_neighbors_B = moleculeB.edge_index
    dist_neighbors_A = findAllDistanceNeighbors( moleculeA, d_cutoff )
    dist_neighbors_B = findAllDistanceNeighbors( moleculeB, d_cutoff )

    all_edges = pt.cat( [ bond_neighbors_A, bond_neighbors_B,
                          dist_neighbors_A, dist_neighbors_B ], dim=0 )

    n_bA = bond_neighbors_A.shape[0]
    n_bB = bond_neighbors_B.shape[0]
    n_dA = dist_neighbors_A.shape[0]
    n_dB = dist_neighbors_B.shape[0]

    # Edge-source tags: 1 if the edge belongs to bonds_A / bonds_B, 0 otherwise.
    edge_type_A = pt.cat([
        pt.ones (n_bA, dtype=pt.float32, device=device),
        pt.zeros(n_bB, dtype=pt.float32, device=device),
        pt.zeros(n_dA, dtype=pt.float32, device=device),
        pt.zeros(n_dB, dtype=pt.float32, device=device),
    ])
    edge_type_B = pt.cat([
        pt.zeros(n_bA, dtype=pt.float32, device=device),
        pt.ones (n_bB, dtype=pt.float32, device=device),
        pt.zeros(n_dA, dtype=pt.float32, device=device),
        pt.zeros(n_dB, dtype=pt.float32, device=device),
    ])

    all_neighbors, inverse = pt.unique( all_edges, dim=0, return_inverse=True )

    is_bond_A = pt.zeros( all_neighbors.shape[0], dtype=pt.float32, device=device )
    is_bond_A = pt.scatter_reduce( is_bond_A, 0, inverse, edge_type_A, reduce="amax", include_self=False )

    is_bond_B = pt.zeros( all_neighbors.shape[0], dtype=pt.float32, device=device )
    is_bond_B = pt.scatter_reduce( is_bond_B, 0, inverse, edge_type_B, reduce="amax", include_self=False )

    return all_neighbors, is_bond_A, is_bond_B


def recenterMolecule( molecule : Molecule ) -> Molecule:
    """
    Center the molecule to have zero unweighted center of mass. If the input 
    molecule is batched, recentering is done on the level of the constituent molecules.
    
    This function does not alter the input molecule and returns a new molecule object.
    """
    if isinstance( molecule, MoleculeGraph ):
        x_centered = molecule.x - pt.mean( molecule.x, dim=0, keepdim=True )
    elif isinstance( molecule, BatchedMoleculeGraph ):
        x = molecule.x

        # Get indices of atoms per molecule.
        unique_ids, inverse = pt.unique( molecule.molecule_id, sorted=True, return_inverse=True )
        n_molecules = unique_ids.numel()
        counts = pt.bincount(inverse, minlength=n_molecules).to(x.dtype)

        # Sum positions per molecule: (B, 3)
        sums = pt.zeros(n_molecules, 3, device=x.device, dtype=x.dtype)
        sums.index_add_(0, inverse, x)

        com = sums / counts[:, None]
        x_centered = x - com[inverse]
    else:
        raise TypeError(f"Unsupported molecule type: {type(molecule)}")

    return molecule.copyWithNewPositions( x_centered )

@pt.no_grad()
def detectRing(molecule: Molecule) -> rings.RingInfo:
    """
    Detect graph rings in a Molecule.

    This treats molecule.edge_index as a directed representation of an
    undirected molecular bond graph.

    Returns
    -------
    RingInfo
        Per-atom ring membership, ring counts, ring sizes, and ring atom sets.
    """
    return rings.detect_ring_info( molecule.Z, molecule.edge_index )


@pt.no_grad()
def detectRingBatched( molecule: Molecule ) -> rings.RingInfo:
    """
    Detect graph rings in a BatchedMoleculeGraph.

    Ring detection is run separately per molecule_id and mapped back to the
    global batch atom indices. This avoids artificial quadratic scaling in the
    number of molecules in the batch.
    """
    # Fall back in case the input molecule is not of batched type
    if not isinstance( molecule, BatchedMoleculeGraph ):
        return detectRing( molecule )
    
    device = molecule.Z.device
    molecule_id = molecule.molecule_id.long()
    n_atoms_total = int(molecule.Z.shape[0])

    local_molecules = unbatchBatchedMolecule(molecule)

    atom_in_ring = pt.zeros(n_atoms_total, dtype=pt.bool, device=device)
    atom_ring_count = pt.zeros(n_atoms_total, dtype=pt.long, device=device)
    atom_ring_sizes: list[set[int]] = [set() for _ in range(n_atoms_total)]
    global_rings: list[tuple[int, ...]] = []

    unique_molecule_ids = pt.unique(molecule_id, sorted=True)

    assert len(local_molecules) == len(unique_molecule_ids), (
        "unbatchBatchedMolecule returned a different number of molecules than "
        "there are unique molecule IDs."
    )

    for local_molecule, mol_id_tensor in zip(local_molecules, unique_molecule_ids):
        mol_id = int(mol_id_tensor.item())

        global_atoms = pt.nonzero( molecule_id == mol_id, as_tuple=False ).flatten()

        local_info = rings.detect_ring_info( local_molecule.Z, local_molecule.edge_index )

        assert local_info.atom_in_ring.shape[0] == global_atoms.numel()

        # Atom-level outputs.
        atom_in_ring[global_atoms] = local_info.atom_in_ring.to( device=device )
        atom_ring_count[global_atoms] = local_info.atom_ring_count.to( device=device )
        for local_idx, global_idx_tensor in enumerate(global_atoms):
            global_idx = int(global_idx_tensor.item())
            atom_ring_sizes[global_idx] = set(local_info.atom_ring_sizes[local_idx])

        # Ring atom indices: local -> global.
        for local_ring in local_info.rings:
            global_ring = tuple(
                sorted( int(global_atoms[local_idx].item()) for local_idx in local_ring )
            )
            global_rings.append(global_ring)

    global_rings = sorted(global_rings, key=lambda r: (len(r), r))

    return rings.RingInfo(
        atom_in_ring=atom_in_ring,
        atom_ring_count=atom_ring_count,
        atom_ring_sizes=atom_ring_sizes,
        rings=global_rings,
    )