import torch as pt

import chemdm.graph.algorithms as alg
import chemdm.graph.rings as rings

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