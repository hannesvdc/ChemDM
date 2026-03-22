import torch as pt
import numpy as np
import copy

import chemdm.graph.algorithms as alg

from abc import ABC, abstractmethod
from typing import List, Tuple, Self

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

    def copyWithNewPositions(self, x: pt.Tensor) -> Self:
        return type(self)( self._Z, x, self._edge_index )

class BatchedMoleculeGraph( Molecule ):

    def __init__(self, molecules: List[MoleculeGraph]) -> None:
        # Concatenate atomic state and positions.
        self._Z = pt.cat([mol.Z for mol in molecules], dim=0)
        self._x = pt.cat([mol.x for mol in molecules], dim=0)

        # Calculate the total offset per molecule.
        n_atoms = pt.tensor([mol.Z.shape[0] for mol in molecules], dtype=pt.long)
        self._molecule_id = pt.repeat_interleave( pt.arange(len(molecules), dtype=pt.long), n_atoms )
        offsets = pt.cumsum( pt.cat([pt.tensor([0], dtype=pt.long), n_atoms[:-1]]), dim=0 )

        # Merge the edge indices.
        edge_list = [ mol.edge_index + offset for mol, offset in zip(molecules, offsets) ]
        self._edge_index = pt.cat(edge_list, dim=0)   # shape: (total_edges, 2)

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
def batchMolecules(molecules: List[MoleculeGraph]) -> BatchedMoleculeGraph:
    """
    Create a batched molecule from a list of single-molecule graphs. 
    
    TODO: Extend later to take a list of Molecules - some already batched - 
    and deal with molecule indexing.

    Arguments
    ---------
    molecules : List[MoleculeGraph]
        List of single-molecule graphs to merge.

    Returns
    -------
    batch_molecule : BatchedMoleculeGraph
        The large merged molecule.
    """
    return BatchedMoleculeGraph(molecules)

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
    distance_neighbors = findAllDistanceNeighbors( moleculeX, d_cutoff) # type: ignore

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