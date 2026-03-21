import torch as pt
import numpy as np
from scipy.spatial import cKDTree

from abc import ABC, abstractmethod
from typing import List

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

def findAllNeighbors( molecule: Molecule,
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
    device = molecule.x.device

    # Move batch separation into an extra coordinate so different molecules
    # cannot become neighbors.
    x_cpu = molecule.x.detach().cpu()
    if isinstance( molecule, BatchedMoleculeGraph ):
        mol_id_cpu = molecule.molecule_id.detach().cpu().to(x_cpu.dtype)[:, None] 
        points = pt.cat([x_cpu, 2.0 * cutoff * mol_id_cpu], dim=1).numpy()
    else:
        points = x_cpu.numpy()

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