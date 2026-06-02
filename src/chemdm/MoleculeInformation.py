from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch as pt

from chemdm.MoleculeGraph import Molecule, BatchedMoleculeGraph, detectRing, detectRingBatched, RingInformation


# ============================================================
# Constants / feature definitions
# ============================================================
DEFAULT_ATOMIC_NUMBERS = [
    1,   # H
    6,   # C
    7,   # N
    8,   # O
    9,   # F
    15,  # P
    16,  # S
    17,  # Cl
    35,  # Br
    53,  # I
]

DEFAULT_RING_SIZES = [
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    12,
]

MAX_ATOMIC_NUMBER = 53

_ATOMIC_MASS_TABLE = pt.zeros(MAX_ATOMIC_NUMBER + 1, dtype=pt.float32)
_ATOMIC_MASS_TABLE[1] = 1.00784        # H
_ATOMIC_MASS_TABLE[2] = 4.002602       # He
_ATOMIC_MASS_TABLE[3] = 6.94           # Li
_ATOMIC_MASS_TABLE[4] = 9.0121831      # Be
_ATOMIC_MASS_TABLE[5] = 10.81          # B
_ATOMIC_MASS_TABLE[6] = 12.011         # C
_ATOMIC_MASS_TABLE[7] = 14.007         # N
_ATOMIC_MASS_TABLE[8] = 15.999         # O
_ATOMIC_MASS_TABLE[9] = 18.998403163   # F
_ATOMIC_MASS_TABLE[10] = 20.1797       # Ne
_ATOMIC_MASS_TABLE[11] = 22.98976928   # Na
_ATOMIC_MASS_TABLE[12] = 24.305        # Mg
_ATOMIC_MASS_TABLE[13] = 26.9815385    # Al
_ATOMIC_MASS_TABLE[14] = 28.085        # Si
_ATOMIC_MASS_TABLE[15] = 30.973761998  # P
_ATOMIC_MASS_TABLE[16] = 32.06         # S
_ATOMIC_MASS_TABLE[17] = 35.45         # Cl
_ATOMIC_MASS_TABLE[18] = 39.948        # Ar
_ATOMIC_MASS_TABLE[19] = 39.0983       # K
_ATOMIC_MASS_TABLE[20] = 40.078        # Ca
_ATOMIC_MASS_TABLE[21] = 44.955908     # Sc
_ATOMIC_MASS_TABLE[22] = 47.867        # Ti
_ATOMIC_MASS_TABLE[23] = 50.9415       # V
_ATOMIC_MASS_TABLE[24] = 51.9961       # Cr
_ATOMIC_MASS_TABLE[25] = 54.938044     # Mn
_ATOMIC_MASS_TABLE[26] = 55.845        # Fe
_ATOMIC_MASS_TABLE[27] = 58.933194     # Co
_ATOMIC_MASS_TABLE[28] = 58.6934       # Ni
_ATOMIC_MASS_TABLE[29] = 63.546        # Cu
_ATOMIC_MASS_TABLE[30] = 65.38         # Zn
_ATOMIC_MASS_TABLE[31] = 69.723        # Ga
_ATOMIC_MASS_TABLE[32] = 72.630        # Ge
_ATOMIC_MASS_TABLE[33] = 74.921595     # As
_ATOMIC_MASS_TABLE[34] = 78.971        # Se
_ATOMIC_MASS_TABLE[35] = 79.904        # Br
_ATOMIC_MASS_TABLE[36] = 83.798        # Kr
_ATOMIC_MASS_TABLE[37] = 85.4678       # Rb
_ATOMIC_MASS_TABLE[38] = 87.62         # Sr
_ATOMIC_MASS_TABLE[39] = 88.90584      # Y
_ATOMIC_MASS_TABLE[40] = 91.224        # Zr
_ATOMIC_MASS_TABLE[41] = 92.90637      # Nb
_ATOMIC_MASS_TABLE[42] = 95.95         # Mo
_ATOMIC_MASS_TABLE[43] = 98.0          # Tc
_ATOMIC_MASS_TABLE[44] = 101.07        # Ru
_ATOMIC_MASS_TABLE[45] = 102.90550     # Rh
_ATOMIC_MASS_TABLE[46] = 106.42        # Pd
_ATOMIC_MASS_TABLE[47] = 107.8682      # Ag
_ATOMIC_MASS_TABLE[48] = 112.414       # Cd
_ATOMIC_MASS_TABLE[49] = 114.818       # In
_ATOMIC_MASS_TABLE[50] = 118.710       # Sn
_ATOMIC_MASS_TABLE[51] = 121.760       # Sb
_ATOMIC_MASS_TABLE[52] = 127.60        # Te
_ATOMIC_MASS_TABLE[53] = 126.90447     # I


# ============================================================
# Data containers
# ============================================================
@dataclass(frozen=True)
class AtomInformation:
    """
    Model-ready atom-level information.

    All tensor fields have first dimension n_atoms.
    """

    Z: pt.Tensor
    atomic_mass: pt.Tensor
    atomic_mass_scaled: pt.Tensor
    degree: pt.Tensor

    atom_type_one_hot: pt.Tensor
    atom_in_ring: pt.Tensor
    atom_ring_count: pt.Tensor
    atom_ring_size_flags: pt.Tensor

@dataclass(frozen=True)
class EdgeInformation:
    edge_index: pt.Tensor

    src: pt.Tensor
    dst: pt.Tensor

    dx: pt.Tensor
    distance: pt.Tensor
    unit_dx: pt.Tensor

    edge_in_ring: pt.Tensor
    edge_ring_count: pt.Tensor
    edge_ring_size_flags: pt.Tensor

    same_molecule: Optional[pt.Tensor]


@dataclass(frozen=True)
class MoleculeInformation:
    """
    Combined atom and edge information for a Molecule or BatchedMoleculeGraph.
    """

    atoms: AtomInformation
    edges: EdgeInformation

    n_atoms: int
    n_edges: int
    
    # Optional fields
    ring_info: Optional[RingInformation]

    molecule_id: Optional[pt.Tensor]


# ============================================================
# Small helpers
# ============================================================

@pt.no_grad()
def one_hot_atomic_numbers( Z: pt.Tensor, allowed_atomic_numbers: list[int], *, dtype : pt.dtype = pt.float32 ) -> pt.Tensor:
    """
    One-hot encode atomic numbers in a vectorized way.

    Unknown atomic numbers are encoded as all zeros. 
    """
    Z_long = Z.long().flatten()

    # Vectorized one-hot encoding
    allowed_atoms = pt.tensor( allowed_atomic_numbers, device=Z.device, dtype=pt.long )
    out = ( Z_long[:,None] == allowed_atoms[None,:] )

    return out.to( dtype=dtype )


@pt.no_grad()
def compute_degree( n_atoms: int, edge_index: pt.Tensor, *, dtype: pt.dtype = pt.float32) -> pt.Tensor:
    """
    Compute directed graph degree from edge_index.

    Since MoleculeGraph usually stores both directions for each bond,
    this gives the usual undirected degree when edges are bidirectional.
    """
    device = edge_index.device
    degree = pt.zeros( (n_atoms,), dtype=dtype, device=device )

    if edge_index.numel() == 0:
        return degree

    src = edge_index[:, 0].long()
    degree.scatter_add_( dim=0, index=src, src=pt.ones_like(src, dtype=dtype, device=device) )

    return degree


@pt.no_grad()
def ring_size_flags( atom_ring_sizes: list[set[int]], allowed_ring_sizes: list[int], *, device: pt.device, dtype : pt.dtype ) -> pt.Tensor:
    """
    Convert per-atom ring-size sets into multi-hot flags.

    Example:
        atom_ring_sizes[i] = {5, 6}
        ring_sizes = [3, 4, 5, 6, 7]

        output[i] = [0, 0, 1, 1, 0]
    """
    n_atoms = len( atom_ring_sizes )

    allowed_ring_sizes_tensor = pt.tensor( allowed_ring_sizes, dtype=pt.long, device=device )
    flags = pt.zeros( (n_atoms, len(allowed_ring_sizes)), dtype=pt.long, device=device )
    for atom_idx, sizes in enumerate(atom_ring_sizes):
        current_ring_sizes = pt.tensor( sorted(sizes), dtype=pt.long, device=device ) # (1, n_rings)
        if len(sizes) == 0:
            continue

        # multi-hot encoding
        flags[atom_idx, :] = pt.isin( allowed_ring_sizes_tensor, current_ring_sizes )

    return flags.to( dtype=dtype )


def safe_normalize( x: pt.Tensor, *, eps: float = 1.0e-12 ) -> pt.Tensor:
    norm = pt.linalg.norm( x, dim=-1, keepdim=True )
    return x / pt.clamp(norm, min=eps)


# ============================================================
# Atom information
# ============================================================

@pt.no_grad()
def computeAtomInformation( molecule: Molecule,
                            *,
                            ring_info : Optional[RingInformation] = None,
                            allowed_atomic_numbers: Optional[list[int]] = None,
                            allowed_ring_sizes: Optional[list[int]] = None,
    ) -> AtomInformation:
    """
    Compute atom-level model features.

    This function performs feature engineering only. It does not mutate the
    molecule and does not belong to graph topology code.
    """
    if allowed_atomic_numbers is None:
        allowed_atomic_numbers = DEFAULT_ATOMIC_NUMBERS

    if allowed_ring_sizes is None:
        allowed_ring_sizes = DEFAULT_RING_SIZES

    Z = molecule.Z.long()
    device = Z.device
    n_atoms = int(Z.shape[0])

    # Use x dtype for floating model features.
    dtype = molecule.x.dtype

    degree = compute_degree( n_atoms=n_atoms, edge_index=molecule.edge_index, dtype=dtype )
    atom_type_one_hot = one_hot_atomic_numbers( Z, allowed_atomic_numbers=allowed_atomic_numbers, dtype=dtype )
    atom_mass = _ATOMIC_MASS_TABLE[Z].to( device=device, dtype=dtype )
    atom_mass_scaled = atom_mass / 100.0 # magic constant, I know...

    if ring_info is not None:
        atom_in_ring = ring_info.atom_in_ring.to(device=device)
        atom_ring_count = ring_info.atom_ring_count.to(device=device)
        atom_ring_size_flags = ring_size_flags( ring_info.atom_ring_sizes, allowed_ring_sizes=allowed_ring_sizes, device=device, dtype=dtype )
    else:
        atom_in_ring = pt.zeros((n_atoms,), dtype=pt.bool, device=device)
        atom_ring_count = pt.zeros((n_atoms,), dtype=pt.long, device=device)
        atom_ring_size_flags = pt.zeros( (n_atoms, len(allowed_ring_sizes)), dtype=dtype, device=device )

    # scalar_features = pt.cat(
    #     [
    #         atom_type_one_hot,
    #         atom_mass_scaled[:,None],
    #         degree[:, None],
    #         atom_in_ring.to(dtype=dtype)[:, None],
    #         atom_ring_count.to(dtype=dtype)[:, None],
    #         atom_ring_size_flags,
    #     ],
    #     dim=-1,
    # )

    return AtomInformation(
        Z=Z,
        atomic_mass=atom_mass,
        atomic_mass_scaled=atom_mass_scaled,
        degree=degree,
        atom_type_one_hot=atom_type_one_hot,
        atom_in_ring=atom_in_ring,
        atom_ring_count=atom_ring_count,
        atom_ring_size_flags=atom_ring_size_flags,
    )


# ============================================================
# Edge information
# ============================================================

@pt.no_grad()
def edge_ring_features( edge_index: pt.Tensor,
                        rings: list[tuple[int, ...]],
                        allowed_ring_sizes: list[int],
    ) -> tuple[pt.Tensor, pt.Tensor, pt.Tensor]:
    """
    Compute edge-level ring features.

    Returns
    -------
    edge_in_ring:
        Bool tensor of shape (n_edges,).

    edge_ring_count:
        Long tensor of shape (n_edges,).

    edge_ring_size_flags:
        Float tensor of shape (n_edges, len(ring_sizes)).
    """
    device = edge_index.device
    edge_index = edge_index.long()
    n_edges = int(edge_index.shape[0])

    edge_in_ring = pt.zeros( (n_edges,), dtype=pt.bool, device=device )
    edge_ring_count = pt.zeros( (n_edges,), dtype=pt.long, device=device )
    edge_ring_size_flags = pt.zeros( (n_edges, len(allowed_ring_sizes)), dtype=pt.long, device=device )

    if n_edges == 0 or len(rings) == 0:
        return edge_in_ring, edge_ring_count, edge_ring_size_flags

    src = edge_index[:, 0]
    dst = edge_index[:, 1]

    for ring in rings:
        ring_atoms = pt.tensor( ring, dtype=pt.long, device=device )
        ring_size = len(ring)

        src_in_ring = pt.isin( src, ring_atoms )
        dst_in_ring = pt.isin( dst, ring_atoms )
        edge_mask = src_in_ring & dst_in_ring

        if not pt.any(edge_mask):
            continue

        edge_in_ring = edge_in_ring | edge_mask
        edge_ring_count[edge_mask] += 1

        if ring_size in allowed_ring_sizes:
            k = allowed_ring_sizes.index(ring_size)
            edge_ring_size_flags[edge_mask, k] = 1.0

    return edge_in_ring, edge_ring_count, edge_ring_size_flags

def computeEdgeInformation( molecule: Molecule,
                            *,
                            ring_info : Optional[RingInformation] = None,
                            allowed_ring_sizes: Optional[list[int]] = None,
                            eps: float = 1.0e-12,
    ) -> EdgeInformation:
    if allowed_ring_sizes is None:
        allowed_ring_sizes = DEFAULT_RING_SIZES

    edge_index = molecule.edge_index.long()
    x = molecule.x
    device = x.device
    dtype = x.dtype

    if edge_index.numel() == 0:
        empty_long = pt.empty((0,), dtype=pt.long, device=device)
        empty_bool = pt.empty((0,), dtype=pt.bool, device=device)
        empty_vec = pt.empty((0, 3), dtype=dtype, device=device)

        return EdgeInformation(
            edge_index=edge_index.to(device=device),
            src=empty_long,
            dst=empty_long,
            dx=empty_vec,
            distance=pt.empty((0, 1), dtype=dtype, device=device),
            unit_dx=empty_vec,
            edge_in_ring=empty_bool,
            edge_ring_count=empty_long,
            edge_ring_size_flags=pt.empty((0, len(allowed_ring_sizes)), dtype=dtype, device=device),
            same_molecule=None,
        )

    edge_index = edge_index.to(device=device)
    src = edge_index[:, 0].long()
    dst = edge_index[:, 1].long()

    dx = x[dst] - x[src]
    distance = pt.linalg.norm(dx, dim=-1, keepdim=True)
    unit_dx = dx / pt.clamp(distance, min=eps)

    if isinstance( molecule, BatchedMoleculeGraph ):
        molecule_id = molecule.molecule_id.to(device=device).long()
        same_molecule = molecule_id[src] == molecule_id[dst]
    else:
        same_molecule = None

    if ring_info is not None:
        edge_in_ring, edge_ring_count, edge_ring_size_flags = edge_ring_features(
            edge_index=edge_index,
            rings=ring_info.rings,
            allowed_ring_sizes=allowed_ring_sizes,
        )
    else:
        edge_in_ring = pt.zeros((edge_index.shape[0],), dtype=pt.bool, device=device)
        edge_ring_count = pt.zeros((edge_index.shape[0],), dtype=pt.long, device=device)
        edge_ring_size_flags = pt.zeros(
            (edge_index.shape[0], len(allowed_ring_sizes)),
            dtype=dtype,
            device=device,
        )

    return EdgeInformation(
        edge_index=edge_index,
        src=src,
        dst=dst,
        dx=dx,
        distance=distance,
        unit_dx=unit_dx,
        edge_in_ring=edge_in_ring,
        edge_ring_count=edge_ring_count,
        edge_ring_size_flags=edge_ring_size_flags,
        same_molecule=same_molecule,
    )


# ============================================================
# Combined molecule information
# ============================================================

def computeMoleculeInformation( molecule: Molecule,
                                *,
                                allowed_atomic_numbers: Optional[list[int]] = None,
                                allowed_ring_sizes: Optional[list[int]] = None,
                                include_ring_information: bool = True,
    ) -> MoleculeInformation:
    if include_ring_information:
        if isinstance(molecule, BatchedMoleculeGraph):
            ring_info = detectRingBatched( molecule )
        else:
            ring_info = detectRing( molecule )
    else:
        ring_info = None

    atoms = computeAtomInformation(
        molecule,
        ring_info=ring_info,
        allowed_atomic_numbers=allowed_atomic_numbers,
        allowed_ring_sizes=allowed_ring_sizes,
    )

    edges = computeEdgeInformation( molecule, ring_info=ring_info )

    if isinstance(molecule, BatchedMoleculeGraph):
        molecule_id = molecule.molecule_id.long()
    else:
        molecule_id = None

    return MoleculeInformation(
        atoms=atoms,
        edges=edges,
        ring_info=ring_info,
        n_atoms=int(molecule.Z.shape[0]),
        n_edges=int(molecule.edge_index.shape[0]),
        molecule_id=molecule_id,
    )