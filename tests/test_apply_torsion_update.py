"""
Tests for `chemdm.geometry.apply_torsion_update`.

The function applies per-bond torsion increments Δτ to a (possibly batched)
atom point cloud, rotating the c-side of each rotatable bond about its
axis while leaving bond lengths and bond angles invariant.

The tests cover:

    1.  Identity at Δτ = 0.
    2.  Round-trip: applying +Δτ then −Δτ returns the original.
    3.  Bond-length preservation.
    4.  Bond-angle preservation.
    5.  Dihedral around (b, c) changes by exactly Δτ.
    6.  Atoms outside the c-side don't move.
    7.  Bond-order independence (Proposition 1 of the paper).
    8.  2π-periodicity in Δτ.
    9.  Sequential additivity for a single bond: Δτ_1 then Δτ_2 == Δτ_1+Δτ_2.
    10. Batched independence: updates to molecule A do not affect molecule B.

All tests use synthetic fixtures so they run without external data files.
"""

import math

import pytest
import torch as pt

from chemdm.geometry import apply_torsion_update


# ============================================================
# Helpers
# ============================================================


def linear_chain(n_atoms: int, bond_length: float = 1.5, dtype: pt.dtype = pt.float64) -> pt.Tensor:
    """A staggered planar chain along x with small zig-zag in y, no degeneracy."""
    x = pt.zeros((n_atoms, 3), dtype=dtype)
    for i in range(n_atoms):
        x[i, 0] = i * bond_length
        x[i, 1] = 0.3 if (i % 2) else -0.3   # zig-zag so bond angles are non-flat
    return x


def covalent_edges(chain: list[tuple[int, int]]) -> pt.Tensor:
    """Return (E, 2) directed edges (both ways) for a list of unordered bonds."""
    edges: list[tuple[int, int]] = []
    for u, v in chain:
        edges.append((u, v))
        edges.append((v, u))
    return pt.tensor(edges, dtype=pt.long)


def dihedral(x: pt.Tensor, a: int, b: int, c: int, d: int) -> pt.Tensor:
    """
    Signed dihedral angle of the planes a-b-c and b-c-d, in radians.
    Standard projection-onto-perpendicular-plane formulation.
    """
    b0 = x[a] - x[b]
    b1 = x[c] - x[b]
    b2 = x[d] - x[c]

    b1_hat = b1 / pt.linalg.norm(b1).clamp_min(1.0e-12)
    v = b0 - (b0 * b1_hat).sum() * b1_hat
    w = b2 - (b2 * b1_hat).sum() * b1_hat
    x_val = (v * w).sum()
    y_val = (pt.linalg.cross(b1_hat, v, dim=-1) * w).sum()
    return pt.atan2(y_val, x_val)


def edge_lengths(x: pt.Tensor, edges: pt.Tensor) -> pt.Tensor:
    """Per-edge distances ‖x[u] - x[v]‖."""
    return pt.linalg.norm(x[edges[:, 1]] - x[edges[:, 0]], dim=-1)


def all_bond_angles(x: pt.Tensor, edges_undirected: list[tuple[int, int]]) -> pt.Tensor:
    """
    For each atom j with at least two neighbors, return the angle subtended
    at j by its first two neighbors (deterministic by insertion order). Good
    enough as an invariance probe.
    """
    nbrs: dict[int, list[int]] = {}
    for u, v in edges_undirected:
        nbrs.setdefault(u, []).append(v)
        nbrs.setdefault(v, []).append(u)
    angles = []
    for j, ns in nbrs.items():
        if len(ns) < 2:
            continue
        u = x[ns[0]] - x[j]
        v = x[ns[1]] - x[j]
        c = (u * v).sum() / (pt.linalg.norm(u) * pt.linalg.norm(v)).clamp_min(1.0e-12)
        angles.append( pt.acos(c.clamp(-1.0, 1.0)) )
    return pt.stack(angles)


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def single_bond_chain():
    """
    4-atom linear chain 0-1-2-3 with one rotatable bond (1, 2).
    c-side of (1, 2) = {2, 3}.
    """
    x             = linear_chain(4)
    edges         = covalent_edges([(0, 1), (1, 2), (2, 3)])
    bonds         = pt.tensor([[1, 2]], dtype=pt.long)
    side_atom_idx = pt.tensor([2, 3], dtype=pt.long)
    side_bond_idx = pt.tensor([0, 0], dtype=pt.long)
    return dict(
        x=x, edges=edges, edges_undirected=[(0, 1), (1, 2), (2, 3)],
        bonds=bonds, side_atom_idx=side_atom_idx, side_bond_idx=side_bond_idx,
    )


@pytest.fixture
def two_bond_chain():
    """
    5-atom linear chain 0-1-2-3-4 with two chained rotatable bonds (1, 2) and (2, 3).
    c-side of (1, 2) = {2, 3, 4};  c-side of (2, 3) = {3, 4}.
    """
    x             = linear_chain(5)
    edges_undir   = [(0, 1), (1, 2), (2, 3), (3, 4)]
    edges         = covalent_edges(edges_undir)
    bonds         = pt.tensor([[1, 2], [2, 3]], dtype=pt.long)
    side_atom_idx = pt.tensor([2, 3, 4, 3, 4], dtype=pt.long)
    side_bond_idx = pt.tensor([0, 0, 0, 1, 1], dtype=pt.long)
    return dict(
        x=x, edges=edges, edges_undirected=edges_undir,
        bonds=bonds, side_atom_idx=side_atom_idx, side_bond_idx=side_bond_idx,
    )


@pytest.fixture
def batched_pair():
    """
    Two independent 4-atom chains packed into one batch with global atom and
    bond offsets — the same input layout the collator produces.

    mol A: atoms 0-3, rotatable bond (1, 2)
    mol B: atoms 4-7, rotatable bond (5, 6)   (= local (1, 2) + offset 4)
    """
    xA = linear_chain(4)
    xB = linear_chain(4) + pt.tensor([10.0, 0.0, 0.0], dtype=pt.float64)  # well-separated
    x  = pt.cat([xA, xB], dim=0)

    bonds         = pt.tensor([[1, 2], [5, 6]], dtype=pt.long)            # global
    side_atom_idx = pt.tensor([2, 3, 6, 7], dtype=pt.long)                # global
    side_bond_idx = pt.tensor([0, 0, 1, 1], dtype=pt.long)
    return dict(
        x=x, bonds=bonds, side_atom_idx=side_atom_idx, side_bond_idx=side_bond_idx,
    )


# ============================================================
# Tests
# ============================================================


def test_identity_at_zero_delta(single_bond_chain):
    """Δτ = 0 leaves the geometry unchanged exactly."""
    s = single_bond_chain
    x_new = apply_torsion_update(
        s["x"], s["bonds"], s["side_atom_idx"], s["side_bond_idx"],
        pt.zeros(1, dtype=pt.float64),
    )
    assert pt.allclose( x_new, s["x"], atol=1.0e-14 )


def test_roundtrip(single_bond_chain):
    """Applying +Δτ then −Δτ returns the original geometry at machine epsilon."""
    s = single_bond_chain
    dt = pt.tensor([0.7], dtype=pt.float64)
    x1 = apply_torsion_update( s["x"], s["bonds"], s["side_atom_idx"], s["side_bond_idx"], +dt )
    x2 = apply_torsion_update( x1,    s["bonds"], s["side_atom_idx"], s["side_bond_idx"], -dt )
    assert pt.allclose( x2, s["x"], atol=1.0e-14 )


def test_bond_lengths_preserved(two_bond_chain):
    """No covalent bond's length changes under torsion update."""
    s = two_bond_chain
    dt = pt.tensor([0.5, -1.3], dtype=pt.float64)
    x_new = apply_torsion_update(
        s["x"], s["bonds"], s["side_atom_idx"], s["side_bond_idx"], dt,
    )
    L_before = edge_lengths(s["x"],   s["edges"])
    L_after  = edge_lengths(x_new,    s["edges"])
    assert pt.allclose( L_before, L_after, atol=1.0e-13 )


def test_bond_angles_preserved(two_bond_chain):
    """No bond angle changes under torsion update."""
    s = two_bond_chain
    dt = pt.tensor([0.4, 0.9], dtype=pt.float64)
    x_new = apply_torsion_update(
        s["x"], s["bonds"], s["side_atom_idx"], s["side_bond_idx"], dt,
    )
    a_before = all_bond_angles(s["x"],   s["edges_undirected"])
    a_after  = all_bond_angles(x_new,    s["edges_undirected"])
    assert pt.allclose( a_before, a_after, atol=1.0e-13 )


@pytest.mark.parametrize("dt", [0.1, -0.7, 1.234, math.pi - 0.01])
def test_dihedral_changes_by_delta(single_bond_chain, dt):
    """
    The dihedral 0-1-2-3 must change by exactly Δτ when we apply Δτ to bond (1, 2).
    Comparison done modulo 2π since the function returns an angle in (-π, π].
    """
    s = single_bond_chain
    dt_t = pt.tensor([dt], dtype=pt.float64)

    phi0 = dihedral(s["x"], 0, 1, 2, 3)
    x1   = apply_torsion_update(
        s["x"], s["bonds"], s["side_atom_idx"], s["side_bond_idx"], dt_t,
    )
    phi1 = dihedral(x1, 0, 1, 2, 3)

    # Wrap (phi1 - phi0 - dt) into (-π, π], expect 0.
    diff = (phi1 - phi0 - dt_t).item()
    diff = ((diff + math.pi) % (2.0 * math.pi)) - math.pi
    assert abs(diff) < 1.0e-12


def test_atoms_outside_c_side_dont_move(single_bond_chain):
    """Atoms NOT in side_atom_idx (i.e. b-side atoms) keep their positions exactly."""
    s = single_bond_chain
    dt = pt.tensor([0.9], dtype=pt.float64)
    x_new = apply_torsion_update(
        s["x"], s["bonds"], s["side_atom_idx"], s["side_bond_idx"], dt,
    )
    moved = set(s["side_atom_idx"].tolist())
    for i in range(s["x"].shape[0]):
        if i in moved:
            continue
        assert pt.allclose( x_new[i], s["x"][i], atol=1.0e-14 ), f"atom {i} (b-side) moved"


def test_bond_order_independence(two_bond_chain):
    """
    Proposition 1 of Jing et al. 2022: the final geometry is independent of
    the order in which bonds are processed. We compare natural order [0, 1]
    against reversed [1, 0]; permuting `bonds` requires remapping
    `side_bond_idx` through the inverse permutation.
    """
    s = two_bond_chain
    dt = pt.tensor([0.7, -1.2], dtype=pt.float64)

    x_fwd = apply_torsion_update(
        s["x"], s["bonds"], s["side_atom_idx"], s["side_bond_idx"], dt,
    )

    perm     = pt.tensor([1, 0])                         # reverse bond order
    inv_perm = perm.argsort()                            # involution here, but in general
    x_rev = apply_torsion_update(
        s["x"], s["bonds"][perm], s["side_atom_idx"],
        inv_perm[s["side_bond_idx"]], dt[perm],
    )
    assert pt.allclose( x_fwd, x_rev, atol=1.0e-13 )


def test_2pi_periodicity(single_bond_chain):
    """Rotations are 2π-periodic: Δτ and Δτ + 2π produce the same geometry."""
    s = single_bond_chain
    dt        = pt.tensor([0.4], dtype=pt.float64)
    dt_plus_2pi = dt + 2.0 * math.pi

    x_a = apply_torsion_update( s["x"], s["bonds"], s["side_atom_idx"], s["side_bond_idx"], dt )
    x_b = apply_torsion_update( s["x"], s["bonds"], s["side_atom_idx"], s["side_bond_idx"], dt_plus_2pi )
    assert pt.allclose( x_a, x_b, atol=1.0e-12 )


def test_sequential_additivity_single_bond(single_bond_chain):
    """
    For one bond, applying Δτ_1 then Δτ_2 must equal applying (Δτ_1 + Δτ_2).
    Rotations about a common axis commute and add.
    """
    s = single_bond_chain
    dt1 = pt.tensor([0.3],  dtype=pt.float64)
    dt2 = pt.tensor([-1.1], dtype=pt.float64)

    x12 = apply_torsion_update( s["x"], s["bonds"], s["side_atom_idx"], s["side_bond_idx"], dt1 )
    x12 = apply_torsion_update( x12,    s["bonds"], s["side_atom_idx"], s["side_bond_idx"], dt2 )

    x_sum = apply_torsion_update(
        s["x"], s["bonds"], s["side_atom_idx"], s["side_bond_idx"], dt1 + dt2,
    )
    assert pt.allclose( x12, x_sum, atol=1.0e-13 )


def test_batched_independence(batched_pair):
    """
    With two molecules concatenated into one (x, bonds, side_*) layout —
    rotating only molecule A's bond must leave molecule B's atoms exactly
    where they were, and vice versa.
    """
    s = batched_pair

    # Rotate only bond 0 (mol A); leave bond 1 (mol B) alone.
    dt_only_A = pt.tensor([0.6, 0.0], dtype=pt.float64)
    x_new = apply_torsion_update(
        s["x"], s["bonds"], s["side_atom_idx"], s["side_bond_idx"], dt_only_A,
    )

    # Mol B atoms (indices 4..7) must be exactly unchanged.
    assert pt.allclose( x_new[4:], s["x"][4:], atol=1.0e-14 ), "mol B leaked through mol A update"

    # And mol A must have actually moved.
    mol_A_moved = (x_new[:4] - s["x"][:4]).abs().max().item()
    assert mol_A_moved > 1.0e-3, "mol A did not move when it should have"

    # Symmetric: rotate only mol B.
    dt_only_B = pt.tensor([0.0, -0.4], dtype=pt.float64)
    x_new = apply_torsion_update(
        s["x"], s["bonds"], s["side_atom_idx"], s["side_bond_idx"], dt_only_B,
    )
    assert pt.allclose( x_new[:4], s["x"][:4], atol=1.0e-14 ), "mol A leaked through mol B update"
    mol_B_moved = (x_new[4:] - s["x"][4:]).abs().max().item()
    assert mol_B_moved > 1.0e-3, "mol B did not move when it should have"
