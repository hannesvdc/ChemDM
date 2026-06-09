"""
Tests for `chemdm.MoleculeInformation.hybridization_features`.

The function computes per-atom local-geometry descriptors from bonded
neighbors. We exercise it on synthetic ideal-geometry fixtures whose feature
values are known in closed form:

    linear      (2 neighbors at 180°)         cos = -1
    trigonal    (3 neighbors at 120°)         cos = -1/2,    coplanar
    tetrahedral (4 neighbors, methane-like)   cos = -1/3
    pyramidal   (3 neighbors, not coplanar)   pyr_vol > 0
    degree 0 / degree 1                       all-zero feature row
"""

import math
import pytest
import torch as pt

from chemdm.MoleculeInformation import (
    hybridization_features,
    HYBRIDIZATION_FEATURE_DIM,
)


# Column indices into the (N, 8) output, in the order the function emits them.
COL_MEAN, COL_MIN, COL_MAX = 0, 1, 2
COL_LINEARITY, COL_TRIGONAL, COL_TETRAHEDRAL = 3, 4, 5
COL_PLANARITY, COL_PYR_VOL = 6, 7


def _both_dirs(pairs: list[tuple[int, int]]) -> pt.Tensor:
    """Edge list (E, 2) with both directions for each unordered bond."""
    out = []
    for u, v in pairs:
        out += [[u, v], [v, u]]
    return pt.tensor(out, dtype=pt.long)


# ============================================================
# Linear: central atom 0 with two neighbors at ±x
# ============================================================

def test_linear_geometry():
    x = pt.tensor([
        [ 0.0, 0.0, 0.0],   # 0: central
        [ 1.0, 0.0, 0.0],   # 1: +x neighbor
        [-1.0, 0.0, 0.0],   # 2: -x neighbor
    ], dtype=pt.float64)
    edges = _both_dirs([(0, 1), (0, 2)])

    f = hybridization_features(x, edges)
    assert f.shape == (3, HYBRIDIZATION_FEATURE_DIM)

    # Central atom: single pair with cos = -1.
    central = f[0]
    assert pt.allclose(central[COL_MEAN], pt.tensor(-1.0, dtype=pt.float64), atol=1e-12)
    assert pt.allclose(central[COL_MIN],  pt.tensor(-1.0, dtype=pt.float64), atol=1e-12)
    assert pt.allclose(central[COL_MAX],  pt.tensor(-1.0, dtype=pt.float64), atol=1e-12)
    # Perfect linearity score = 1 (min_cos = -1 → arg of exp = 0).
    assert pt.allclose(central[COL_LINEARITY], pt.tensor(1.0, dtype=pt.float64), atol=1e-12)
    # Degree != 3 → planarity & pyramidal volume = 0.
    assert central[COL_PLANARITY] == 0.0
    assert central[COL_PYR_VOL] == 0.0

    # Terminal atoms (degree 1) → all zeros.
    for i in (1, 2):
        assert pt.allclose(f[i], pt.zeros(HYBRIDIZATION_FEATURE_DIM, dtype=pt.float64))


# ============================================================
# Trigonal planar: central atom 0 with three neighbors at 120° in xy
# ============================================================

def test_trigonal_planar_geometry():
    s32 = math.sqrt(3.0) / 2.0
    x = pt.tensor([
        [ 0.0,  0.0, 0.0],     # 0: central
        [ 1.0,  0.0, 0.0],     # 1
        [-0.5,  s32, 0.0],     # 2
        [-0.5, -s32, 0.0],     # 3
    ], dtype=pt.float64)
    edges = _both_dirs([(0, 1), (0, 2), (0, 3)])

    f = hybridization_features(x, edges)
    central = f[0]

    # All three pairwise cosines are -0.5.
    assert pt.allclose(central[COL_MEAN], pt.tensor(-0.5, dtype=pt.float64), atol=1e-12)
    assert pt.allclose(central[COL_MIN],  pt.tensor(-0.5, dtype=pt.float64), atol=1e-12)
    assert pt.allclose(central[COL_MAX],  pt.tensor(-0.5, dtype=pt.float64), atol=1e-12)
    # Trigonal score: (cos + 1/2)² = 0  ⇒  exp(0) = 1.
    assert pt.allclose(central[COL_TRIGONAL], pt.tensor(1.0, dtype=pt.float64), atol=1e-12)
    # Coplanar → pyramidal volume = 0, planarity = 1.
    assert pt.allclose(central[COL_PYR_VOL],   pt.tensor(0.0, dtype=pt.float64), atol=1e-12)
    assert pt.allclose(central[COL_PLANARITY], pt.tensor(1.0, dtype=pt.float64), atol=1e-12)


# ============================================================
# Tetrahedral: central atom 0 with four neighbors at the vertices of a tetrahedron
# ============================================================

def test_tetrahedral_geometry():
    # Standard tetrahedral arrangement: (±1, ±1, ±1) with an even number of minuses.
    x = pt.tensor([
        [ 0.0,  0.0,  0.0],
        [ 1.0,  1.0,  1.0],
        [ 1.0, -1.0, -1.0],
        [-1.0,  1.0, -1.0],
        [-1.0, -1.0,  1.0],
    ], dtype=pt.float64)
    edges = _both_dirs([(0, 1), (0, 2), (0, 3), (0, 4)])

    f = hybridization_features(x, edges)
    central = f[0]

    # All six pairwise cosines are -1/3.
    assert pt.allclose(central[COL_MEAN], pt.tensor(-1.0/3.0, dtype=pt.float64), atol=1e-12)
    assert pt.allclose(central[COL_MIN],  pt.tensor(-1.0/3.0, dtype=pt.float64), atol=1e-12)
    assert pt.allclose(central[COL_MAX],  pt.tensor(-1.0/3.0, dtype=pt.float64), atol=1e-12)
    # Tetrahedral score: (cos + 1/3)² = 0  ⇒  exp(0) = 1.
    assert pt.allclose(central[COL_TETRAHEDRAL], pt.tensor(1.0, dtype=pt.float64), atol=1e-12)
    # Degree != 3 → planarity & pyramidal volume = 0.
    assert central[COL_PLANARITY] == 0.0
    assert central[COL_PYR_VOL] == 0.0


# ============================================================
# Pyramidal degree-3 (NH3-like): three neighbors below the central atom
# ============================================================

def test_pyramidal_degree3_has_nonzero_volume():
    # All three neighbors share a z < 0 component → non-coplanar with the
    # central atom at origin → triple product of unit vectors is nonzero.
    s32 = math.sqrt(3.0) / 2.0
    x = pt.tensor([
        [ 0.0,  0.0,  0.0],     # 0: central
        [ 1.0,  0.0, -0.5],
        [-0.5,  s32, -0.5],
        [-0.5, -s32, -0.5],
    ], dtype=pt.float64)
    edges = _both_dirs([(0, 1), (0, 2), (0, 3)])

    f = hybridization_features(x, edges)
    central = f[0]

    assert central[COL_PYR_VOL] > 0.0, "Pyramidal degree-3 geometry should have nonzero volume"
    # Planarity score should be strictly less than 1 (pyr_vol > 0 → exp(-positive) < 1).
    assert central[COL_PLANARITY] < 1.0
    assert central[COL_PLANARITY] > 0.0


# ============================================================
# Edge cases: degree 0 and degree 1 → all-zero feature row
# ============================================================

def test_isolated_atom_returns_zeros():
    x = pt.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=pt.float64)
    edges = pt.empty((0, 2), dtype=pt.long)

    f = hybridization_features(x, edges)
    assert f.shape == (2, HYBRIDIZATION_FEATURE_DIM)
    assert pt.allclose(f, pt.zeros_like(f))


def test_degree_one_atom_returns_zeros():
    # 2-atom molecule: each atom has degree 1.
    x = pt.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=pt.float64)
    edges = _both_dirs([(0, 1)])

    f = hybridization_features(x, edges)
    assert pt.allclose(f, pt.zeros_like(f))


# ============================================================
# Soft-score ordering sanity: each ideal geometry should score highest
# on its own target.
# ============================================================

@pytest.mark.parametrize("geom_name,expected_best_col", [
    ("linear",     COL_LINEARITY),
    ("trigonal",   COL_TRIGONAL),
    ("tetrahedral", COL_TETRAHEDRAL),
])
def test_soft_score_ordering(geom_name, expected_best_col):
    s32 = math.sqrt(3.0) / 2.0
    if geom_name == "linear":
        x = pt.tensor([[0,0,0],[1,0,0],[-1,0,0]], dtype=pt.float64)
        edges = _both_dirs([(0, 1), (0, 2)])
    elif geom_name == "trigonal":
        x = pt.tensor([[0,0,0],[1,0,0],[-0.5,s32,0],[-0.5,-s32,0]], dtype=pt.float64)
        edges = _both_dirs([(0, 1), (0, 2), (0, 3)])
    elif geom_name == "tetrahedral":
        x = pt.tensor([[0,0,0],[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]], dtype=pt.float64)
        edges = _both_dirs([(0, 1), (0, 2), (0, 3), (0, 4)])

    f = hybridization_features(x, edges)
    soft_scores = f[0, COL_LINEARITY:COL_TETRAHEDRAL + 1]
    best = int(soft_scores.argmax().item()) + COL_LINEARITY
    assert best == expected_best_col, (
        f"{geom_name} geometry should score highest on column {expected_best_col} "
        f"but argmax is {best} (scores = {soft_scores.tolist()})"
    )
