import torch as pt
import pytest

from chemdm.MoleculeGraph import (
    MoleculeGraph,
    BatchedMoleculeGraph,
    batchMolecules,
    detectRing,
    detectRingBatched,
)


# ============================================================
# Helpers
# ============================================================


def directed_edges(undirected_edges: list[tuple[int, int]]) -> pt.Tensor:
    edges = []

    for i, j in undirected_edges:
        edges.append((i, j))
        edges.append((j, i))

    if len(edges) == 0:
        return pt.empty((0, 2), dtype=pt.long)

    return pt.tensor(edges, dtype=pt.long)


def make_molecule(
    n_atoms: int,
    undirected_edges: list[tuple[int, int]],
    *,
    dtype: pt.dtype = pt.float64,
) -> MoleculeGraph:
    return MoleculeGraph(
        Z=pt.full((n_atoms,), 6, dtype=pt.long),
        x=pt.zeros((n_atoms, 3), dtype=dtype),
        bonds=directed_edges(undirected_edges),
    )


def make_cycle(n_atoms: int) -> MoleculeGraph:
    return make_molecule(
        n_atoms=n_atoms,
        undirected_edges=[
            (i, (i + 1) % n_atoms)
            for i in range(n_atoms)
        ],
    )


def assert_same_rings(
    actual: list[tuple[int, ...]],
    expected: list[tuple[int, ...]],
) -> None:
    actual_set = {tuple(sorted(r)) for r in actual}
    expected_set = {tuple(sorted(r)) for r in expected}

    assert actual_set == expected_set, (
        f"\nActual rings:\n{sorted(actual_set, key=lambda r: (len(r), r))}"
        f"\nExpected rings:\n{sorted(expected_set, key=lambda r: (len(r), r))}"
    )


def assert_same_bool_tensor(actual: pt.Tensor, expected: list[bool]) -> None:
    actual_list = actual.detach().cpu().bool().tolist()

    assert actual_list == expected, (
        f"\nActual:\n{actual_list}"
        f"\nExpected:\n{expected}"
    )


def assert_same_long_tensor(actual: pt.Tensor, expected: list[int]) -> None:
    actual_list = actual.detach().cpu().long().tolist()

    assert actual_list == expected, (
        f"\nActual:\n{actual_list}"
        f"\nExpected:\n{expected}"
    )


def assert_same_ring_size_sets(
    actual: list[set[int]],
    expected: list[set[int]],
) -> None:
    assert actual == expected, (
        f"\nActual:\n{actual}"
        f"\nExpected:\n{expected}"
    )


def assert_ring_info(
    info,
    *,
    expected_rings: list[tuple[int, ...]],
    expected_in_ring: list[bool],
    expected_counts: list[int],
    expected_size_sets: list[set[int]],
) -> None:
    assert_same_rings(info.rings, expected_rings)
    assert_same_bool_tensor(info.atom_in_ring, expected_in_ring)
    assert_same_long_tensor(info.atom_ring_count, expected_counts)
    assert_same_ring_size_sets(info.atom_ring_sizes, expected_size_sets)


# ============================================================
# Basic no-ring tests
# ============================================================

def test_empty_molecule_has_no_rings() -> None:
    mol = MoleculeGraph(
        Z=pt.empty((0,), dtype=pt.long),
        x=pt.empty((0, 3), dtype=pt.float64),
        bonds=pt.empty((0, 2), dtype=pt.long),
    )

    info = detectRing(mol)

    assert info.rings == []
    assert info.atom_in_ring.shape == (0,)
    assert info.atom_ring_count.shape == (0,)
    assert info.atom_ring_sizes == []


def test_single_atom_has_no_rings() -> None:
    mol = make_molecule(
        n_atoms=1,
        undirected_edges=[],
    )

    info = detectRing(mol)

    assert_ring_info(
        info,
        expected_rings=[],
        expected_in_ring=[False],
        expected_counts=[0],
        expected_size_sets=[set()],
    )


def test_multi_atom_no_bonds_has_no_rings() -> None:
    mol = make_molecule(
        n_atoms=5,
        undirected_edges=[],
    )

    info = detectRing(mol)

    assert_ring_info(
        info,
        expected_rings=[],
        expected_in_ring=[False, False, False, False, False],
        expected_counts=[0, 0, 0, 0, 0],
        expected_size_sets=[set(), set(), set(), set(), set()],
    )


def test_two_atoms_single_bond_has_no_rings() -> None:
    mol = make_molecule(
        n_atoms=2,
        undirected_edges=[(0, 1)],
    )

    info = detectRing(mol)

    assert_ring_info(
        info,
        expected_rings=[],
        expected_in_ring=[False, False],
        expected_counts=[0, 0],
        expected_size_sets=[set(), set()],
    )


def test_three_atom_chain_has_no_rings() -> None:
    mol = make_molecule(
        n_atoms=3,
        undirected_edges=[
            (0, 1),
            (1, 2),
        ],
    )

    info = detectRing(mol)

    assert_ring_info(
        info,
        expected_rings=[],
        expected_in_ring=[False, False, False],
        expected_counts=[0, 0, 0],
        expected_size_sets=[set(), set(), set()],
    )


def test_disconnected_acyclic_components_have_no_rings() -> None:
    mol = make_molecule(
        n_atoms=6,
        undirected_edges=[
            (0, 1),
            (1, 2),
            (3, 4),
        ],
    )

    info = detectRing(mol)

    assert_ring_info(
        info,
        expected_rings=[],
        expected_in_ring=[False, False, False, False, False, False],
        expected_counts=[0, 0, 0, 0, 0, 0],
        expected_size_sets=[set(), set(), set(), set(), set(), set()],
    )


def test_star_graph_has_no_rings() -> None:
    mol = make_molecule(
        n_atoms=5,
        undirected_edges=[
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
        ],
    )

    info = detectRing(mol)

    assert_ring_info(
        info,
        expected_rings=[],
        expected_in_ring=[False, False, False, False, False],
        expected_counts=[0, 0, 0, 0, 0],
        expected_size_sets=[set(), set(), set(), set(), set()],
    )


# ============================================================
# Simple rings, including macrocycles
# ============================================================

def test_triangle_ring() -> None:
    mol = make_molecule(
        n_atoms=3,
        undirected_edges=[
            (0, 1),
            (1, 2),
            (2, 0),
        ],
    )

    info = detectRing(mol)

    assert_ring_info(
        info,
        expected_rings=[(0, 1, 2)],
        expected_in_ring=[True, True, True],
        expected_counts=[1, 1, 1],
        expected_size_sets=[{3}, {3}, {3}],
    )


def test_square_ring() -> None:
    mol = make_molecule(
        n_atoms=4,
        undirected_edges=[
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
        ],
    )

    info = detectRing(mol)

    assert_ring_info(
        info,
        expected_rings=[(0, 1, 2, 3)],
        expected_in_ring=[True, True, True, True],
        expected_counts=[1, 1, 1, 1],
        expected_size_sets=[{4}, {4}, {4}, {4}],
    )


def test_pentagon_ring() -> None:
    mol = make_molecule(
        n_atoms=5,
        undirected_edges=[
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 0),
        ],
    )

    info = detectRing(mol)

    assert_ring_info(
        info,
        expected_rings=[(0, 1, 2, 3, 4)],
        expected_in_ring=[True, True, True, True, True],
        expected_counts=[1, 1, 1, 1, 1],
        expected_size_sets=[{5}, {5}, {5}, {5}, {5}],
    )


def test_hexagon_ring() -> None:
    mol = make_molecule(
        n_atoms=6,
        undirected_edges=[
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 0),
        ],
    )

    info = detectRing(mol)

    assert_ring_info(
        info,
        expected_rings=[(0, 1, 2, 3, 4, 5)],
        expected_in_ring=[True, True, True, True, True, True],
        expected_counts=[1, 1, 1, 1, 1, 1],
        expected_size_sets=[{6}, {6}, {6}, {6}, {6}, {6}],
    )


def test_heptagon_ring() -> None:
    mol = make_cycle(7)

    info = detectRing(mol)

    assert_ring_info(
        info,
        expected_rings=[tuple(range(7))],
        expected_in_ring=[True] * 7,
        expected_counts=[1] * 7,
        expected_size_sets=[{7} for _ in range(7)],
    )


def test_macrocycle_12_ring() -> None:
    mol = make_cycle(12)

    info = detectRing(mol)

    assert_ring_info(
        info,
        expected_rings=[tuple(range(12))],
        expected_in_ring=[True] * 12,
        expected_counts=[1] * 12,
        expected_size_sets=[{12} for _ in range(12)],
    )


def test_macrocycle_20_ring() -> None:
    mol = make_cycle(20)

    info = detectRing(mol)

    assert_ring_info(
        info,
        expected_rings=[tuple(range(20))],
        expected_in_ring=[True] * 20,
        expected_counts=[1] * 20,
        expected_size_sets=[{20} for _ in range(20)],
    )


# ============================================================
# Rings with substituents / disconnected components
# ============================================================

def test_hexagon_with_one_substituent() -> None:
    mol = make_molecule(
        n_atoms=7,
        undirected_edges=[
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 0),
            (2, 6),
        ],
    )

    info = detectRing(mol)

    assert_ring_info(
        info,
        expected_rings=[(0, 1, 2, 3, 4, 5)],
        expected_in_ring=[True, True, True, True, True, True, False],
        expected_counts=[1, 1, 1, 1, 1, 1, 0],
        expected_size_sets=[{6}, {6}, {6}, {6}, {6}, {6}, set()],
    )


def test_hexagon_with_two_substituents() -> None:
    mol = make_molecule(
        n_atoms=8,
        undirected_edges=[
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 0),
            (2, 6),
            (4, 7),
        ],
    )

    info = detectRing(mol)

    assert_ring_info(
        info,
        expected_rings=[(0, 1, 2, 3, 4, 5)],
        expected_in_ring=[True, True, True, True, True, True, False, False],
        expected_counts=[1, 1, 1, 1, 1, 1, 0, 0],
        expected_size_sets=[{6}, {6}, {6}, {6}, {6}, {6}, set(), set()],
    )


def test_disconnected_triangle_and_chain_single_molecule_graph() -> None:
    mol = make_molecule(
        n_atoms=6,
        undirected_edges=[
            (0, 1),
            (1, 2),
            (2, 0),
            (3, 4),
            (4, 5),
        ],
    )

    info = detectRing(mol)

    assert_ring_info(
        info,
        expected_rings=[(0, 1, 2)],
        expected_in_ring=[True, True, True, False, False, False],
        expected_counts=[1, 1, 1, 0, 0, 0],
        expected_size_sets=[{3}, {3}, {3}, set(), set(), set()],
    )


def test_two_disconnected_rings_single_molecule_graph() -> None:
    mol = make_molecule(
        n_atoms=7,
        undirected_edges=[
            (0, 1),
            (1, 2),
            (2, 0),
            (3, 4),
            (4, 5),
            (5, 6),
            (6, 3),
        ],
    )

    info = detectRing(mol)

    assert_ring_info(
        info,
        expected_rings=[
            (0, 1, 2),
            (3, 4, 5, 6),
        ],
        expected_in_ring=[True, True, True, True, True, True, True],
        expected_counts=[1, 1, 1, 1, 1, 1, 1],
        expected_size_sets=[{3}, {3}, {3}, {4}, {4}, {4}, {4}],
    )


# ============================================================
# Edge weirdness: duplicates, self-edges, one-way directed bonds
# ============================================================

def test_duplicate_directed_edges_do_not_duplicate_rings() -> None:
    bonds = pt.tensor([
        [0, 1], [1, 0],
        [0, 1], [1, 0],
        [1, 2], [2, 1],
        [2, 0], [0, 2],
    ], dtype=pt.long)

    mol = MoleculeGraph(
        Z=pt.tensor([6, 6, 6]),
        x=pt.zeros((3, 3), dtype=pt.float64),
        bonds=bonds,
    )

    info = detectRing(mol)

    assert_ring_info(
        info,
        expected_rings=[(0, 1, 2)],
        expected_in_ring=[True, True, True],
        expected_counts=[1, 1, 1],
        expected_size_sets=[{3}, {3}, {3}],
    )


def test_self_edges_are_ignored() -> None:
    bonds = pt.tensor([
        [0, 0],
        [1, 1],
        [0, 1], [1, 0],
        [1, 2], [2, 1],
        [2, 0], [0, 2],
    ], dtype=pt.long)

    mol = MoleculeGraph(
        Z=pt.tensor([6, 6, 6]),
        x=pt.zeros((3, 3), dtype=pt.float64),
        bonds=bonds,
    )

    info = detectRing(mol)

    assert_ring_info(
        info,
        expected_rings=[(0, 1, 2)],
        expected_in_ring=[True, True, True],
        expected_counts=[1, 1, 1],
        expected_size_sets=[{3}, {3}, {3}],
    )


def test_one_way_directed_edges_still_treated_as_undirected() -> None:
    bonds = pt.tensor([
        [0, 1],
        [1, 2],
        [2, 0],
    ], dtype=pt.long)

    mol = MoleculeGraph(
        Z=pt.tensor([6, 6, 6]),
        x=pt.zeros((3, 3), dtype=pt.float64),
        bonds=bonds,
    )

    info = detectRing(mol)

    assert_ring_info(
        info,
        expected_rings=[(0, 1, 2)],
        expected_in_ring=[True, True, True],
        expected_counts=[1, 1, 1],
        expected_size_sets=[{3}, {3}, {3}],
    )


def test_reversed_one_way_directed_edges_still_treated_as_undirected() -> None:
    bonds = pt.tensor([
        [1, 0],
        [2, 1],
        [0, 2],
    ], dtype=pt.long)

    mol = MoleculeGraph(
        Z=pt.tensor([6, 6, 6]),
        x=pt.zeros((3, 3), dtype=pt.float64),
        bonds=bonds,
    )

    info = detectRing(mol)

    assert_ring_info(
        info,
        expected_rings=[(0, 1, 2)],
        expected_in_ring=[True, True, True],
        expected_counts=[1, 1, 1],
        expected_size_sets=[{3}, {3}, {3}],
    )


# ============================================================
# Multi-ring structures
# ============================================================

def test_spiro_two_triangles() -> None:
    # Two triangles sharing one atom:
    # 0-1-2-0 and 0-3-4-0.
    mol = make_molecule(
        n_atoms=5,
        undirected_edges=[
            (0, 1),
            (1, 2),
            (2, 0),
            (0, 3),
            (3, 4),
            (4, 0),
        ],
    )

    info = detectRing(mol)

    assert_ring_info(
        info,
        expected_rings=[
            (0, 1, 2),
            (0, 3, 4),
        ],
        expected_in_ring=[True, True, True, True, True],
        expected_counts=[2, 1, 1, 1, 1],
        expected_size_sets=[{3}, {3}, {3}, {3}, {3}],
    )


def test_spiro_triangle_and_square() -> None:
    # Triangle 0-1-2-0 and square 0-3-4-5-0 share atom 0.
    mol = make_molecule(
        n_atoms=6,
        undirected_edges=[
            (0, 1),
            (1, 2),
            (2, 0),
            (0, 3),
            (3, 4),
            (4, 5),
            (5, 0),
        ],
    )

    info = detectRing(mol)

    assert_ring_info(
        info,
        expected_rings=[
            (0, 1, 2),
            (0, 3, 4, 5),
        ],
        expected_in_ring=[True, True, True, True, True, True],
        expected_counts=[2, 1, 1, 1, 1, 1],
        expected_size_sets=[{3, 4}, {3}, {3}, {4}, {4}, {4}],
    )


def test_spiro_two_pentagons() -> None:
    # Two pentagons sharing atom 0:
    # 0-1-2-3-4-0 and 0-5-6-7-8-0.
    # Atom 0 belongs to two rings of the same size, so atom_ring_sizes[0]
    # should still be {5}, while atom_ring_count[0] should be 2.
    mol = make_molecule(
        n_atoms=9,
        undirected_edges=[
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 0),
            (0, 5),
            (5, 6),
            (6, 7),
            (7, 8),
            (8, 0),
        ],
    )

    info = detectRing(mol)

    assert_ring_info(
        info,
        expected_rings=[
            (0, 1, 2, 3, 4),
            (0, 5, 6, 7, 8),
        ],
        expected_in_ring=[True] * 9,
        expected_counts=[2, 1, 1, 1, 1, 1, 1, 1, 1],
        expected_size_sets=[{5}, {5}, {5}, {5}, {5}, {5}, {5}, {5}, {5}],
    )


def test_fused_two_triangles_share_edge() -> None:
    # Two triangles sharing edge 1-2:
    # 0-1-2-0 and 1-3-2-1.
    mol = make_molecule(
        n_atoms=4,
        undirected_edges=[
            (0, 1),
            (1, 2),
            (2, 0),
            (1, 3),
            (3, 2),
        ],
    )

    info = detectRing(mol)

    assert_ring_info(
        info,
        expected_rings=[
            (0, 1, 2),
            (1, 2, 3),
        ],
        expected_in_ring=[True, True, True, True],
        expected_counts=[1, 2, 2, 1],
        expected_size_sets=[{3}, {3}, {3}, {3}],
    )


def test_fused_two_squares_share_edge() -> None:
    # Square A: 0-1-2-3-0
    # Square B: 2-1-4-5-2
    # shared edge: 1-2.
    mol = make_molecule(
        n_atoms=6,
        undirected_edges=[
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (1, 4),
            (4, 5),
            (5, 2),
        ],
    )

    info = detectRing(mol)

    assert_ring_info(
        info,
        expected_rings=[
            (0, 1, 2, 3),
            (1, 2, 4, 5),
        ],
        expected_in_ring=[True, True, True, True, True, True],
        expected_counts=[1, 2, 2, 1, 1, 1],
        expected_size_sets=[{4}, {4}, {4}, {4}, {4}, {4}],
    )


def test_fused_two_hexagons_share_edge() -> None:
    # Ring A: 0-1-2-3-4-5-0
    # Ring B: 3-6-7-8-9-4-3
    # shared edge: 3-4.
    mol = make_molecule(
        n_atoms=10,
        undirected_edges=[
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 0),
            (3, 6),
            (6, 7),
            (7, 8),
            (8, 9),
            (9, 4),
        ],
    )

    info = detectRing(mol)

    assert_ring_info(
        info,
        expected_rings=[
            (0, 1, 2, 3, 4, 5),
            (3, 4, 6, 7, 8, 9),
        ],
        expected_in_ring=[True] * 10,
        expected_counts=[1, 1, 1, 2, 2, 1, 1, 1, 1, 1],
        expected_size_sets=[{6} for _ in range(10)],
    )


def test_bridged_bicyclic_two_five_cycles() -> None:
    # Hexagon 0-1-2-3-4-5-0 with bridge 1-6-4.
    #
    # Shortest-cycle-per-bond behavior detects two 5-cycles:
    #   0-1-6-4-5-0
    #   1-2-3-4-6-1
    mol = make_molecule(
        n_atoms=7,
        undirected_edges=[
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 0),
            (1, 6),
            (6, 4),
        ],
    )

    info = detectRing(mol)

    assert_ring_info(
        info,
        expected_rings=[
            (0, 1, 4, 5, 6),
            (1, 2, 3, 4, 6),
        ],
        expected_in_ring=[True] * 7,
        expected_counts=[1, 2, 1, 1, 2, 1, 2],
        expected_size_sets=[{5} for _ in range(7)],
    )


def test_complete_graph_k4_detects_three_triangle_cycle_basis() -> None:
    # K4 has cycle rank E - V + 1 = 6 - 4 + 1 = 3.
    # The current shortest-path-per-bond detector returns a deterministic
    # 3-cycle basis, not all four possible triangles.
    mol = make_molecule(
        n_atoms=4,
        undirected_edges=[
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 3),
            (2, 3),
        ],
    )

    info = detectRing(mol)

    assert_ring_info(
        info,
        expected_rings=[
            (0, 1, 2),
            (0, 1, 3),
            (0, 2, 3),
        ],
        expected_in_ring=[True, True, True, True],
        expected_counts=[3, 2, 2, 2],
        expected_size_sets=[{3}, {3}, {3}, {3}],
    )


def test_cubane_cube_graph_five_square_cycle_basis() -> None:
    # Cube graph:
    # bottom square: 0-1-2-3-0
    # top square:    4-5-6-7-4
    # verticals:     0-4, 1-5, 2-6, 3-7
    #
    # The cube has cycle rank E - V + 1 = 12 - 8 + 1 = 5.
    # The current detector returns a deterministic 5-cycle basis, not all
    # six square faces.
    mol = make_molecule(
        n_atoms=8,
        undirected_edges=[
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ],
    )

    info = detectRing(mol)

    assert_ring_info(
        info,
        expected_rings=[
            (0, 1, 2, 3),
            (0, 1, 4, 5),
            (0, 3, 4, 7),
            (1, 2, 5, 6),
            (2, 3, 6, 7),
        ],
        expected_in_ring=[True] * 8,
        expected_counts=[3, 3, 3, 3, 2, 2, 2, 2],
        expected_size_sets=[{4} for _ in range(8)],
    )


# ============================================================
# Chorded rings / shortest-cycle behavior
# ============================================================

def test_square_with_diagonal_detects_two_triangles() -> None:
    # Square 0-1-2-3-0 with diagonal 0-2.
    #
    # With shortest-cycle-per-bond logic, the useful smallest cycles are
    # triangles 0-1-2 and 0-2-3.
    mol = make_molecule(
        n_atoms=4,
        undirected_edges=[
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (0, 2),
        ],
    )

    info = detectRing(mol)

    assert_ring_info(
        info,
        expected_rings=[
            (0, 1, 2),
            (0, 2, 3),
        ],
        expected_in_ring=[True, True, True, True],
        expected_counts=[2, 1, 2, 1],
        expected_size_sets=[{3}, {3}, {3}, {3}],
    )


def test_hexagon_with_one_chord_detects_two_four_cycles() -> None:
    # Hexagon with chord 0-3.
    #
    # The chord splits the hexagon into two 4-cycles:
    # 0-1-2-3-0 and 0-5-4-3-0.
    mol = make_molecule(
        n_atoms=6,
        undirected_edges=[
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 0),
            (0, 3),
        ],
    )

    info = detectRing(mol)

    assert_ring_info(
        info,
        expected_rings=[
            (0, 1, 2, 3),
            (0, 3, 4, 5),
        ],
        expected_in_ring=[True, True, True, True, True, True],
        expected_counts=[2, 1, 1, 2, 1, 1],
        expected_size_sets=[{4}, {4}, {4}, {4}, {4}, {4}],
    )


# ============================================================
# Batched detection
# ============================================================

def test_detectRingBatched_size_one_batch_triangle_matches_single() -> None:
    mol = make_molecule(
        n_atoms=3,
        undirected_edges=[
            (0, 1),
            (1, 2),
            (2, 0),
        ],
    )

    info_single = detectRing(mol)

    batch = batchMolecules([mol])
    info_batched = detectRingBatched(batch)

    assert_same_rings(info_batched.rings, info_single.rings)
    assert pt.equal(info_batched.atom_in_ring.cpu(), info_single.atom_in_ring.cpu())
    assert pt.equal(info_batched.atom_ring_count.cpu(), info_single.atom_ring_count.cpu())
    assert_same_ring_size_sets(info_batched.atom_ring_sizes, info_single.atom_ring_sizes)


def test_detectRingBatched_size_one_batch_chain_matches_single() -> None:
    mol = make_molecule(
        n_atoms=4,
        undirected_edges=[
            (0, 1),
            (1, 2),
            (2, 3),
        ],
    )

    info_single = detectRing(mol)

    batch = batchMolecules([mol])
    info_batched = detectRingBatched(batch)

    assert_same_rings(info_batched.rings, info_single.rings)
    assert pt.equal(info_batched.atom_in_ring.cpu(), info_single.atom_in_ring.cpu())
    assert pt.equal(info_batched.atom_ring_count.cpu(), info_single.atom_ring_count.cpu())
    assert_same_ring_size_sets(info_batched.atom_ring_sizes, info_single.atom_ring_sizes)


def test_batched_triangle_and_square() -> None:
    mol1 = make_molecule(
        n_atoms=3,
        undirected_edges=[
            (0, 1),
            (1, 2),
            (2, 0),
        ],
    )

    mol2 = make_molecule(
        n_atoms=4,
        undirected_edges=[
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
        ],
    )

    batch = batchMolecules([mol1, mol2])
    info = detectRingBatched(batch)

    assert_ring_info(
        info,
        expected_rings=[
            (0, 1, 2),
            (3, 4, 5, 6),
        ],
        expected_in_ring=[True, True, True, True, True, True, True],
        expected_counts=[1, 1, 1, 1, 1, 1, 1],
        expected_size_sets=[{3}, {3}, {3}, {4}, {4}, {4}, {4}],
    )


def test_batched_chain_and_triangle() -> None:
    mol1 = make_molecule(
        n_atoms=3,
        undirected_edges=[
            (0, 1),
            (1, 2),
        ],
    )

    mol2 = make_molecule(
        n_atoms=3,
        undirected_edges=[
            (0, 1),
            (1, 2),
            (2, 0),
        ],
    )

    batch = batchMolecules([mol1, mol2])
    info = detectRingBatched(batch)

    assert_ring_info(
        info,
        expected_rings=[
            (3, 4, 5),
        ],
        expected_in_ring=[False, False, False, True, True, True],
        expected_counts=[0, 0, 0, 1, 1, 1],
        expected_size_sets=[set(), set(), set(), {3}, {3}, {3}],
    )


def test_batched_two_chains_no_rings() -> None:
    mol1 = make_molecule(
        n_atoms=3,
        undirected_edges=[
            (0, 1),
            (1, 2),
        ],
    )

    mol2 = make_molecule(
        n_atoms=4,
        undirected_edges=[
            (0, 1),
            (1, 2),
            (2, 3),
        ],
    )

    batch = batchMolecules([mol1, mol2])
    info = detectRingBatched(batch)

    assert_ring_info(
        info,
        expected_rings=[],
        expected_in_ring=[False] * 7,
        expected_counts=[0] * 7,
        expected_size_sets=[set() for _ in range(7)],
    )


def test_batched_three_rings_different_sizes() -> None:
    mol1 = make_cycle(3)
    mol2 = make_cycle(4)
    mol3 = make_cycle(5)

    batch = batchMolecules([mol1, mol2, mol3])
    info = detectRingBatched(batch)

    assert_ring_info(
        info,
        expected_rings=[
            (0, 1, 2),
            (3, 4, 5, 6),
            (7, 8, 9, 10, 11),
        ],
        expected_in_ring=[True] * 12,
        expected_counts=[1] * 12,
        expected_size_sets=[
            {3}, {3}, {3},
            {4}, {4}, {4}, {4},
            {5}, {5}, {5}, {5}, {5},
        ],
    )


def test_batched_no_cross_talk_same_coordinates() -> None:
    # Coordinates do not matter for graph ring detection.
    # This test still uses identical coordinates to guard against any future
    # accidental distance-based cross-talk in this path.
    mol1 = MoleculeGraph(
        Z=pt.tensor([6, 6, 6]),
        x=pt.zeros((3, 3), dtype=pt.float64),
        bonds=directed_edges([
            (0, 1),
            (1, 2),
            (2, 0),
        ]),
    )

    mol2 = MoleculeGraph(
        Z=pt.tensor([6, 6, 6]),
        x=pt.zeros((3, 3), dtype=pt.float64),
        bonds=directed_edges([
            (0, 1),
            (1, 2),
        ]),
    )

    batch = batchMolecules([mol1, mol2])
    info = detectRingBatched(batch)

    assert_ring_info(
        info,
        expected_rings=[
            (0, 1, 2),
        ],
        expected_in_ring=[True, True, True, False, False, False],
        expected_counts=[1, 1, 1, 0, 0, 0],
        expected_size_sets=[{3}, {3}, {3}, set(), set(), set()],
    )


def test_detectRingBatched_matches_individual_detectRing_with_offsets() -> None:
    mol1 = make_cycle(3)
    mol2 = make_cycle(4)
    mol3 = make_cycle(5)

    batch = batchMolecules([mol1, mol2, mol3])
    batched_info = detectRingBatched(batch)

    info1 = detectRing(mol1)
    info2 = detectRing(mol2)
    info3 = detectRing(mol3)

    expected_rings = []
    expected_rings += [tuple(sorted(i for i in ring)) for ring in info1.rings]
    expected_rings += [tuple(sorted(3 + i for i in ring)) for ring in info2.rings]
    expected_rings += [tuple(sorted(7 + i for i in ring)) for ring in info3.rings]

    expected_in_ring = (
        info1.atom_in_ring.cpu().tolist()
        + info2.atom_in_ring.cpu().tolist()
        + info3.atom_in_ring.cpu().tolist()
    )

    expected_counts = (
        info1.atom_ring_count.cpu().tolist()
        + info2.atom_ring_count.cpu().tolist()
        + info3.atom_ring_count.cpu().tolist()
    )

    expected_size_sets = (
        info1.atom_ring_sizes
        + info2.atom_ring_sizes
        + info3.atom_ring_sizes
    )

    assert_same_rings(batched_info.rings, expected_rings)
    assert_same_bool_tensor(batched_info.atom_in_ring, expected_in_ring)
    assert_same_long_tensor(batched_info.atom_ring_count, expected_counts)
    assert_same_ring_size_sets(batched_info.atom_ring_sizes, expected_size_sets)


# ============================================================
# Nested batches and raw tensor batches
# ============================================================

def test_nested_batch_triangle_chain_square() -> None:
    mol1 = make_cycle(3)

    mol2 = make_molecule(
        n_atoms=2,
        undirected_edges=[
            (0, 1),
        ],
    )

    mol3 = make_cycle(4)

    batch12 = batchMolecules([mol1, mol2])
    nested = batchMolecules([batch12, mol3])

    info = detectRingBatched(nested)

    assert_ring_info(
        info,
        expected_rings=[
            (0, 1, 2),
            (5, 6, 7, 8),
        ],
        expected_in_ring=[
            True, True, True,
            False, False,
            True, True, True, True,
        ],
        expected_counts=[
            1, 1, 1,
            0, 0,
            1, 1, 1, 1,
        ],
        expected_size_sets=[
            {3}, {3}, {3},
            set(), set(),
            {4}, {4}, {4}, {4},
        ],
    )


def test_from_raw_tensors_noncontiguous_two_triangles() -> None:
    # molecule 0 atoms: global 0, 2, 4
    # molecule 1 atoms: global 1, 3, 5
    Z = pt.full((6,), 6, dtype=pt.long)
    x = pt.zeros((6, 3), dtype=pt.float64)
    molecule_id = pt.tensor([0, 1, 0, 1, 0, 1], dtype=pt.long)

    edge_index = pt.tensor([
        [0, 2], [2, 0],
        [2, 4], [4, 2],
        [4, 0], [0, 4],
        [1, 3], [3, 1],
        [3, 5], [5, 3],
        [5, 1], [1, 5],
    ], dtype=pt.long)

    batch = BatchedMoleculeGraph.fromRawTensors(
        Z=Z,
        x=x,
        edge_index=edge_index,
        molecule_id=molecule_id,
    )

    info = detectRingBatched(batch)

    assert_ring_info(
        info,
        expected_rings=[
            (0, 2, 4),
            (1, 3, 5),
        ],
        expected_in_ring=[True, True, True, True, True, True],
        expected_counts=[1, 1, 1, 1, 1, 1],
        expected_size_sets=[{3}, {3}, {3}, {3}, {3}, {3}],
    )


def test_from_raw_tensors_noncontiguous_ring_and_chain() -> None:
    # molecule 0 atoms: global 0, 2, 4 form triangle
    # molecule 1 atoms: global 1, 3, 5 form chain
    Z = pt.full((6,), 6, dtype=pt.long)
    x = pt.zeros((6, 3), dtype=pt.float64)
    molecule_id = pt.tensor([0, 1, 0, 1, 0, 1], dtype=pt.long)

    edge_index = pt.tensor([
        [0, 2], [2, 0],
        [2, 4], [4, 2],
        [4, 0], [0, 4],
        [1, 3], [3, 1],
        [3, 5], [5, 3],
    ], dtype=pt.long)

    batch = BatchedMoleculeGraph.fromRawTensors(
        Z=Z,
        x=x,
        edge_index=edge_index,
        molecule_id=molecule_id,
    )

    info = detectRingBatched(batch)

    assert_ring_info(
        info,
        expected_rings=[
            (0, 2, 4),
        ],
        expected_in_ring=[True, False, True, False, True, False],
        expected_counts=[1, 0, 1, 0, 1, 0],
        expected_size_sets=[{3}, set(), {3}, set(), {3}, set()],
    )


def test_from_raw_tensors_nonzero_gapped_molecule_ids() -> None:
    # molecule 2 atoms: global 0, 2, 4 form triangle
    # molecule 5 atoms: global 1, 3, 5 form chain
    Z = pt.full((6,), 6, dtype=pt.long)
    x = pt.zeros((6, 3), dtype=pt.float64)
    molecule_id = pt.tensor([2, 5, 2, 5, 2, 5], dtype=pt.long)

    edge_index = pt.tensor([
        [0, 2], [2, 0],
        [2, 4], [4, 2],
        [4, 0], [0, 4],
        [1, 3], [3, 1],
        [3, 5], [5, 3],
    ], dtype=pt.long)

    batch = BatchedMoleculeGraph.fromRawTensors(
        Z=Z,
        x=x,
        edge_index=edge_index,
        molecule_id=molecule_id,
    )

    info = detectRingBatched(batch)

    assert_ring_info(
        info,
        expected_rings=[
            (0, 2, 4),
        ],
        expected_in_ring=[True, False, True, False, True, False],
        expected_counts=[1, 0, 1, 0, 1, 0],
        expected_size_sets=[{3}, set(), {3}, set(), {3}, set()],
    )


def test_from_raw_tensors_reverse_sorted_molecule_ids() -> None:
    # molecule 5 atoms: global 0, 2, 4 form triangle
    # molecule 2 atoms: global 1, 3, 5 form chain
    Z = pt.full((6,), 6, dtype=pt.long)
    x = pt.zeros((6, 3), dtype=pt.float64)
    molecule_id = pt.tensor([5, 2, 5, 2, 5, 2], dtype=pt.long)

    edge_index = pt.tensor([
        [0, 2], [2, 0],
        [2, 4], [4, 2],
        [4, 0], [0, 4],
        [1, 3], [3, 1],
        [3, 5], [5, 3],
    ], dtype=pt.long)

    batch = BatchedMoleculeGraph.fromRawTensors(
        Z=Z,
        x=x,
        edge_index=edge_index,
        molecule_id=molecule_id,
    )

    info = detectRingBatched(batch)

    assert_ring_info(
        info,
        expected_rings=[
            (0, 2, 4),
        ],
        expected_in_ring=[True, False, True, False, True, False],
        expected_counts=[1, 0, 1, 0, 1, 0],
        expected_size_sets=[{3}, set(), {3}, set(), {3}, set()],
    )


# ============================================================
# Ordering / determinism
# ============================================================

def test_detectRing_order_is_deterministic_and_sorted() -> None:
    mol = make_molecule(
        n_atoms=7,
        undirected_edges=[
            (3, 4),
            (4, 5),
            (5, 6),
            (6, 3),
            (0, 1),
            (1, 2),
            (2, 0),
        ],
    )

    info1 = detectRing(mol)
    info2 = detectRing(mol)

    assert info1.rings == info2.rings
    assert info1.rings == sorted(info1.rings, key=lambda r: (len(r), r))


def test_detectRingBatched_order_is_deterministic_and_sorted() -> None:
    mol1 = make_cycle(4)
    mol2 = make_cycle(3)

    batch = batchMolecules([mol1, mol2])

    info1 = detectRingBatched(batch)
    info2 = detectRingBatched(batch)

    assert info1.rings == info2.rings
    assert info1.rings == sorted(info1.rings, key=lambda r: (len(r), r))


# ============================================================
# CUDA device tests, skipped on CPU-only machines
# ============================================================

def test_detectRing_cuda_device_if_available() -> None:
    if not pt.cuda.is_available():
        pytest.skip("CUDA not available")
        return

    mol = make_cycle(3)
    mol = mol.to(device=pt.device("cuda"), dtype=pt.float64)

    info = detectRing(mol)

    assert info.atom_in_ring.device.type == "cuda"
    assert info.atom_ring_count.device.type == "cuda"


def test_detectRingBatched_cuda_device_if_available() -> None:
    if not pt.cuda.is_available():
        pytest.skip("CUDA not available")
        return

    mol1 = make_cycle(3)
    mol2 = make_cycle(4)

    batch = batchMolecules([mol1, mol2])
    batch = batch.to(device=pt.device("cuda"), dtype=pt.float64)

    info = detectRingBatched(batch)

    assert info.atom_in_ring.device.type == "cuda"
    assert info.atom_ring_count.device.type == "cuda"

def test_detectRing_mps_device_if_available() -> None:
    if not pt.backends.mps.is_available():
        pytest.skip("MPS not available")
        return

    mol = make_cycle(3)
    mol = mol.to(device=pt.device("mps"), dtype=pt.float32)

    info = detectRing(mol)

    assert info.atom_in_ring.device.type == "mps"
    assert info.atom_ring_count.device.type == "mps"


def test_detectRingBatched_mps_device_if_available() -> None:
    if not pt.backends.mps.is_available():
        pytest.skip("MPS not available")
        return

    mol1 = make_cycle(3)
    mol2 = make_cycle(4)

    batch = batchMolecules([mol1, mol2])
    batch = batch.to(device=pt.device("mps"), dtype=pt.float32)

    info = detectRingBatched(batch)

    assert info.atom_in_ring.device.type == "mps"
    assert info.atom_ring_count.device.type == "mps"
