import torch as pt

from chemdm.MoleculeGraph import MoleculeGraph, batchMolecules, findAllNeighbors

def assert_same_edges(actual: pt.Tensor, expected: pt.Tensor) -> None:
    actual_set = set(map(tuple, actual.tolist()))
    expected_set = set(map(tuple, expected.tolist()))
    assert actual_set == expected_set, f"\nActual:\n{actual}\nExpected:\n{expected}"

def assert_edge_flags(
    edges: pt.Tensor,
    is_bond: pt.Tensor,
    expected: dict[tuple[int, int], float]
) -> None:
    actual = {tuple(e): float(f) for e, f in zip(edges.tolist(), is_bond.tolist())}
    assert actual == expected, f"\nActual:\n{actual}\nExpected:\n{expected}"


def run_test(name: str, test_fn) -> bool:
    try:
        test_fn()
        print(f"[PASS] {name}")
        return True
    except AssertionError as e:
        print(f"[FAIL] {name}")
        print(e)
        return False
    except Exception as e:
        print(f"[ERROR] {name}")
        print(e)
        return False


def test_findAllNeighbors_single_bond_and_distance() -> None:
    # 0--1 is a bond, 1--2 is only a distance neighbor
    mol = MoleculeGraph(
        Z=pt.tensor([6, 1, 8]),
        x=pt.tensor([
            [0.0, 0.0, 0.0],   # atom 0
            [1.0, 0.0, 0.0],   # atom 1
            [2.2, 0.0, 0.0],   # atom 2
        ]),
        bonds=pt.tensor([
            [0, 1],
            [1, 0],
        ], dtype=pt.long),
    )

    all_neighbors, is_bond = findAllNeighbors(mol, d_cutoff=1.5)

    expected_edges = pt.tensor([
        [0, 1], [1, 0],   # bond
        [1, 2], [2, 1],   # geometric only
    ], dtype=pt.long)

    expected_flags = {
        (0, 1): 1.0,
        (1, 0): 1.0,
        (1, 2): 0.0,
        (2, 1): 0.0,
    }

    assert_same_edges(all_neighbors, expected_edges)
    assert_edge_flags(all_neighbors, is_bond, expected_flags)


def test_findAllNeighbors_overlap_marks_bond() -> None:
    # Bonded atoms are also within cutoff: should still be marked as bond
    mol = MoleculeGraph(
        Z=pt.tensor([6, 1]),
        x=pt.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]),
        bonds=pt.tensor([
            [0, 1],
            [1, 0],
        ], dtype=pt.long),
    )

    all_neighbors, is_bond = findAllNeighbors(mol, d_cutoff=1.5)

    expected_edges = pt.tensor([
        [0, 1],
        [1, 0],
    ], dtype=pt.long)

    expected_flags = {
        (0, 1): 1.0,
        (1, 0): 1.0,
    }

    assert_same_edges(all_neighbors, expected_edges)
    assert_edge_flags(all_neighbors, is_bond, expected_flags)


def test_findAllNeighbors_single_no_neighbors() -> None:
    mol = MoleculeGraph(
        Z=pt.tensor([6, 8]),
        x=pt.tensor([
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ]),
        bonds=pt.empty((0, 2), dtype=pt.long),
    )

    all_neighbors, is_bond = findAllNeighbors(mol, d_cutoff=1.0)

    assert_same_edges(all_neighbors, pt.empty((0, 2), dtype=pt.long))
    assert is_bond.numel() == 0


def test_findAllNeighbors_batched() -> None:
    mol1 = MoleculeGraph(
        Z=pt.tensor([6, 1]),
        x=pt.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]),
        bonds=pt.tensor([
            [0, 1],
            [1, 0],
        ], dtype=pt.long),
    )

    mol2 = MoleculeGraph(
        Z=pt.tensor([8, 1, 1]),
        x=pt.tensor([
            [10.0, 0.0, 0.0],
            [11.0, 0.0, 0.0],
            [12.2, 0.0, 0.0],
        ]),
        bonds=pt.empty((0, 2), dtype=pt.long),
    )

    batch = batchMolecules([mol1, mol2])
    all_neighbors, is_bond = findAllNeighbors(batch, d_cutoff=1.5)

    expected_edges = pt.tensor([
        [0, 1], [1, 0],   # mol1 bond
        [2, 3], [3, 2],   # mol2 geometric
        [3, 4], [4, 3],   # mol2 geometric
    ], dtype=pt.long)

    expected_flags = {
        (0, 1): 1.0,
        (1, 0): 1.0,
        (2, 3): 0.0,
        (3, 2): 0.0,
        (3, 4): 0.0,
        (4, 3): 0.0,
    }

    assert_same_edges(all_neighbors, expected_edges)
    assert_edge_flags(all_neighbors, is_bond, expected_flags)


def test_findAllNeighbors_batched_no_cross_talk() -> None:
    # Same coordinates in different molecules should NOT interact
    mol1 = MoleculeGraph(
        Z=pt.tensor([6, 1]),
        x=pt.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]),
        bonds=pt.empty((0, 2), dtype=pt.long),
    )

    mol2 = MoleculeGraph(
        Z=pt.tensor([8, 1]),
        x=pt.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]),
        bonds=pt.empty((0, 2), dtype=pt.long),
    )

    batch = batchMolecules([mol1, mol2])
    all_neighbors, is_bond = findAllNeighbors(batch, d_cutoff=1.5)

    expected_edges = pt.tensor([
        [0, 1], [1, 0],
        [2, 3], [3, 2],
    ], dtype=pt.long)

    expected_flags = {
        (0, 1): 0.0,
        (1, 0): 0.0,
        (2, 3): 0.0,
        (3, 2): 0.0,
    }

    assert_same_edges(all_neighbors, expected_edges)
    assert_edge_flags(all_neighbors, is_bond, expected_flags)


def test_findAllNeighbors_properties() -> None:
    mol = MoleculeGraph(
        Z=pt.tensor([6, 1, 1]),
        x=pt.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.2, 0.0, 0.0],
        ]),
        bonds=pt.empty((0, 2), dtype=pt.long),
    )

    all_neighbors, is_bond = findAllNeighbors(mol, d_cutoff=1.5)

    # No self-edges
    assert not pt.any(all_neighbors[:, 0] == all_neighbors[:, 1]), \
        f"Found self-edges:\n{all_neighbors}"

    # Symmetry
    edge_set = set(map(tuple, all_neighbors.tolist()))
    for i, j in edge_set:
        assert (j, i) in edge_set, f"Missing reverse edge for {(i, j)}"

    # Flags should match edge count
    assert is_bond.shape[0] == all_neighbors.shape[0]


if __name__ == "__main__":
    tests = [
        ("single_bond_and_distance", test_findAllNeighbors_single_bond_and_distance),
        ("overlap_marks_bond", test_findAllNeighbors_overlap_marks_bond),
        ("single_no_neighbors", test_findAllNeighbors_single_no_neighbors),
        ("batched", test_findAllNeighbors_batched),
        ("batched_no_cross_talk", test_findAllNeighbors_batched_no_cross_talk),
        ("properties", test_findAllNeighbors_properties),
    ]

    results = [run_test(name, fn) for name, fn in tests]
    print(f"\nPassed {sum(results)}/{len(results)} tests.")