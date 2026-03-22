import torch as pt

from chemdm.graph.MoleculeGraph import MoleculeGraph, batchMolecules, findAllDistanceNeighbors

def assert_same_edges(actual: pt.Tensor, expected: pt.Tensor) -> None:
    actual_set = set(map(tuple, actual.tolist()))
    expected_set = set(map(tuple, expected.tolist()))
    assert actual_set == expected_set, f"\nActual:\n{actual}\nExpected:\n{expected}"

def test_findAllDistanceNeighbors_single() -> None:
    mol = MoleculeGraph(
        Z=pt.tensor([6, 1, 1]),
        x=pt.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ]),
        bonds=pt.empty((0, 2), dtype=pt.long),
    )

    edges = findAllDistanceNeighbors(mol, cutoff=1.5)

    expected = pt.tensor([
        [0, 1],
        [1, 0],
    ], dtype=pt.long)

    assert_same_edges(edges, expected)
    print("test_findAllDistanceNeighbors_single passed")


def test_findAllDistanceNeighbors_single_none() -> None:
    mol = MoleculeGraph(
        Z=pt.tensor([6, 1]),
        x=pt.tensor([
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ]),
        bonds=pt.empty((0, 2), dtype=pt.long),
    )

    edges = findAllDistanceNeighbors(mol, cutoff=1.0)
    expected = pt.empty((0, 2), dtype=pt.long)

    assert_same_edges(edges, expected)
    print("test_findAllDistanceNeighbors_single_none passed")


def test_findAllDistanceNeighbors_batched() -> None:
    mol1 = MoleculeGraph(
        Z=pt.tensor([6, 1]),
        x=pt.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]),
        bonds=pt.empty((0, 2), dtype=pt.long),
    )

    mol2 = MoleculeGraph(
        Z=pt.tensor([8, 1, 1]),
        x=pt.tensor([
            [10.0, 0.0, 0.0],
            [11.0, 0.0, 0.0],
            [14.0, 0.0, 0.0],
        ]),
        bonds=pt.empty((0, 2), dtype=pt.long),
    )

    batch = batchMolecules([mol1, mol2])
    edges = findAllDistanceNeighbors(batch, cutoff=1.5)

    expected = pt.tensor([
        [0, 1], [1, 0],   # mol1
        [2, 3], [3, 2],   # mol2 (offset by 2)
    ], dtype=pt.long)

    assert_same_edges(edges, expected)
    print("test_findAllDistanceNeighbors_batched passed")


def test_findAllDistanceNeighbors_batched_no_cross_talk() -> None:
    # Same coordinates in two different molecules.
    # In a broken batched implementation, these would become neighbors.
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
    edges = findAllDistanceNeighbors(batch, cutoff=1.5)

    expected = pt.tensor([
        [0, 1], [1, 0],   # only inside mol1
        [2, 3], [3, 2],   # only inside mol2
    ], dtype=pt.long)

    assert_same_edges(edges, expected)
    print("test_findAllDistanceNeighbors_batched_no_cross_talk passed")


def test_findAllDistanceNeighbors_properties() -> None:
    mol = MoleculeGraph(
        Z=pt.tensor([6, 1, 1]),
        x=pt.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.2, 0.0, 0.0],
        ]),
        bonds=pt.empty((0, 2), dtype=pt.long),
    )

    edges = findAllDistanceNeighbors(mol, cutoff=1.5)

    # no self-edges
    assert not pt.any(edges[:, 0] == edges[:, 1]), f"Found self-edges:\n{edges}"

    # symmetry: if (i,j) exists then (j,i) exists
    edge_set = set(map(tuple, edges.tolist()))
    for i, j in edge_set:
        assert (j, i) in edge_set, f"Missing reverse edge for {(i, j)}"

    print("test_findAllDistanceNeighbors_properties passed")

def run_test(name: str, test_fn) -> bool:
    try:
        test_fn()
        print(f"[PASS] {name}\n")
        return True
    except AssertionError as e:
        print(f"[FAIL] {name}\n")
        print(e)
        return False
    except Exception as e:
        print(f"[ERROR] {name}\n")
        print(e)
        return False

if __name__ == "__main__":
    tests = [
        ("single", test_findAllDistanceNeighbors_single),
        ("single_none", test_findAllDistanceNeighbors_single_none),
        ("batched", test_findAllDistanceNeighbors_batched),
        ("batched_no_cross_talk", test_findAllDistanceNeighbors_batched_no_cross_talk),
        ("properties", test_findAllDistanceNeighbors_properties),
    ]

    results = [run_test(name, fn) for name, fn in tests]
    n_passed = sum(results)
    n_total = len(results)

    print("---")
    print(f"Passed {n_passed}/{n_total} tests.")