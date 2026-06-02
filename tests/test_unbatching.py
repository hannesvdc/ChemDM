import torch as pt
import pytest

from chemdm.MoleculeGraph import MoleculeGraph, BatchedMoleculeGraph, batchMolecules, unbatchBatchedMolecule


def assert_same_edges(actual: pt.Tensor, expected: pt.Tensor) -> None:
    actual_set = set(map(tuple, actual.cpu().long().tolist()))
    expected_set = set(map(tuple, expected.cpu().long().tolist()))
    assert actual_set == expected_set, f"\nActual:\n{actual}\nExpected:\n{expected}"


def assert_same_tensor(actual: pt.Tensor, expected: pt.Tensor) -> None:
    assert actual.shape == expected.shape, (
        f"Shape mismatch: actual {actual.shape}, expected {expected.shape}\n"
        f"Actual:\n{actual}\nExpected:\n{expected}"
    )
    assert pt.allclose(actual, expected), f"\nActual:\n{actual}\nExpected:\n{expected}"



def test_unbatch_single_molecule_returns_one() -> None:
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

    molecules = unbatchBatchedMolecule(mol)

    assert len(molecules) == 1
    assert molecules[0] is mol


def test_unbatch_two_molecules_shapes_and_values() -> None:
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
            [10.0, 1.0, 0.0],
        ]),
        bonds=pt.tensor([
            [0, 1],
            [1, 0],
            [0, 2],
            [2, 0],
        ], dtype=pt.long),
    )

    batch = batchMolecules([mol1, mol2])
    molecules = unbatchBatchedMolecule(batch)

    assert len(molecules) == 2

    assert_same_tensor(molecules[0].Z, mol1.Z)
    assert_same_tensor(molecules[0].x, mol1.x)
    assert_same_edges(molecules[0].edge_index, mol1.edge_index)

    assert_same_tensor(molecules[1].Z, mol2.Z)
    assert_same_tensor(molecules[1].x, mol2.x)
    assert_same_edges(molecules[1].edge_index, mol2.edge_index)


def test_unbatch_local_edge_indices_are_reset() -> None:
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
            [12.0, 0.0, 0.0],
        ]),
        bonds=pt.tensor([
            [0, 1],
            [1, 0],
            [1, 2],
            [2, 1],
        ], dtype=pt.long),
    )

    batch = batchMolecules([mol1, mol2])
    molecules = unbatchBatchedMolecule(batch)

    expected_mol2_edges = pt.tensor([
        [0, 1],
        [1, 0],
        [1, 2],
        [2, 1],
    ], dtype=pt.long)

    assert_same_edges(molecules[1].edge_index, expected_mol2_edges)

    assert int(molecules[1].edge_index.min().item()) == 0
    assert int(molecules[1].edge_index.max().item()) == 2


def test_unbatch_molecule_ids_preserve_order() -> None:
    mol1 = MoleculeGraph(
        Z=pt.tensor([6]),
        x=pt.tensor([[0.0, 0.0, 0.0]]),
        bonds=pt.empty((0, 2), dtype=pt.long),
    )

    mol2 = MoleculeGraph(
        Z=pt.tensor([7, 1]),
        x=pt.tensor([
            [10.0, 0.0, 0.0],
            [11.0, 0.0, 0.0],
        ]),
        bonds=pt.tensor([
            [0, 1],
            [1, 0],
        ], dtype=pt.long),
    )

    mol3 = MoleculeGraph(
        Z=pt.tensor([8, 1, 1]),
        x=pt.tensor([
            [20.0, 0.0, 0.0],
            [21.0, 0.0, 0.0],
            [20.0, 1.0, 0.0],
        ]),
        bonds=pt.tensor([
            [0, 1],
            [1, 0],
            [0, 2],
            [2, 0],
        ], dtype=pt.long),
    )

    batch = batchMolecules([mol1, mol2, mol3])
    molecules = unbatchBatchedMolecule(batch)

    assert len(molecules) == 3

    assert_same_tensor(molecules[0].Z, mol1.Z)
    assert_same_tensor(molecules[1].Z, mol2.Z)
    assert_same_tensor(molecules[2].Z, mol3.Z)


def test_unbatch_nested_batched_molecule() -> None:
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
        Z=pt.tensor([8, 1]),
        x=pt.tensor([
            [10.0, 0.0, 0.0],
            [11.0, 0.0, 0.0],
        ]),
        bonds=pt.tensor([
            [0, 1],
            [1, 0],
        ], dtype=pt.long),
    )

    mol3 = MoleculeGraph(
        Z=pt.tensor([7, 1, 1]),
        x=pt.tensor([
            [20.0, 0.0, 0.0],
            [21.0, 0.0, 0.0],
            [20.0, 1.0, 0.0],
        ]),
        bonds=pt.tensor([
            [0, 1],
            [1, 0],
            [0, 2],
            [2, 0],
        ], dtype=pt.long),
    )

    batch12 = batchMolecules([mol1, mol2])
    nested = batchMolecules([batch12, mol3])

    molecules = unbatchBatchedMolecule(nested)

    assert len(molecules) == 3

    assert_same_tensor(molecules[0].Z, mol1.Z)
    assert_same_tensor(molecules[1].Z, mol2.Z)
    assert_same_tensor(molecules[2].Z, mol3.Z)

    assert_same_edges(molecules[0].edge_index, mol1.edge_index)
    assert_same_edges(molecules[1].edge_index, mol2.edge_index)
    assert_same_edges(molecules[2].edge_index, mol3.edge_index)


def test_unbatch_empty_edges() -> None:
    mol1 = MoleculeGraph(
        Z=pt.tensor([6, 1]),
        x=pt.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]),
        bonds=pt.empty((0, 2), dtype=pt.long),
    )

    mol2 = MoleculeGraph(
        Z=pt.tensor([8]),
        x=pt.tensor([[10.0, 0.0, 0.0]]),
        bonds=pt.empty((0, 2), dtype=pt.long),
    )

    batch = batchMolecules([mol1, mol2])
    molecules = unbatchBatchedMolecule(batch)

    assert len(molecules) == 2

    assert molecules[0].edge_index.shape == (0, 2)
    assert molecules[1].edge_index.shape == (0, 2)

    assert_same_tensor(molecules[0].Z, mol1.Z)
    assert_same_tensor(molecules[1].Z, mol2.Z)


def test_unbatch_from_raw_tensors_noncontiguous_atoms() -> None:
    # This tests the robust global_to_local mapping behavior.
    # molecule_id is deliberately interleaved:
    # molecule 0 atoms: global 0, 2
    # molecule 1 atoms: global 1, 3
    Z = pt.tensor([6, 8, 1, 1])
    x = pt.tensor([
        [0.0, 0.0, 0.0],   # mol 0 local 0
        [10.0, 0.0, 0.0],  # mol 1 local 0
        [1.0, 0.0, 0.0],   # mol 0 local 1
        [11.0, 0.0, 0.0],  # mol 1 local 1
    ])
    molecule_id = pt.tensor([0, 1, 0, 1])

    edge_index = pt.tensor([
        [0, 2],
        [2, 0],
        [1, 3],
        [3, 1],
    ], dtype=pt.long)

    batch = BatchedMoleculeGraph.fromRawTensors(
        Z=Z,
        x=x,
        edge_index=edge_index,
        molecule_id=molecule_id,
    )

    molecules = unbatchBatchedMolecule(batch)

    assert len(molecules) == 2

    assert_same_tensor(molecules[0].Z, pt.tensor([6, 1]))
    assert_same_tensor(molecules[1].Z, pt.tensor([8, 1]))

    expected_edges = pt.tensor([
        [0, 1],
        [1, 0],
    ], dtype=pt.long)

    assert_same_edges(molecules[0].edge_index, expected_edges)
    assert_same_edges(molecules[1].edge_index, expected_edges)


def test_unbatch_cross_molecule_edges_raise() -> None:
    Z = pt.tensor([6, 1, 8, 1])
    x = pt.zeros((4, 3))
    molecule_id = pt.tensor([0, 0, 1, 1])

    edge_index = pt.tensor([
        [0, 1],
        [1, 0],
        [2, 3],
        [3, 2],
        [1, 2],  # invalid cross-molecule edge
    ], dtype=pt.long)

    batch = BatchedMoleculeGraph.fromRawTensors(
        Z=Z,
        x=x,
        edge_index=edge_index,
        molecule_id=molecule_id,
    )

    did_raise = False
    try:
        _ = unbatchBatchedMolecule(batch)
    except ValueError:
        did_raise = True

    assert did_raise, "Expected unbatchBatchedMolecule to raise on cross-molecule edge."
