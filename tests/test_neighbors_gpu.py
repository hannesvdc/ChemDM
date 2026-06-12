"""Tests for the on-device neighbor functions in `chemdm.MoleculeGraph`.

Pin two contracts:
1. `findAllDistanceNeighbors_gpu` returns the same edge set as the scipy-backed
   `findAllDistanceNeighbors`, including the batched no-cross-talk guarantee.
2. `findAllNeighbors_gpu` returns the same (edge_set, is_bond) as the
   `pt.unique`-based `findAllNeighbors`, without using `pt.unique(dim=0)`
   or `scatter_reduce` (the MPS-syncing patterns it replaces).
"""
import torch as pt
import pytest

from chemdm.MoleculeGraph import (
    MoleculeGraph,
    batchMolecules,
    findAllDistanceNeighbors,
    findAllDistanceNeighbors_gpu,
    findAllNeighbors,
    findAllNeighbors_gpu,
    findAllNeighborsReactantProduct,
    findAllNeighborsReactantProduct_gpu,
    findFixedUnionNeighbors,
    findFixedUnionNeighbors_gpu,
)


def _edge_set(edges: pt.Tensor) -> set:
    return set(map(tuple, edges.tolist()))


def _edge_to_isbond_map(edges: pt.Tensor, is_bond: pt.Tensor) -> dict:
    return {tuple(e.tolist()): float(b.item()) for e, b in zip(edges, is_bond)}


# ---------- findAllDistanceNeighbors_gpu ----------

def test_distance_gpu_matches_cpu_single():
    pt.manual_seed(0)
    mol = MoleculeGraph(
        Z=pt.tensor([6, 1, 1, 8]),
        x=pt.rand(4, 3),
        bonds=pt.empty((0, 2), dtype=pt.long),
    )
    cpu = findAllDistanceNeighbors(mol, cutoff=0.6)
    gpu = findAllDistanceNeighbors_gpu(mol, cutoff=0.6)
    assert _edge_set(gpu) == _edge_set(cpu)


def test_distance_gpu_symmetric_no_self_edges():
    pt.manual_seed(1)
    mol = MoleculeGraph(
        Z=pt.tensor([6, 1, 1]),
        x=pt.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.2, 0.0, 0.0]]),
        bonds=pt.empty((0, 2), dtype=pt.long),
    )
    edges = findAllDistanceNeighbors_gpu(mol, cutoff=1.5)
    assert not pt.any(edges[:, 0] == edges[:, 1])
    edges_set = _edge_set(edges)
    for i, j in edges_set:
        assert (j, i) in edges_set


def test_distance_gpu_empty_below_cutoff():
    mol = MoleculeGraph(
        Z=pt.tensor([6, 1]),
        x=pt.tensor([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]),
        bonds=pt.empty((0, 2), dtype=pt.long),
    )
    edges = findAllDistanceNeighbors_gpu(mol, cutoff=1.0)
    assert edges.shape == (0, 2)
    assert edges.dtype == pt.long


def test_distance_gpu_batched_matches_cpu():
    pt.manual_seed(2)
    mol1 = MoleculeGraph(
        Z=pt.tensor([6, 1, 1]),
        x=pt.rand(3, 3),
        bonds=pt.empty((0, 2), dtype=pt.long),
    )
    mol2 = MoleculeGraph(
        Z=pt.tensor([8, 1, 1, 6]),
        x=pt.rand(4, 3) + 100.0,  # offset so no spurious overlap
        bonds=pt.empty((0, 2), dtype=pt.long),
    )
    batch = batchMolecules([mol1, mol2])
    assert _edge_set(findAllDistanceNeighbors_gpu(batch, cutoff=0.5)) == \
           _edge_set(findAllDistanceNeighbors(batch, cutoff=0.5))


def test_distance_gpu_batched_no_cross_talk():
    """Identical coordinates in two batched molecules → no cross-molecule edges."""
    coords = pt.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    mol1 = MoleculeGraph(
        Z=pt.tensor([6, 1]), x=coords.clone(), bonds=pt.empty((0, 2), dtype=pt.long),
    )
    mol2 = MoleculeGraph(
        Z=pt.tensor([8, 1]), x=coords.clone(), bonds=pt.empty((0, 2), dtype=pt.long),
    )
    batch = batchMolecules([mol1, mol2])
    edges = findAllDistanceNeighbors_gpu(batch, cutoff=1.5)
    expected = pt.tensor([[0, 1], [1, 0], [2, 3], [3, 2]], dtype=pt.long)
    assert _edge_set(edges) == _edge_set(expected)


# ---------- findAllNeighbors_gpu ----------

def test_neighbors_gpu_no_bonds():
    pt.manual_seed(3)
    mol = MoleculeGraph(
        Z=pt.tensor([6, 1, 1]),
        x=pt.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [3.0, 0.0, 0.0]]),
        bonds=pt.empty((0, 2), dtype=pt.long),
    )
    cpu_e, cpu_b = findAllNeighbors(mol, d_cutoff=1.0)
    gpu_e, gpu_b = findAllNeighbors_gpu(mol, d_cutoff=1.0)
    assert _edge_to_isbond_map(gpu_e, gpu_b) == _edge_to_isbond_map(cpu_e, cpu_b)


def test_neighbors_gpu_disjoint_bond_and_distance():
    """Bond between far atoms; distance edge between near ones — no overlap."""
    mol = MoleculeGraph(
        Z=pt.tensor([6, 1, 1]),
        x=pt.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [10.0, 0.0, 0.0]]),
        bonds=pt.tensor([[0, 2], [2, 0]], dtype=pt.long),
    )
    gpu_e, gpu_b = findAllNeighbors_gpu(mol, d_cutoff=1.0)
    m = _edge_to_isbond_map(gpu_e, gpu_b)
    assert m == {(0, 2): 1.0, (2, 0): 1.0, (0, 1): 0.0, (1, 0): 0.0}


def test_neighbors_gpu_overlapping_bond_and_distance():
    """Same pair appears in both bonds and distance → must dedupe and flag as bond."""
    mol = MoleculeGraph(
        Z=pt.tensor([6, 1]),
        x=pt.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        bonds=pt.tensor([[0, 1], [1, 0]], dtype=pt.long),
    )
    gpu_e, gpu_b = findAllNeighbors_gpu(mol, d_cutoff=1.0)
    m = _edge_to_isbond_map(gpu_e, gpu_b)
    assert m == {(0, 1): 1.0, (1, 0): 1.0}


def test_neighbors_gpu_matches_cpu_random_single():
    pt.manual_seed(4)
    N = 30
    mol = MoleculeGraph(
        Z=pt.randint(1, 10, (N,)),
        x=pt.rand(N, 3),
        bonds=pt.tensor([[0, 1], [1, 0], [5, 7], [7, 5], [10, 12], [12, 10]], dtype=pt.long),
    )
    cpu_e, cpu_b = findAllNeighbors(mol, d_cutoff=0.4)
    gpu_e, gpu_b = findAllNeighbors_gpu(mol, d_cutoff=0.4)
    assert _edge_to_isbond_map(gpu_e, gpu_b) == _edge_to_isbond_map(cpu_e, cpu_b)


def test_neighbors_gpu_matches_cpu_batched():
    pt.manual_seed(5)
    mols = []
    for _ in range(3):
        N = int(pt.randint(5, 12, (1,)).item())
        mols.append(MoleculeGraph(
            Z=pt.randint(1, 10, (N,)),
            x=pt.rand(N, 3),
            bonds=pt.tensor([[0, 1], [1, 0]], dtype=pt.long) if N >= 2 else pt.empty((0, 2), dtype=pt.long),
        ))
    batch = batchMolecules(mols)
    cpu_e, cpu_b = findAllNeighbors(batch, d_cutoff=0.4)
    gpu_e, gpu_b = findAllNeighbors_gpu(batch, d_cutoff=0.4)
    assert _edge_to_isbond_map(gpu_e, gpu_b) == _edge_to_isbond_map(cpu_e, cpu_b)


def test_neighbors_gpu_is_bond_dtype_and_shape():
    mol = MoleculeGraph(
        Z=pt.tensor([6, 1]),
        x=pt.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        bonds=pt.tensor([[0, 1], [1, 0]], dtype=pt.long),
    )
    edges, is_bond = findAllNeighbors_gpu(mol, d_cutoff=1.0)
    assert edges.dtype == pt.long
    assert is_bond.dtype == pt.float32
    assert is_bond.shape == (edges.shape[0],)


# ---------- findAllNeighborsReactantProduct_gpu ----------

def _edge_flags_map(edges: pt.Tensor, *flags: pt.Tensor) -> dict:
    return {tuple(e.tolist()): tuple(float(f[i].item()) for f in flags) for i, e in enumerate(edges)}


def test_reactant_product_gpu_disjoint_bonds():
    Z = pt.tensor([6, 1, 1])
    xA = pt.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    xB = pt.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    bonds_A = pt.tensor([[0, 1], [1, 0]], dtype=pt.long)
    bonds_B = pt.tensor([[0, 2], [2, 0]], dtype=pt.long)
    molA = MoleculeGraph(Z=Z, x=xA, bonds=bonds_A)
    molB = MoleculeGraph(Z=Z, x=xB, bonds=bonds_B)
    x_mid = (xA + xB) / 2.0

    e_c, ab_c, bb_c = findAllNeighborsReactantProduct(molA, molB, x_mid, d_cutoff=1.0)
    e_g, ab_g, bb_g = findAllNeighborsReactantProduct_gpu(molA, molB, x_mid, d_cutoff=1.0)
    assert _edge_flags_map(e_g, ab_g, bb_g) == _edge_flags_map(e_c, ab_c, bb_c)


def test_reactant_product_gpu_matches_cpu_random_single():
    pt.manual_seed(11)
    N = 25
    Z = pt.randint(1, 10, (N,))
    bonds_A = pt.tensor([[0, 1], [1, 0], [3, 4], [4, 3]], dtype=pt.long)
    bonds_B = pt.tensor([[0, 1], [1, 0], [7, 8], [8, 7]], dtype=pt.long)  # overlaps A on (0,1)
    molA = MoleculeGraph(Z=Z, x=pt.rand(N, 3),       bonds=bonds_A)
    molB = MoleculeGraph(Z=Z, x=pt.rand(N, 3) + 0.1, bonds=bonds_B)
    x_mid = (molA.x + molB.x) / 2.0

    e_c, ab_c, bb_c = findAllNeighborsReactantProduct(molA, molB, x_mid, d_cutoff=0.4)
    e_g, ab_g, bb_g = findAllNeighborsReactantProduct_gpu(molA, molB, x_mid, d_cutoff=0.4)
    assert _edge_flags_map(e_g, ab_g, bb_g) == _edge_flags_map(e_c, ab_c, bb_c)


def test_reactant_product_gpu_matches_cpu_batched():
    pt.manual_seed(13)
    mols_A, mols_B = [], []
    for _ in range(3):
        N = int(pt.randint(5, 12, (1,)).item())
        Z = pt.randint(1, 10, (N,))
        b = pt.tensor([[0, 1], [1, 0]], dtype=pt.long) if N >= 2 else pt.empty((0, 2), dtype=pt.long)
        mols_A.append(MoleculeGraph(Z=Z, x=pt.rand(N, 3),       bonds=b))
        mols_B.append(MoleculeGraph(Z=Z, x=pt.rand(N, 3) + 0.1, bonds=b))
    batchA = batchMolecules(mols_A)
    batchB = batchMolecules(mols_B)
    x_mid  = (batchA.x + batchB.x) / 2.0

    e_c, ab_c, bb_c = findAllNeighborsReactantProduct(batchA, batchB, x_mid, d_cutoff=0.4)
    e_g, ab_g, bb_g = findAllNeighborsReactantProduct_gpu(batchA, batchB, x_mid, d_cutoff=0.4)
    assert _edge_flags_map(e_g, ab_g, bb_g) == _edge_flags_map(e_c, ab_c, bb_c)


def test_reactant_product_gpu_flag_semantics():
    """Edge in both bond sets → both flags True. Distance-only edge → both False."""
    Z = pt.tensor([6, 1])
    xA = pt.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])  # within 1.0
    xB = pt.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    bonds_A = pt.tensor([[0, 1], [1, 0]], dtype=pt.long)
    bonds_B = pt.tensor([[0, 1], [1, 0]], dtype=pt.long)
    molA = MoleculeGraph(Z=Z, x=xA, bonds=bonds_A)
    molB = MoleculeGraph(Z=Z, x=xB, bonds=bonds_B)

    e, ab, bb = findAllNeighborsReactantProduct_gpu(molA, molB, xA, d_cutoff=1.0)
    m = _edge_flags_map(e, ab, bb)
    assert m == {(0, 1): (1.0, 1.0), (1, 0): (1.0, 1.0)}


# ---------- findFixedUnionNeighbors_gpu ----------

def test_fixed_union_gpu_matches_cpu_single():
    pt.manual_seed(17)
    N = 20
    Z = pt.randint(1, 10, (N,))
    bonds_A = pt.tensor([[0, 1], [1, 0], [3, 4], [4, 3]], dtype=pt.long)
    bonds_B = pt.tensor([[0, 1], [1, 0], [5, 6], [6, 5]], dtype=pt.long)
    molA = MoleculeGraph(Z=Z, x=pt.rand(N, 3),       bonds=bonds_A)
    molB = MoleculeGraph(Z=Z, x=pt.rand(N, 3) + 0.05, bonds=bonds_B)

    e_c, ab_c, bb_c = findFixedUnionNeighbors(molA, molB, d_cutoff=0.4)
    e_g, ab_g, bb_g = findFixedUnionNeighbors_gpu(molA, molB, d_cutoff=0.4)
    assert _edge_flags_map(e_g, ab_g, bb_g) == _edge_flags_map(e_c, ab_c, bb_c)


def test_fixed_union_gpu_matches_cpu_batched():
    pt.manual_seed(19)
    mols_A, mols_B = [], []
    for _ in range(3):
        N = int(pt.randint(5, 12, (1,)).item())
        Z = pt.randint(1, 10, (N,))
        b = pt.tensor([[0, 1], [1, 0]], dtype=pt.long) if N >= 2 else pt.empty((0, 2), dtype=pt.long)
        mols_A.append(MoleculeGraph(Z=Z, x=pt.rand(N, 3),         bonds=b))
        mols_B.append(MoleculeGraph(Z=Z, x=pt.rand(N, 3) + 0.05,  bonds=b))
    batchA = batchMolecules(mols_A)
    batchB = batchMolecules(mols_B)

    e_c, ab_c, bb_c = findFixedUnionNeighbors(batchA, batchB, d_cutoff=0.4)
    e_g, ab_g, bb_g = findFixedUnionNeighbors_gpu(batchA, batchB, d_cutoff=0.4)
    assert _edge_flags_map(e_g, ab_g, bb_g) == _edge_flags_map(e_c, ab_c, bb_c)


# ---------- MPS device ----------

@pytest.mark.skipif(not pt.backends.mps.is_available(), reason="MPS not available")
def test_neighbors_gpu_runs_on_mps():
    pt.manual_seed(7)
    mol = MoleculeGraph(
        Z=pt.randint(1, 10, (20,)),
        x=pt.rand(20, 3),
        bonds=pt.tensor([[0, 1], [1, 0], [2, 3], [3, 2]], dtype=pt.long),
    ).to(device=pt.device("mps"), dtype=pt.float32)
    edges, is_bond = findAllNeighbors_gpu(mol, d_cutoff=0.4)
    assert edges.device.type == "mps" and is_bond.device.type == "mps"
