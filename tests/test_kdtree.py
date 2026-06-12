import numpy as np
import pytest
import torch as pt
from scipy.spatial import KDTree as ScipyKDTree

from chemdm.graph.kdtree import KDTree


def _lex_sort(pairs_np: np.ndarray) -> np.ndarray:
    if pairs_np.size == 0:
        return pairs_np.reshape(0, 2)
    return pairs_np[np.lexsort((pairs_np[:, 1], pairs_np[:, 0]))]


def _scipy_pairs(x_np: np.ndarray, r: float) -> np.ndarray:
    pairs = ScipyKDTree(x_np).query_pairs(r, output_type="ndarray")
    if pairs.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    return pairs.astype(np.int64)


def _brute_pairs(x: pt.Tensor, r: float) -> set:
    d = pt.cdist(x, x)
    i, j = pt.where(d < r)
    mask = i < j
    return {(int(a), int(b)) for a, b in zip(i[mask], j[mask])}


# ---------- native shape / dtype / contract tests ----------

def test_query_pairs_return_shape_and_dtype():
    x = pt.rand(50, 3)
    pairs = KDTree(x).query_pairs(r=0.3)
    assert pairs.dim() == 2 and pairs.shape[1] == 2
    assert pairs.dtype == pt.long
    assert pairs.device == x.device


def test_single_point_no_pairs():
    pairs = KDTree(pt.zeros(1, 3)).query_pairs(r=1.0)
    assert pairs.shape == (0, 2)


def test_two_points_inside_cutoff():
    x = pt.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    pairs = KDTree(x).query_pairs(r=1.0)
    assert pairs.tolist() == [[0, 1]]


def test_two_points_outside_cutoff():
    x = pt.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    pairs = KDTree(x).query_pairs(r=1.0)
    assert pairs.shape == (0, 2)


def test_strict_less_than_radius():
    """Distance exactly r must NOT be emitted; strict < is the contract."""
    x = pt.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    tree = KDTree(x)
    assert tree.query_pairs(r=1.0).shape == (0, 2)
    assert tree.query_pairs(r=1.0001).shape == (1, 2)


def test_all_pairs_when_radius_exceeds_diameter():
    pt.manual_seed(0)
    N = 50
    x = pt.rand(N, 3)
    pairs = KDTree(x).query_pairs(r=10.0)
    assert pairs.shape == (N * (N - 1) // 2, 2)
    assert (pairs[:, 0] < pairs[:, 1]).all()


def test_pairs_have_ordered_indices():
    pt.manual_seed(1)
    pairs = KDTree(pt.rand(200, 3)).query_pairs(r=0.25)
    assert (pairs[:, 0] < pairs[:, 1]).all()


def test_no_duplicate_pairs():
    pt.manual_seed(2)
    pairs = KDTree(pt.rand(300, 3)).query_pairs(r=0.2)
    key = pairs[:, 0] * (pairs[:, 1].max().item() + 1) + pairs[:, 1]
    assert len(set(key.tolist())) == pairs.shape[0]


def test_emitted_distances_below_radius():
    pt.manual_seed(3)
    x = pt.rand(200, 3)
    r = 0.2
    pairs = KDTree(x).query_pairs(r=r)
    d = (x[pairs[:, 0]] - x[pairs[:, 1]]).norm(dim=1)
    assert (d < r).all()


def test_completeness_vs_brute_force():
    pt.manual_seed(4)
    x = pt.rand(80, 3)
    r = 0.25
    kd_set = {(int(a), int(b)) for a, b in KDTree(x).query_pairs(r=r)}
    assert kd_set == _brute_pairs(x, r)


def test_duplicate_points_handled():
    """Coincident points should produce pairs (distance 0 < r)."""
    x = pt.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    pairs = KDTree(x).query_pairs(r=0.5)
    assert pairs.tolist() == [[0, 1]]


def test_grid_known_neighbours():
    """3x3x3 unit grid: cutoff 1.01 gives only face-adjacent pairs."""
    coords = pt.stack(pt.meshgrid(
        pt.arange(3.0), pt.arange(3.0), pt.arange(3.0), indexing="ij"
    ), dim=-1).reshape(-1, 3)
    pairs = KDTree(coords).query_pairs(r=1.01)
    d = (coords[pairs[:, 0]] - coords[pairs[:, 1]]).norm(dim=1)
    assert pt.allclose(d, pt.ones_like(d))
    # 3 axis-aligned directions x 2*9 face-pairs each = 54
    assert pairs.shape[0] == 54


# ---------- scipy comparison ----------

@pytest.mark.parametrize("N", [1, 10, 31, 32, 33, 64, 65, 200, 1000])
@pytest.mark.parametrize("D", [3, 4])
def test_matches_scipy_random_uniform(N, D):
    pt.manual_seed(N * 100 + D)
    x = pt.rand(N, D)
    r = 0.3
    ours = _lex_sort(KDTree(x).query_pairs(r=r).cpu().numpy())
    theirs = _lex_sort(_scipy_pairs(x.numpy(), r))
    np.testing.assert_array_equal(ours, theirs)


@pytest.mark.parametrize("leaf_size", [1, 2, 8, 32, 128])
def test_matches_scipy_across_leaf_sizes(leaf_size):
    pt.manual_seed(7)
    x = pt.rand(400, 3)
    r = 0.15
    ours = _lex_sort(KDTree(x, leaf_size=leaf_size).query_pairs(r=r).cpu().numpy())
    theirs = _lex_sort(_scipy_pairs(x.numpy(), r))
    np.testing.assert_array_equal(ours, theirs)


@pytest.mark.parametrize("r", [0.05, 0.1, 0.3, 0.5, 1.0, 2.0])
def test_matches_scipy_across_radii(r):
    pt.manual_seed(11)
    x = pt.rand(300, 3)
    ours = _lex_sort(KDTree(x).query_pairs(r=r).cpu().numpy())
    theirs = _lex_sort(_scipy_pairs(x.numpy(), r))
    np.testing.assert_array_equal(ours, theirs)


def test_matches_scipy_clustered_points():
    """Two tight clusters far apart — bbox pruning regression."""
    pt.manual_seed(13)
    a = 0.1 * pt.randn(50, 3) + pt.tensor([0.0, 0.0, 0.0])
    b = 0.1 * pt.randn(50, 3) + pt.tensor([10.0, 0.0, 0.0])
    x = pt.cat([a, b], dim=0)
    r = 0.3
    ours = _lex_sort(KDTree(x).query_pairs(r=r).cpu().numpy())
    theirs = _lex_sort(_scipy_pairs(x.numpy(), r))
    np.testing.assert_array_equal(ours, theirs)
    # No cross-cluster edges expected at this radius
    assert ((ours[:, 0] < 50) == (ours[:, 1] < 50)).all()


# ---------- 4D bias-dimension (batched molecules) ----------

def test_4d_bias_excludes_cross_molecule_edges():
    pt.manual_seed(17)
    mol_a = pt.rand(20, 3)
    mol_b = pt.rand(30, 3)
    r = 0.2
    bias = 10.0 * r
    a4 = pt.cat([mol_a, pt.zeros(20, 1)], dim=1)
    b4 = pt.cat([mol_b, pt.full((30, 1), bias)], dim=1)
    x = pt.cat([a4, b4], dim=0)

    pairs = KDTree(x).query_pairs(r=r)
    in_a = (pairs[:, 0] < 20) & (pairs[:, 1] < 20)
    in_b = (pairs[:, 0] >= 20) & (pairs[:, 1] >= 20)
    assert (in_a | in_b).all()


def test_4d_bias_matches_per_molecule_scipy():
    pt.manual_seed(19)
    mol_a = pt.rand(30, 3)
    mol_b = pt.rand(40, 3)
    r = 0.3
    bias = 10.0 * r
    a4 = pt.cat([mol_a, pt.zeros(30, 1)], dim=1)
    b4 = pt.cat([mol_b, pt.full((40, 1), bias)], dim=1)
    x = pt.cat([a4, b4], dim=0)

    ours = _lex_sort(KDTree(x).query_pairs(r=r).cpu().numpy())

    pa = _scipy_pairs(mol_a.numpy(), r)
    pb = _scipy_pairs(mol_b.numpy(), r) + 30
    expected = _lex_sort(np.concatenate([pa, pb], axis=0))
    np.testing.assert_array_equal(ours, expected)


# ---------- query_radius ----------

def test_query_radius_against_brute_force():
    pt.manual_seed(23)
    x = pt.rand(100, 3)
    q = pt.rand(20, 3)
    r = 0.25
    edges = KDTree(x).query_radius(q, r=r)
    d = (q[edges[:, 0]] - x[edges[:, 1]]).norm(dim=1)
    assert (d < r).all()
    # Count check vs brute force
    bf = (pt.cdist(q, x) < r).sum().item()
    assert edges.shape[0] == bf


def test_query_radius_no_self_filtering():
    """query_radius treats queries/data as separate sets; identical points
    on both sides must produce (i, i)-style edges (not filtered)."""
    x = pt.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    edges = KDTree(x).query_radius(x, r=0.1)
    # Each point matches itself
    pairs = {(int(a), int(b)) for a, b in edges}
    assert (0, 0) in pairs and (1, 1) in pairs


def test_query_radius_dimension_check():
    x = pt.rand(10, 3)
    tree = KDTree(x)
    with pytest.raises(ValueError):
        tree.query_radius(pt.rand(5, 4), r=0.1)


# ---------- validation ----------

def test_rejects_non_2d_input():
    with pytest.raises(ValueError):
        KDTree(pt.rand(10))
    with pytest.raises(ValueError):
        KDTree(pt.rand(2, 3, 4))


def test_rejects_empty_input():
    with pytest.raises(ValueError):
        KDTree(pt.empty(0, 3))


def test_rejects_bad_leaf_size():
    with pytest.raises(ValueError):
        KDTree(pt.rand(10, 3), leaf_size=0)


# ---------- device ----------

@pytest.mark.skipif(not pt.backends.mps.is_available(), reason="MPS not available")
def test_runs_on_mps_and_matches_scipy():
    pt.manual_seed(29)
    x_cpu = pt.rand(300, 3)
    x = x_cpu.to("mps")
    r = 0.2
    pairs = KDTree(x).query_pairs(r=r)
    assert pairs.device.type == "mps"
    ours = _lex_sort(pairs.cpu().numpy())
    theirs = _lex_sort(_scipy_pairs(x_cpu.numpy(), r))
    np.testing.assert_array_equal(ours, theirs)


@pytest.mark.skipif(not pt.cuda.is_available(), reason="CUDA not available")
def test_runs_on_cuda_and_matches_scipy():
    pt.manual_seed(31)
    x_cpu = pt.rand(500, 3)
    x = x_cpu.to("cuda")
    r = 0.2
    pairs = KDTree(x).query_pairs(r=r)
    assert pairs.device.type == "cuda"
    ours = _lex_sort(pairs.cpu().numpy())
    theirs = _lex_sort(_scipy_pairs(x_cpu.numpy(), r))
    np.testing.assert_array_equal(ours, theirs)
