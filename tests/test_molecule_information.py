import pytest
import torch as pt

from chemdm.MoleculeGraph import (
    MoleculeGraph,
    BatchedMoleculeGraph,
    batchMolecules,
    detectRing,
)
from chemdm.MoleculeInformation import (
    DEFAULT_ATOMIC_NUMBERS,
    DEFAULT_RING_SIZES,
    MAX_ATOMIC_NUMBER,
    compute_degree,
    computeAtomInformation,
    computeEdgeInformation,
    computeMoleculeInformation,
    edge_ring_features,
    one_hot_atomic_numbers,
    ring_size_flags,
    safe_normalize,
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
    Z: pt.Tensor | None = None,
    x: pt.Tensor | None = None,
    dtype: pt.dtype = pt.float64,
) -> MoleculeGraph:
    if Z is None:
        Z = pt.full((n_atoms,), 6, dtype=pt.long)
    if x is None:
        x = pt.zeros((n_atoms, 3), dtype=dtype)
    return MoleculeGraph(Z=Z, x=x, bonds=directed_edges(undirected_edges))


# ============================================================
# one_hot_atomic_numbers — silent contracts
# ============================================================

def test_one_hot_unknown_atomic_number_is_silently_all_zeros() -> None:
    # Atoms with Z not in allowed_atomic_numbers must produce an all-zero row,
    # not raise and not encode into a "default" column. This is a silent
    # behavior callers can easily miss.
    Z = pt.tensor([6, 99, 8, 200], dtype=pt.long)
    out = one_hot_atomic_numbers(Z, allowed_atomic_numbers=DEFAULT_ATOMIC_NUMBERS)

    assert out.shape == (4, len(DEFAULT_ATOMIC_NUMBERS))
    assert out[0].sum() == 1.0
    assert out[1].sum() == 0.0
    assert out[2].sum() == 1.0
    assert out[3].sum() == 0.0

    c_col = DEFAULT_ATOMIC_NUMBERS.index(6)
    o_col = DEFAULT_ATOMIC_NUMBERS.index(8)
    assert out[0, c_col] == 1.0
    assert out[2, o_col] == 1.0


def test_one_hot_column_order_follows_allowed_list_not_Z() -> None:
    # Column k must correspond to allowed_atomic_numbers[k]. Pinning this
    # prevents a model from silently re-indexing if the list order changes.
    Z = pt.tensor([6, 1, 8], dtype=pt.long)
    allowed = [8, 6, 1]
    out = one_hot_atomic_numbers(Z, allowed_atomic_numbers=allowed)

    expected = pt.tensor([
        [0.0, 1.0, 0.0],  # C → col 1
        [0.0, 0.0, 1.0],  # H → col 2
        [1.0, 0.0, 0.0],  # O → col 0
    ])
    assert pt.allclose(out, expected)


def test_one_hot_duplicate_allowed_entries_set_multiple_columns() -> None:
    # A duplicate in allowed_atomic_numbers will flag both matching columns
    # for that element. Pin the current behavior so accidental dupes are
    # caught by inspection rather than silently doubling the feature width.
    Z = pt.tensor([6], dtype=pt.long)
    out = one_hot_atomic_numbers(Z, allowed_atomic_numbers=[1, 6, 6, 8])
    assert out.tolist() == [[0.0, 1.0, 1.0, 0.0]]


def test_one_hot_default_dtype_is_float32() -> None:
    # The dtype kwarg defaults to pt.float32 — pin so downstream concatenation
    # against float32 features doesn't break if the default is changed.
    out = one_hot_atomic_numbers(pt.tensor([6, 8], dtype=pt.long), [1, 6, 8])
    assert out.dtype == pt.float32


def test_one_hot_empty_Z_returns_correctly_shaped_zero_rows() -> None:
    out = one_hot_atomic_numbers(
        pt.empty((0,), dtype=pt.long),
        allowed_atomic_numbers=DEFAULT_ATOMIC_NUMBERS,
    )
    assert out.shape == (0, len(DEFAULT_ATOMIC_NUMBERS))


# ============================================================
# compute_degree — counts src occurrences, not edges
# ============================================================

def test_compute_degree_counts_src_occurrences_not_undirected_edges() -> None:
    # The function counts how often each atom appears as src. This equals
    # undirected degree ONLY when edges are bidirectional. If the caller
    # accidentally passes a one-directional edge list, degrees are halved.
    n_atoms = 3

    one_way = pt.tensor([[0, 1], [1, 2], [2, 0]], dtype=pt.long)
    assert pt.equal(compute_degree(n_atoms, one_way), pt.tensor([1.0, 1.0, 1.0]))

    two_way = directed_edges([(0, 1), (1, 2), (2, 0)])
    assert pt.equal(compute_degree(n_atoms, two_way), pt.tensor([2.0, 2.0, 2.0]))


def test_compute_degree_self_loops_and_duplicates_are_counted_once_per_occurrence() -> None:
    edges = pt.tensor([
        [0, 0],   # self-loop
        [0, 1],
        [0, 1],   # duplicate edge
        [2, 2],
    ], dtype=pt.long)
    out = compute_degree(3, edges)
    assert pt.equal(out, pt.tensor([3.0, 0.0, 1.0]))


def test_compute_degree_empty_edges_returns_zero_vector_of_right_shape() -> None:
    out = compute_degree(5, pt.empty((0, 2), dtype=pt.long))
    assert out.shape == (5,)
    assert pt.equal(out, pt.zeros(5))


# ============================================================
# ring_size_flags — silent contracts
# ============================================================

def test_ring_size_flags_size_not_in_allowed_is_silently_dropped() -> None:
    # If an atom belongs to a ring whose size is not in allowed_ring_sizes,
    # no flag is set and no error is raised. Pin this so a tightening of the
    # allowed list doesn't quietly erase ring membership from features.
    atom_ring_sizes = [{3}, {11}, {6, 99}, set()]
    out = ring_size_flags(
        atom_ring_sizes,
        allowed_ring_sizes=DEFAULT_RING_SIZES,
        device=pt.device("cpu"),
        dtype=pt.float32,
    )

    col3 = DEFAULT_RING_SIZES.index(3)
    col6 = DEFAULT_RING_SIZES.index(6)

    assert out[0].sum() == 1.0 and out[0, col3] == 1.0
    assert out[1].sum() == 0.0
    assert out[2].sum() == 1.0 and out[2, col6] == 1.0
    assert out[3].sum() == 0.0


def test_ring_size_flags_atom_in_multiple_rings_is_multi_hot() -> None:
    out = ring_size_flags(
        [{3, 6}],
        allowed_ring_sizes=[3, 4, 5, 6],
        device=pt.device("cpu"),
        dtype=pt.float32,
    )
    assert pt.equal(out, pt.tensor([[1.0, 0.0, 0.0, 1.0]]))


def test_ring_size_flags_empty_atom_list_returns_correctly_shaped_zero_tensor() -> None:
    out = ring_size_flags(
        [],
        allowed_ring_sizes=[3, 5, 6],
        device=pt.device("cpu"),
        dtype=pt.float32,
    )
    assert out.shape == (0, 3)
    assert out.dtype == pt.float32


# ============================================================
# safe_normalize
# ============================================================

def test_safe_normalize_zero_vector_returns_finite_zero_not_nan() -> None:
    # Without the eps clamp this would produce NaNs. Equivariant networks
    # that consume unit_dx must never see NaN, so this is load-bearing.
    out = safe_normalize(pt.zeros((3, 3), dtype=pt.float64))
    assert pt.all(pt.isfinite(out))
    assert pt.equal(out, pt.zeros((3, 3), dtype=pt.float64))


def test_safe_normalize_produces_unit_vectors_along_last_dim() -> None:
    x = pt.tensor([
        [3.0, 0.0, 0.0],
        [0.0, 4.0, 0.0],
        [1.0, 1.0, 1.0],
    ], dtype=pt.float64)
    norms = pt.linalg.norm(safe_normalize(x), dim=-1)
    assert pt.allclose(norms, pt.ones(3, dtype=pt.float64))


# ============================================================
# computeAtomInformation
# ============================================================

def test_computeAtomInformation_with_ring_info_None_zero_fills_ring_fields() -> None:
    # Passing ring_info=None on a molecule that DOES have rings must still
    # produce zero-filled ring features without raising. This is the
    # "skip ring detection" path used at training time on hot loops.
    mol = make_molecule(3, [(0, 1), (1, 2), (2, 0)])
    info = computeAtomInformation(mol, ring_info=None)

    assert info.atom_in_ring.dtype == pt.bool
    assert not info.atom_in_ring.any()
    assert pt.equal(info.atom_ring_count, pt.zeros(3, dtype=pt.long))
    assert info.atom_ring_size_flags.sum() == 0
    assert info.atom_ring_size_flags.shape == (3, len(DEFAULT_RING_SIZES))


def test_computeAtomInformation_with_ring_info_propagates_to_atom_ring_size_flags() -> None:
    mol = make_molecule(3, [(0, 1), (1, 2), (2, 0)])
    ring_info = detectRing(mol)
    info = computeAtomInformation(mol, ring_info=ring_info)

    col3 = DEFAULT_RING_SIZES.index(3)
    assert info.atom_in_ring.tolist() == [True, True, True]
    assert info.atom_ring_count.tolist() == [1, 1, 1]
    for atom_idx in range(3):
        assert info.atom_ring_size_flags[atom_idx, col3] == 1.0
        assert info.atom_ring_size_flags[atom_idx].sum() == 1.0


def test_computeAtomInformation_float_dtype_follows_x_not_Z() -> None:
    # Z is always promoted to long, but floating features (mass, one-hot,
    # degree, ring flags) take their dtype from x. Loud regression here
    # would mean float32 leaking into a float64 training run.
    mol = MoleculeGraph(
        Z=pt.tensor([1, 6, 7], dtype=pt.long),
        x=pt.zeros((3, 3), dtype=pt.float64),
        bonds=directed_edges([(0, 1)]),
    )
    info = computeAtomInformation(mol, ring_info=None)
    assert info.atomic_mass.dtype == pt.float64
    assert info.atomic_mass_scaled.dtype == pt.float64
    assert info.atom_type_one_hot.dtype == pt.float64
    assert info.degree.dtype == pt.float64
    assert info.Z.dtype == pt.long


def test_computeAtomInformation_atomic_mass_table_values_and_scaling() -> None:
    Z = pt.tensor([1, 6, 8, 53], dtype=pt.long)  # H, C, O, I (last entry in table)
    mol = MoleculeGraph(
        Z=Z,
        x=pt.zeros((4, 3), dtype=pt.float64),
        bonds=pt.empty((0, 2), dtype=pt.long),
    )
    info = computeAtomInformation(mol, ring_info=None)
    expected = pt.tensor([1.00784, 12.011, 15.999, 126.90447], dtype=pt.float64)
    assert pt.allclose(info.atomic_mass, expected)
    assert pt.allclose(info.atomic_mass_scaled, expected / 100.0)


def test_computeAtomInformation_Z_above_table_raises_index_error() -> None:
    # _ATOMIC_MASS_TABLE has length MAX_ATOMIC_NUMBER + 1 = 54. Anything
    # heavier than iodine (Z=53) is unsupported and must fail loudly rather
    # than read past the end. Pinning this so we notice if the table grows
    # silently or someone replaces the indexing with a clamped lookup.
    mol = MoleculeGraph(
        Z=pt.tensor([6, MAX_ATOMIC_NUMBER + 1], dtype=pt.long),
        x=pt.zeros((2, 3), dtype=pt.float64),
        bonds=pt.empty((0, 2), dtype=pt.long),
    )
    with pytest.raises(IndexError):
        computeAtomInformation(mol, ring_info=None)


def test_computeAtomInformation_respects_allowed_atomic_numbers_kwarg() -> None:
    # Verify the kwarg actually flows through and is not silently replaced
    # by DEFAULT_ATOMIC_NUMBERS. Use a tight list that excludes H so the row
    # for H atom is observably zero.
    mol = MoleculeGraph(
        Z=pt.tensor([1, 6], dtype=pt.long),
        x=pt.zeros((2, 3), dtype=pt.float64),
        bonds=directed_edges([(0, 1)]),
    )
    info = computeAtomInformation(mol, ring_info=None, allowed_atomic_numbers=[6, 8])

    # Width follows the supplied list, not the default of length 10.
    assert info.atom_type_one_hot.shape == (2, 2)
    # H is excluded -> all zeros. C -> column 0.
    assert pt.allclose(
        info.atom_type_one_hot,
        pt.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=pt.float64),
    )


def test_computeAtomInformation_respects_allowed_ring_sizes_kwarg() -> None:
    # Verify the ring-size kwarg actually flows through and that excluding
    # the relevant ring size results in zero size-flags while leaving
    # atom_in_ring and atom_ring_count untouched.
    mol = make_molecule(3, [(0, 1), (1, 2), (2, 0)])
    ring_info = detectRing(mol)

    info = computeAtomInformation(
        mol, ring_info=ring_info, allowed_ring_sizes=[5, 6, 7],
    )
    assert info.atom_ring_size_flags.shape == (3, 3)
    assert info.atom_ring_size_flags.sum() == 0.0
    # The boolean and count fields are independent of allowed_ring_sizes.
    assert info.atom_in_ring.all()
    assert pt.all(info.atom_ring_count == 1)


def test_computeAtomInformation_n_atoms_zero_produces_empty_tensors() -> None:
    mol = MoleculeGraph(
        Z=pt.empty((0,), dtype=pt.long),
        x=pt.empty((0, 3), dtype=pt.float64),
        bonds=pt.empty((0, 2), dtype=pt.long),
    )
    info = computeAtomInformation(mol, ring_info=None)
    assert info.Z.shape == (0,)
    assert info.degree.shape == (0,)
    assert info.atomic_mass.shape == (0,)
    assert info.atom_type_one_hot.shape == (0, len(DEFAULT_ATOMIC_NUMBERS))
    assert info.atom_ring_size_flags.shape == (0, len(DEFAULT_RING_SIZES))


# ============================================================
# edge_ring_features — the semantic gotcha
# ============================================================

def test_edge_ring_features_triangle_all_edges_in_one_ring_of_size_three() -> None:
    edge_index = directed_edges([(0, 1), (1, 2), (2, 0)])
    in_ring, count, flags = edge_ring_features(edge_index, [(0, 1, 2)], [3, 4, 5, 6])

    assert pt.all(in_ring)
    assert pt.equal(count, pt.tensor([1, 1, 1, 1, 1, 1], dtype=pt.long))
    assert pt.equal(flags[:, 0], pt.ones(6, dtype=pt.long))
    assert flags[:, 1:].sum() == 0


def test_edge_ring_features_fused_edge_appears_in_two_rings() -> None:
    # Two triangles sharing edge (1,2). The shared edge should be counted
    # in both rings, while the non-shared edges should be counted in only one.
    edge_index = directed_edges([(0, 1), (1, 2), (2, 0), (1, 3), (3, 2)])
    rings = [(0, 1, 2), (1, 2, 3)]
    in_ring, count, _ = edge_ring_features(edge_index, rings, [3])

    src, dst = edge_index[:, 0], edge_index[:, 1]
    shared_mask = ((src == 1) & (dst == 2)) | ((src == 2) & (dst == 1))
    non_shared_mask = ~shared_mask

    assert pt.all(in_ring)
    assert pt.all(count[shared_mask] == 2)
    assert pt.all(count[non_shared_mask] == 1)


def test_edge_ring_features_flags_any_edge_with_both_endpoints_in_a_ring() -> None:
    # SEMANTIC GOTCHA worth pinning: the function flags an edge whenever
    # src AND dst both belong to a ring's atom set, regardless of whether
    # the edge is actually one of the ring's cycle edges. This matters if
    # edge_index ever contains non-bond neighbors (e.g. distance-cutoff
    # edges) — a chord that is not part of the SSSR will still be flagged
    # as in_ring. Documenting the current contract so any future tightening
    # is a deliberate decision, not an accident.
    bond_edges = directed_edges([
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0),
    ])
    pseudo_chord = pt.tensor([[0, 3], [3, 0]], dtype=pt.long)
    edge_index = pt.cat([bond_edges, pseudo_chord], dim=0)

    rings = [(0, 1, 2, 3, 4, 5)]
    in_ring, count, _ = edge_ring_features(edge_index, rings, [6])

    chord_mask = (edge_index[:, 0] == 0) & (edge_index[:, 1] == 3)
    assert pt.all(in_ring[chord_mask])
    assert pt.all(count[chord_mask] == 1)


def test_edge_ring_features_ring_size_outside_allowed_increments_count_but_not_flag() -> None:
    edge_index = directed_edges([(0, 1), (1, 2), (2, 3), (3, 0)])
    in_ring, count, flags = edge_ring_features(edge_index, [(0, 1, 2, 3)], [3, 5, 6])

    assert pt.all(in_ring)
    assert pt.all(count == 1)
    assert flags.sum() == 0


def test_edge_ring_features_empty_edges_or_empty_rings_return_zeroed_outputs() -> None:
    in_ring, count, flags = edge_ring_features(
        pt.empty((0, 2), dtype=pt.long), [(0, 1, 2)], [3],
    )
    assert in_ring.shape == (0,) and count.shape == (0,) and flags.shape == (0, 1)

    edge_index = directed_edges([(0, 1)])
    in_ring, count, flags = edge_ring_features(edge_index, [], [3])
    assert not in_ring.any()
    assert count.sum() == 0
    assert flags.sum() == 0


# ============================================================
# computeEdgeInformation
# ============================================================

def test_computeEdgeInformation_dx_is_dst_minus_src_and_unit_dx_has_unit_norm() -> None:
    # Pin the direction convention. Equivariant message passing depends on
    # the sign of unit_dx, so a flipped convention is a silent correctness
    # bug that only surfaces during training.
    x = pt.tensor([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]], dtype=pt.float64)
    mol = MoleculeGraph(
        Z=pt.tensor([6, 6], dtype=pt.long),
        x=x,
        bonds=pt.tensor([[0, 1], [1, 0]], dtype=pt.long),
    )
    edges = computeEdgeInformation(mol, ring_info=None)

    assert pt.allclose(edges.dx[0], pt.tensor([3.0, 4.0, 0.0], dtype=pt.float64))
    assert pt.allclose(edges.dx[1], pt.tensor([-3.0, -4.0, 0.0], dtype=pt.float64))
    assert edges.distance.shape == (2, 1)
    assert pt.allclose(edges.distance.squeeze(-1), pt.tensor([5.0, 5.0], dtype=pt.float64))
    assert pt.allclose(pt.linalg.norm(edges.unit_dx, dim=-1), pt.ones(2, dtype=pt.float64))


def test_computeEdgeInformation_zero_length_edge_produces_finite_unit_dx() -> None:
    # Coincident atoms must not produce NaN unit_dx. This is the same
    # safety net as safe_normalize but enforced through the public API.
    mol = MoleculeGraph(
        Z=pt.tensor([6, 6], dtype=pt.long),
        x=pt.zeros((2, 3), dtype=pt.float64),
        bonds=pt.tensor([[0, 1]], dtype=pt.long),
    )
    edges = computeEdgeInformation(mol, ring_info=None)
    assert pt.all(pt.isfinite(edges.unit_dx))
    assert pt.linalg.norm(edges.unit_dx) < 1e-6


def test_computeEdgeInformation_same_molecule_is_None_for_unbatched() -> None:
    edges = computeEdgeInformation(make_molecule(3, [(0, 1), (1, 2)]), ring_info=None)
    assert edges.same_molecule is None


def test_computeEdgeInformation_same_molecule_detects_cross_batch_edges() -> None:
    # `same_molecule` is the only signal that distinguishes within-molecule
    # edges from accidental or constructed cross-molecule edges in a
    # batched graph. Build a raw batch with a deliberate cross-mol edge
    # and check the flag fires.
    Z = pt.tensor([6, 6, 6, 6], dtype=pt.long)
    x = pt.zeros((4, 3), dtype=pt.float64)
    molecule_id = pt.tensor([0, 0, 1, 1], dtype=pt.long)
    edge_index = pt.tensor([[0, 1], [2, 3], [1, 2]], dtype=pt.long)

    batch = BatchedMoleculeGraph.fromRawTensors(
        Z=Z, x=x, edge_index=edge_index, molecule_id=molecule_id,
    )
    edges = computeEdgeInformation(batch, ring_info=None)
    assert edges.same_molecule is not None
    assert edges.same_molecule.tolist() == [True, True, False]


def test_computeEdgeInformation_ring_features_propagate_when_ring_info_given() -> None:
    # Direct test of computeEdgeInformation's ring path (previously only
    # exercised transitively through computeMoleculeInformation).
    mol = make_molecule(3, [(0, 1), (1, 2), (2, 0)])
    edges = computeEdgeInformation(mol, ring_info=detectRing(mol))

    col3 = DEFAULT_RING_SIZES.index(3)
    assert edges.edge_in_ring.all()
    assert pt.all(edges.edge_ring_count == 1)
    assert pt.all(edges.edge_ring_size_flags[:, col3] == 1)
    # All other size columns stay zero.
    mask = pt.ones(len(DEFAULT_RING_SIZES), dtype=pt.bool)
    mask[col3] = False
    assert edges.edge_ring_size_flags[:, mask].sum() == 0


def test_computeEdgeInformation_edge_ring_size_flags_dtype_disagrees_across_branches() -> None:
    # KNOWN INCONSISTENCY worth flagging.
    #
    # edge_ring_features allocates edge_ring_size_flags as pt.long and never
    # casts, so the with-ring branch of computeEdgeInformation returns long.
    # The without-ring branch allocates with dtype=molecule.x.dtype (float).
    # The atom-level analogue (atom_ring_size_flags) is consistently float
    # because ring_size_flags accepts and applies a dtype kwarg.
    #
    # This asymmetry is most likely unintentional. If you decide to fix it,
    # this test will need to be updated to assert the unified dtype.
    mol = make_molecule(3, [(0, 1), (1, 2), (2, 0)])

    edges_with_rings = computeEdgeInformation(mol, ring_info=detectRing(mol))
    edges_without_rings = computeEdgeInformation(mol, ring_info=None)

    assert edges_with_rings.edge_ring_size_flags.dtype == pt.long
    assert edges_without_rings.edge_ring_size_flags.dtype == mol.x.dtype


def test_computeEdgeInformation_empty_edge_index_returns_correctly_shaped_empties() -> None:
    mol = MoleculeGraph(
        Z=pt.tensor([6, 6], dtype=pt.long),
        x=pt.zeros((2, 3), dtype=pt.float64),
        bonds=pt.empty((0, 2), dtype=pt.long),
    )
    edges = computeEdgeInformation(mol, ring_info=None)

    assert edges.dx.shape == (0, 3)
    assert edges.distance.shape == (0, 1)
    assert edges.unit_dx.shape == (0, 3)
    assert edges.edge_in_ring.shape == (0,) and edges.edge_in_ring.dtype == pt.bool
    assert edges.edge_ring_count.shape == (0,)
    assert edges.edge_ring_size_flags.shape == (0, len(DEFAULT_RING_SIZES))
    assert edges.same_molecule is None


# ============================================================
# computeMoleculeInformation — orchestration
# ============================================================

def test_computeMoleculeInformation_include_ring_info_false_zero_fills_atoms_and_edges() -> None:
    # The opt-out path: even on a triangle, ring fields must all be zero
    # and ring_info must be None. This is the fast path during training
    # and a quiet bug here would inject spurious ring features.
    mol = make_molecule(3, [(0, 1), (1, 2), (2, 0)])
    info = computeMoleculeInformation(mol, include_ring_information=False)

    assert info.ring_info is None
    assert not info.atoms.atom_in_ring.any()
    assert info.atoms.atom_ring_count.sum() == 0
    assert info.atoms.atom_ring_size_flags.sum() == 0
    assert not info.edges.edge_in_ring.any()
    assert info.edges.edge_ring_count.sum() == 0


def test_computeMoleculeInformation_include_ring_info_true_propagates_to_atoms_and_edges() -> None:
    mol = make_molecule(3, [(0, 1), (1, 2), (2, 0)])
    info = computeMoleculeInformation(mol, include_ring_information=True)

    assert info.ring_info is not None
    assert info.atoms.atom_in_ring.all()
    assert pt.all(info.atoms.atom_ring_count == 1)
    assert info.edges.edge_in_ring.all()
    assert pt.all(info.edges.edge_ring_count == 1)


def test_computeMoleculeInformation_molecule_id_present_only_for_batched() -> None:
    mol = make_molecule(3, [(0, 1), (1, 2), (2, 0)])
    info_single = computeMoleculeInformation(mol)
    assert info_single.molecule_id is None

    batch = batchMolecules([mol, mol])
    info_batched = computeMoleculeInformation(batch)
    assert info_batched.molecule_id is not None
    assert info_batched.molecule_id.tolist() == [0, 0, 0, 1, 1, 1]


def test_computeMoleculeInformation_n_atoms_and_n_edges_match_inputs() -> None:
    # Hardcode the expected counts rather than deriving them from the same
    # expression the implementation uses (4 undirected bonds -> 8 directed).
    mol = make_molecule(5, [(0, 1), (1, 2), (2, 3), (3, 4)])
    info = computeMoleculeInformation(mol)
    assert info.n_atoms == 5
    assert info.n_edges == 8
    # The fields are plain Python ints, not 0-d tensors -- pin that contract.
    assert isinstance(info.n_atoms, int) and isinstance(info.n_edges, int)


def test_computeMoleculeInformation_batched_atom_fields_equal_per_molecule_concatenation() -> None:
    # Strong correctness check: building features per-molecule then
    # concatenating should give the same atom-level result as building
    # features on the batched graph. This catches off-by-one errors and
    # wrong device coercions in the batched ring path.
    mol1 = make_molecule(3, [(0, 1), (1, 2), (2, 0)])
    mol2 = make_molecule(4, [(0, 1), (1, 2), (2, 3)])

    info_a = computeMoleculeInformation(mol1)
    info_b = computeMoleculeInformation(mol2)
    info_batched = computeMoleculeInformation(batchMolecules([mol1, mol2]))

    for field in ("Z", "atom_in_ring", "atom_ring_count", "degree", "atomic_mass"):
        expected = pt.cat(
            [getattr(info_a.atoms, field), getattr(info_b.atoms, field)],
            dim=0,
        )
        actual = getattr(info_batched.atoms, field)
        assert pt.equal(actual, expected), f"Mismatch on field {field!r}"


@pytest.mark.parametrize(
    "factory",
    [
        lambda mol: computeAtomInformation(mol, ring_info=None),
        lambda mol: computeEdgeInformation(mol, ring_info=None),
        lambda mol: computeMoleculeInformation(mol),
    ],
    ids=["AtomInformation", "EdgeInformation", "MoleculeInformation"],
)
def test_info_dataclasses_are_frozen(factory) -> None:
    # All three info dataclasses are declared frozen=True. A single
    # parametrized test catches the case where a future refactor drops
    # frozen=True on any one of them.
    mol = make_molecule(2, [(0, 1)])
    instance = factory(mol)
    field_name = next(iter(instance.__dataclass_fields__))
    with pytest.raises(Exception):
        setattr(instance, field_name, None)
