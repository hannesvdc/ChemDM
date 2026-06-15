"""
Tests for chemdm.lanczos.lanczos_lowest.

Strategy: build matrices whose eigenvalues and eigenvectors we know in closed
form, run Lanczos against them via a matvec, and verify the smallest eigvalue
and (up-to-sign) the eigenvector. Also exercise the V_project deflation, the
matrix-free path, and the seed-noise breakdown workaround.
"""

import numpy as np
import pytest

from chemdm.lanczos import lanczos_lowest


# ---------- helpers ----------

def make_symmetric_with_spectrum( eigvals: np.ndarray, seed: int = 0 ) -> tuple[np.ndarray, np.ndarray]:
    """Return (H, U) such that H = U diag(eigvals) U.T with U random orthogonal."""
    rng = np.random.default_rng(seed)
    n = eigvals.size
    A = rng.standard_normal((n, n))
    Q, _ = np.linalg.qr(A)                 # random orthogonal
    H = Q @ np.diag(eigvals) @ Q.T
    H = 0.5 * (H + H.T)                    # symmetrise against roundoff
    return H, Q

def matvec_from_matrix( H: np.ndarray ):
    return lambda v: H @ v


# ---------- diagonal matrix ----------

def test_diagonal_matrix_returns_smallest_eigval_and_eigvec():
    eigvals = np.array([3.0, 1.0, 5.0, 2.0])
    H = np.diag(eigvals)
    v0 = np.ones(4) / 2.0
    u, lam = lanczos_lowest(matvec_from_matrix(H), v0, max_iter=10)

    assert abs(lam - 1.0) < 1e-10
    # Smallest eigval (1.0) is at index 1 → eigvec is e_1.
    assert abs(abs(u[1]) - 1.0) < 1e-8
    assert np.allclose(u[[0, 2, 3]], 0.0, atol=1e-8)


# ---------- random symmetric matrix ----------

@pytest.mark.parametrize("n,seed", [(20, 0), (50, 1), (100, 7)])
def test_random_symmetric_smallest_eigval(n, seed):
    rng = np.random.default_rng(seed)
    # Construct a spectrum with a well-separated smallest eigenvalue so
    # Lanczos converges quickly.
    eigvals = np.sort(rng.uniform(2.0, 10.0, size=n))
    eigvals[0] = 0.5
    H, U = make_symmetric_with_spectrum(eigvals, seed=seed)

    v0 = rng.standard_normal(n)
    u, lam = lanczos_lowest(matvec_from_matrix(H), v0, max_iter=30, tol=1e-12)

    # Smallest Ritz value should match smallest true eigenvalue to high precision.
    assert abs(lam - eigvals[0]) < 1e-9

    # Up-to-sign overlap with the true smallest eigenvector should be ~1.
    true_v = U[:, 0]
    assert abs(abs(u @ true_v) - 1.0) < 1e-6


# ---------- negative eigenvalue present ----------

def test_finds_negative_eigenvalue():
    # Smallest is genuinely negative — Lanczos should return it.
    eigvals = np.array([-2.5, 1.0, 3.0, 7.5])
    H, U = make_symmetric_with_spectrum(eigvals, seed=42)

    rng = np.random.default_rng(0)
    v0 = rng.standard_normal(4)
    u, lam = lanczos_lowest(matvec_from_matrix(H), v0, max_iter=10, tol=1e-12)

    assert abs(lam - (-2.5)) < 1e-10
    assert abs(abs(u @ U[:, 0]) - 1.0) < 1e-8


# ---------- projection deflation ----------

def test_projection_deflates_one_eigenmode():
    # Project out the lowest mode; Lanczos should return the SECOND smallest.
    eigvals = np.array([0.5, 1.7, 3.0, 8.0])
    H, U = make_symmetric_with_spectrum(eigvals, seed=11)

    # V_project = the lowest eigenvector (orthonormal column).
    V = U[:, 0:1]

    rng = np.random.default_rng(0)
    v0 = rng.standard_normal(4)
    u, lam = lanczos_lowest(
        matvec_from_matrix(H), v0,
        max_iter=10, tol=1e-12, V_project=V,
    )

    # Should land at the second eigenvalue.
    assert abs(lam - eigvals[1]) < 1e-9
    # Returned vector should be orthogonal to the deflated mode.
    assert abs(u @ U[:, 0]) < 1e-8
    # And have unit overlap with the second eigenvector.
    assert abs(abs(u @ U[:, 1]) - 1.0) < 1e-6


def test_projection_deflates_multiple_eigenmodes():
    # Project out the bottom k=2 modes; Lanczos should return the third smallest.
    eigvals = np.array([0.2, 0.5, 1.0, 4.0, 7.5])
    H, U = make_symmetric_with_spectrum(eigvals, seed=3)

    V = U[:, 0:2]                          # deflate two modes

    rng = np.random.default_rng(0)
    v0 = rng.standard_normal(5)
    u, lam = lanczos_lowest(
        matvec_from_matrix(H), v0,
        max_iter=15, tol=1e-12, V_project=V,
    )

    assert abs(lam - eigvals[2]) < 1e-9
    assert abs(u @ U[:, 0]) < 1e-8
    assert abs(u @ U[:, 1]) < 1e-8


# ---------- matrix-free matvec ----------

def test_matrix_free_matvec_works():
    # Pass a lambda that computes Hv without ever forming H explicitly.
    n = 30
    eigvals = np.arange(1.0, n + 1.0)
    # H = U diag(eigvals) U^T applied lazily.
    rng = np.random.default_rng(5)
    A = rng.standard_normal((n, n))
    Q, _ = np.linalg.qr(A)
    diag = eigvals.copy()

    def matvec(v):
        # H v = Q (diag * (Q^T v))
        return Q @ (diag * (Q.T @ v))

    v0 = rng.standard_normal(n)
    u, lam = lanczos_lowest(matvec, v0, max_iter=20, tol=1e-12)

    # Closely-spaced spectrum (gap of 1 between adjacent eigvals) gives
    # moderate Lanczos convergence — ~1e-7 in 20 iterations is realistic.
    assert abs(lam - 1.0) < 1e-6
    assert abs(abs(u @ Q[:, 0]) - 1.0) < 1e-3


# ---------- seed-noise breakdown workaround ----------

def test_seed_exactly_higher_eigenvector_breaks_down_without_noise():
    # If v0 IS an eigenvector for a non-target eigenvalue, Lanczos breaks
    # down at step 1 (β=0). The returned Ritz value is just that eigenvector's
    # eigenvalue, NOT the smallest.
    eigvals = np.array([0.5, 1.0, 3.0])
    H, U = make_symmetric_with_spectrum(eigvals, seed=7)

    v0_bad = U[:, 2].copy()    # eigenvector of the LARGEST eigval
    u, lam = lanczos_lowest(
        matvec_from_matrix(H), v0_bad,
        max_iter=10, tol=1e-12, seed_noise=0.0,
    )
    # Without noise, Lanczos returns the (only) Ritz value it could compute.
    # It's NOT the smallest true eigenvalue.
    assert abs(lam - 3.0) < 1e-9
    assert abs(lam - 0.5) > 1.0    # explicitly not the true min


def test_seed_noise_recovers_when_seed_is_an_eigenvector():
    eigvals = np.array([0.5, 1.0, 3.0])
    H, U = make_symmetric_with_spectrum(eigvals, seed=7)

    v0_bad = U[:, 2].copy()
    u, lam = lanczos_lowest(
        matvec_from_matrix(H), v0_bad,
        max_iter=20, tol=1e-12, seed_noise=1e-2, random_state=0,
    )
    # With noise, the Krylov subspace expands and Lanczos finds the true min.
    assert abs(lam - 0.5) < 1e-7
    # Overlap with the true smallest eigenvector should be ~1 (up to sign).
    assert abs(abs(u @ U[:, 0]) - 1.0) < 1e-6


# ---------- edge cases ----------

def test_seed_vanishes_after_projection_raises():
    eigvals = np.array([1.0, 2.0, 3.0])
    H, U = make_symmetric_with_spectrum(eigvals, seed=0)
    V = U[:, 0:1]
    v0_in_V = U[:, 0].copy()         # the seed lives entirely inside V_project
    with pytest.raises(ValueError, match="vanishes after projection"):
        lanczos_lowest(matvec_from_matrix(H), v0_in_V, V_project=V)


def test_v_project_shape_validation():
    H = np.eye(4)
    v0 = np.ones(4)
    bad_V = np.eye(3)               # wrong N
    with pytest.raises(ValueError, match="V_project must have shape"):
        lanczos_lowest(matvec_from_matrix(H), v0, V_project=bad_V)


def test_v_project_orthonormality_validation():
    # Non-orthonormal V (two parallel columns) → must be rejected.
    H = np.eye(5)
    v0 = np.ones(5)
    bad_V = np.array([
        [1.0, 1.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
    ])
    with pytest.raises(ValueError, match="orthonormal"):
        lanczos_lowest(matvec_from_matrix(H), v0, V_project=bad_V)


def test_v_project_non_unit_norm_rejected():
    # A single column with norm 2 is not orthonormal either.
    H = np.eye(5)
    v0 = np.ones(5)
    V = np.zeros((5, 1))
    V[0, 0] = 2.0                   # ‖V‖ = 2, not 1
    with pytest.raises(ValueError, match="orthonormal"):
        lanczos_lowest(matvec_from_matrix(H), v0, V_project=V)


# ============================================================
# Early break path: Krylov subspace exhaustion (β -> 0)
# ============================================================
#
# These tests verify that Lanczos terminates correctly when the Krylov
# subspace becomes invariant under A (β < tol). The β-becomes-zero path
# was an off-by-one bug magnet in the original implementation; the
# `betas[:k_dim - 1]` slicing must give the right shape regardless of
# whether the loop runs to completion or breaks early.

def test_early_break_when_seed_supported_on_3_eigenvectors():
    # A is diagonal with eigenvalues 1..10. Seed lies in the 3-dimensional
    # invariant subspace spanned by e_0, e_1, e_2. After 3 Lanczos iterations,
    # the Krylov subspace is exhausted and β at iter 3 must trigger the break.
    n = 10
    A = np.diag(np.arange(1.0, n + 1.0))
    v0 = np.zeros(n)
    v0[:3] = 1.0    # only the first 3 eigenvectors are reachable

    u, lam = lanczos_lowest( matvec_from_matrix(A), v0, max_iter=10, tol=1e-12 )    # max_iter way larger than 3 → must break early

    # Smallest eigvalue of A restricted to span{e_0, e_1, e_2} is 1.0.
    assert abs(lam - 1.0) < 1e-12
    # Eigenvector should be e_0 (up to sign).
    assert abs(abs(u[0]) - 1.0) < 1e-10
    np.testing.assert_allclose(u[3:], 0.0, atol=1e-10)


def test_early_break_at_iteration_5():
    # Same construction at a different k. Pushes the break further into the
    # loop and confirms the slicing is right regardless of where the break lands.
    n = 20
    A = np.diag(np.arange(1.0, n + 1.0))
    rng = np.random.default_rng(0)
    v0 = np.zeros(n)
    v0[:5] = rng.standard_normal(5)   # random coefficients across 5 eigvecs

    u, lam = lanczos_lowest(
        matvec_from_matrix(A), v0,
        max_iter=15, tol=1e-12,
    )

    # Smallest eigvalue in the 5-dim invariant subspace is 1.0.
    assert abs(lam - 1.0) < 1e-10
    # Returned eigenvector should be e_0 (up to sign).
    assert abs(abs(u[0]) - 1.0) < 1e-8
    np.testing.assert_allclose(u[5:], 0.0, atol=1e-10)


def test_early_break_produces_no_nans():
    # Explicit defence: when β -> 0 the break must fire, otherwise the next
    # iteration would divide r by 0 and inject NaNs into the returned vector.
    n = 8
    A = np.diag(np.arange(1.0, n + 1.0))
    v0 = np.zeros(n)
    v0[:2] = [1.0, 1.0]   # 2-dim invariant subspace, exhausts after 2 iters

    u, lam = lanczos_lowest( matvec_from_matrix(A), v0,  max_iter=20, tol=1e-12 )

    assert np.isfinite(lam)
    assert np.all(np.isfinite(u))
    assert abs(lam - 1.0) < 1e-12


def test_early_break_with_projection_combined():
    # Both an early break AND deflation simultaneously. A has eigvals 1..8,
    # we project out eigvec 0 (eigval 1), and v0 is supported only on eigvecs
    # 1, 2, 3. Krylov subspace is 3-dim (after projection it's still 3-dim
    # because projection eliminates e_0 but v0 had no e_0 component anyway).
    # Should return eigval 2 (second smallest).
    n = 8
    A = np.diag(np.arange(1.0, n + 1.0))
    e0 = np.zeros((n, 1)); e0[0, 0] = 1.0     # orthonormal (1 column, norm 1)

    v0 = np.zeros(n)
    v0[1:4] = [1.0, 1.0, 1.0]

    u, lam = lanczos_lowest( matvec_from_matrix(A), v0, max_iter=10, tol=1e-12, V_project=e0 )

    # A's smallest eigval in span{e_1, e_2, e_3} after projecting out e_0 is 2.0
    # (eigvec e_1).
    assert abs(lam - 2.0) < 1e-10
    assert abs(abs(u[1]) - 1.0) < 1e-8
    assert abs(u[0]) < 1e-10                  # deflated component


# ============================================================
# Large-N test cases with analytically known eigenvalues / eigenvectors
# ============================================================
#
# Tridiagonal Toeplitz matrix T_N(a, b) — diagonal a, off-diagonal b — has
# closed-form spectrum:
#     λ_k = a + 2b cos(k π / (N+1)),       k = 1..N
#     v_k[j] = sqrt(2/(N+1)) sin(j k π / (N+1)),  j = 1..N
# These tests verify Lanczos on this classical family at N=100, where eigvals
# and eigvecs aren't synthesised via random rotation but are mathematically
# pinned to the matrix structure.

def _toeplitz_tridiag(N: int, a: float, b: float) -> np.ndarray:
    H = np.zeros((N, N))
    np.fill_diagonal(H, a)
    for i in range(N - 1):
        H[i, i + 1] = b
        H[i + 1, i] = b
    return H


def _toeplitz_eigval(N: int, a: float, b: float, k: int) -> float:
    """The k-th eigenvalue (k = 1..N) of tridiagonal Toeplitz T_N(a, b)."""
    return a + 2.0 * b * np.cos(k * np.pi / (N + 1))


def _toeplitz_eigvec(N: int, k: int) -> np.ndarray:
    """The k-th eigenvector (k = 1..N), already unit-normalised."""
    j = np.arange(1, N + 1)
    return np.sqrt(2.0 / (N + 1)) * np.sin(j * k * np.pi / (N + 1))


def test_lanczos_100x100_discrete_laplacian():
    # Tridiagonal Toeplitz with a=2, b=-1 — the discrete 1D Laplacian.
    # Positive definite, smallest eigval λ_1 = 2 - 2 cos(π/101) ≈ 9.67e-4.
    # Eigenvalues at the bottom are densely clustered (gap ~3·(π/N)²), so
    # convergence to the smallest eigval is the slowest Lanczos regime. We
    # set max_iter > N which forces Krylov-subspace exhaustion (β → 0 at
    # iter N) and the eigval is then recovered exactly up to roundoff.
    N = 100
    a, b = 2.0, -1.0
    H = _toeplitz_tridiag(N, a, b)
    expected_min = _toeplitz_eigval(N, a, b, k=1)
    expected_eigvec = _toeplitz_eigvec(N, k=1)

    rng = np.random.default_rng(42)
    v0 = rng.standard_normal(N)

    u, lam = lanczos_lowest( matvec_from_matrix(H), v0, max_iter=110, tol=1e-12, seed_noise=0.0 )

    # With max_iter > N, the Krylov basis spans all of R^N (provided v0 has
    # nonzero overlap with every eigenvector, which a random Gaussian does
    # with probability 1). Eigenvalues recovered to machine precision.
    assert abs(lam - expected_min) < 1e-9, f"expected {expected_min:.6e}, got {lam:.6e}"
    overlap = abs(u @ expected_eigvec)
    assert overlap > 1.0 - 1e-6, f"eigvec overlap with true v_1 is only {overlap:.10f}"


def test_lanczos_100x100_indefinite_toeplitz():
    # Tridiagonal Toeplitz with a=0, b=1 — eigenvalues span [-2, 2]. Smallest
    # is the most negative (k=N), eigval = 2 cos(N π / (N+1)) = -2 cos(π/(N+1))
    # ≈ -1.9995. Eigvec is the high-frequency sine vector v_N. Tests Lanczos at
    # scale on an indefinite operator with the target at the negative extreme.
    # As in the Laplacian case the spectrum is clustered at both ends; we use
    # max_iter > N to recover the eigval exactly.
    N = 100
    a, b = 0.0, 1.0
    H = _toeplitz_tridiag(N, a, b)
    expected_min = _toeplitz_eigval(N, a, b, k=N)
    expected_eigvec = _toeplitz_eigvec(N, k=N)

    rng = np.random.default_rng(7)
    v0 = rng.standard_normal(N)

    u, lam = lanczos_lowest( matvec_from_matrix(H), v0, max_iter=110, tol=1e-12, seed_noise=0.0 )

    assert abs(lam - expected_min) < 1e-9, f"expected {expected_min:.6f}, got {lam:.6f}"
    overlap = abs(u @ expected_eigvec)
    assert overlap > 1.0 - 1e-6, f"eigvec overlap with true v_N is only {overlap:.10f}"


def test_lanczos_100x100_2d_discrete_laplacian():
    # 2D 5-point discrete Laplacian on a 10x10 grid → 100x100 matrix.
    # Sparse but *non-tridiagonal*: each interior row has nonzeros at columns
    # k-N, k-1, k, k+1, k+N (where N=10 is the row-stride). Exercises Lanczos
    # on a different sparsity / spectrum structure than the 1D Toeplitz cases.
    #
    # Eigenvalues are sums of 1D eigvals:
    #     λ_{j,k} = (2 - 2cos(j π / 11)) + (2 - 2cos(k π / 11))   j, k = 1..10
    # Eigenvectors are tensor products of 1D sines:
    #     v_{j,k}(i, ℓ) ∝ sin(i j π / 11) sin(ℓ k π / 11)
    # Smallest: λ_{1,1} = 2 · (2 - 2cos(π/11)) ≈ 0.163, eigvec is the outer
    # product of the two 1D ground-state sines (a single arch in each axis).
    M = N = 10
    n_dim = M * N

    # Build the matrix via the 5-point stencil.
    H = np.zeros((n_dim, n_dim))
    for i in range(M):
        for j in range(N):
            k = i * N + j
            H[k, k] = 4.0
            if j + 1 < N: H[k, k + 1] = -1.0       # right neighbour (off-diag ±1)
            if j > 0:     H[k, k - 1] = -1.0       # left neighbour
            if i + 1 < M: H[k, k + N] = -1.0       # down neighbour (off-diag ±N)
            if i > 0:     H[k, k - N] = -1.0       # up neighbour
    # Sanity-check the non-tridiagonal sparsity pattern is what we expect.
    assert H[0, N] == -1.0 and H[N, 0] == -1.0, "expected coupling at off-diag ±N"

    lam_1d = 2.0 - 2.0 * np.cos(np.pi / (M + 1))
    expected_min = 2.0 * lam_1d                    # (j, k) = (1, 1)
    # Smallest eigenvector: outer product of the two 1D ground-state sines.
    v_1d = np.sin(np.arange(1, M + 1) * np.pi / (M + 1))
    expected_eigvec = np.outer(v_1d, v_1d).reshape(-1)
    expected_eigvec /= np.linalg.norm(expected_eigvec)

    rng = np.random.default_rng(13)
    v0 = rng.standard_normal(n_dim)

    u, lam = lanczos_lowest(
        matvec_from_matrix(H), v0,
        max_iter=110, tol=1e-12, seed_noise=0.0,
    )

    assert abs(lam - expected_min) < 1e-9, f"expected {expected_min:.6e}, got {lam:.6e}"
    overlap = abs(u @ expected_eigvec)
    assert overlap > 1.0 - 1e-6, f"eigvec overlap with true v_(1,1) is only {overlap:.10f}"


def test_lanczos_100x100_with_deflation():
    # Combines the 100x100 case with V_project deflation. Discrete Laplacian,
    # deflate out the lowest TWO eigenvectors → Lanczos should return the
    # third-smallest eigval (k=3). max_iter > N forces exact convergence.
    N = 100
    a, b = 2.0, -1.0
    H = _toeplitz_tridiag(N, a, b)

    # First two eigenvectors as orthonormal V_project columns.
    # _toeplitz_eigvec returns unit-norm vectors; sine-eigvecs of distinct k
    # are exactly orthogonal, so V is orthonormal by construction.
    V = np.column_stack([_toeplitz_eigvec(N, k=1), _toeplitz_eigvec(N, k=2)])

    expected_min = _toeplitz_eigval(N, a, b, k=3)
    expected_eigvec = _toeplitz_eigvec(N, k=3)

    rng = np.random.default_rng(99)
    v0 = rng.standard_normal(N)

    u, lam = lanczos_lowest(
        matvec_from_matrix(H), v0,
        max_iter=110, tol=1e-12, V_project=V,
    )

    assert abs(lam - expected_min) < 1e-9, f"expected {expected_min:.6e}, got {lam:.6e}"
    overlap = abs(u @ expected_eigvec)
    assert overlap > 1.0 - 1e-6, f"eigvec overlap with true v_3 is only {overlap:.10f}"
    # Returned vector must be orthogonal to the deflated subspace.
    assert abs(u @ V[:, 0]) < 1e-8
    assert abs(u @ V[:, 1]) < 1e-8


def test_early_break_at_iteration_1_returns_eigenvector_as_is():
    # Pathological boundary: seed is exactly an eigenvector → β = 0 at the
    # very first iteration → break before any further Krylov expansion.
    # k_dim = 1, betas = [] (empty), tridiagonal is 1×1 = [[λ]].
    # This is the case where the early break path hits with the SMALLEST
    # possible k_dim, exercising the boundary of the slicing logic.
    A = np.diag([1.0, 3.0, 5.0])
    v0 = np.array([0.0, 1.0, 0.0])   # exactly the e_1 eigenvector → eigval 3.0

    u, lam = lanczos_lowest(
        matvec_from_matrix(A), v0,
        max_iter=10, tol=1e-12, seed_noise=0.0,
    )

    # Lanczos returns the eigenvalue of the seed (not the global smallest).
    assert abs(lam - 3.0) < 1e-12
    assert abs(abs(u[1]) - 1.0) < 1e-10


def test_converges_to_numpy_eigh_smallest():
    # End-to-end: build a moderately-sized matrix, run Lanczos with ample
    # iterations, compare to scipy.linalg.eigh's smallest eigenvalue.
    rng = np.random.default_rng(123)
    n = 80
    eigvals = np.sort(rng.uniform(0.1, 50.0, size=n))
    H, U = make_symmetric_with_spectrum(eigvals, seed=123)

    np_eigvals = np.linalg.eigvalsh(H)
    expected_min = float(np_eigvals[0])

    v0 = rng.standard_normal(n)
    u, lam = lanczos_lowest(matvec_from_matrix(H), v0, max_iter=40, tol=1e-14)

    # 40 Lanczos iterations on n=80 with a moderately-spread spectrum gives
    # ~1e-7 precision for the smallest eigenvalue.
    assert abs(lam - expected_min) < 1e-6
