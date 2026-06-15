"""
Lanczos iteration for the smallest eigenvalue (and Ritz vector) of a symmetric
linear operator given as a matvec callable.

The operator does not need to be stored explicitly. Optional projection lets
the caller deflate known null modes (e.g. trans/rot for molecular Hessians)
so Lanczos returns the smallest eigenvalue in the orthogonal complement of
a supplied subspace.

Full reorthogonalization is used — O(N²) per iteration but numerically
bullet-proof at the problem sizes this code is used for (few hundred
dimensions). For larger problems, switch to scipy.sparse.linalg.eigsh.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
from scipy.linalg import eigh_tridiagonal

def lanczos_lowest( matvec: Callable[[np.ndarray], np.ndarray],
                    v0: np.ndarray,
                    *,
                    max_iter: int = 15,
                    tol: float = 1e-10,
                    V_project: Optional[np.ndarray] = None,
                    seed_noise: float = 0.0,
                    random_state: int = 0,
    ) -> Tuple[np.ndarray, float]:
    """
    Smallest Ritz value / vector of a symmetric linear operator via Lanczos.

    Parameters
    ----------
    matvec : callable
        ``v -> A v``. Must be linear and symmetric. The matrix `A` itself
        never needs to exist explicitly.
    v0 : (N,) array_like
        Seed vector. Projected (if `V_project` given), perturbed (if
        `seed_noise > 0`), and normalised before the iteration starts.
    max_iter : int, default 15
        Maximum Krylov subspace dimension. 10-20 usually suffices for a
        well-separated lowest eigenvalue.
    tol : float, default 1e-10
        β tolerance for early termination — Lanczos has converged when the
        next basis vector has near-zero norm (the Krylov subspace is
        invariant under `A`).
    V_project : (N, k) array_like, optional
        Orthonormal columns spanning a subspace to project OUT of every
        Krylov vector. The effective operator becomes
        ``P A P  with  P = I - V Vᵀ``,
        and the returned smallest eigenvalue is the smallest of `A`'s
        eigenvalues in the orthogonal complement of `V_project`. Use this
        to deflate known null modes (e.g. trans/rot for molecular Hessians).
    seed_noise : float, default 0.0
        If positive, add ``seed_noise * standard_normal(N)`` to the seed
        before normalisation. Use this when the seed might coincide with a
        non-target eigenvector of `A` — without noise, Lanczos breaks down
        at step 1 (β=0) and returns the seed unchanged. A reasonable default
        for general-purpose use is ``1e-2``.
    random_state : int, default 0
        Seed for the noise perturbation (only consulted if `seed_noise > 0`).

    Returns
    -------
    u : (N,) ndarray
        Unit-norm Ritz vector for the smallest Ritz value, lying in the
        orthogonal complement of `V_project` if that was supplied.
    lam : float
        Smallest Ritz value.

    Raises
    ------
    ValueError
        If the seed vector vanishes after projection.

    Notes
    -----
    Returns *Ritz* values — approximations to the true eigenvalues of `A`
    restricted to the Krylov subspace built so far. Convergence is monotone
    in `max_iter` and quadratic near the limit for well-separated extreme
    eigenvalues. For matvecs that are expensive (e.g. finite-difference
    gradient probes), each Lanczos iteration costs one matvec call.
    """
    v0 = np.asarray(v0, dtype=np.float64).reshape(-1)
    dim = v0.size

    if V_project is not None:
        V_project = np.asarray( V_project, dtype=np.float64 )
        if V_project.ndim != 2 or V_project.shape[0] != dim:
            raise ValueError( f"V_project must have shape ({dim}, k); got {V_project.shape}." )
        # Orthonormality is essential for correctness: the "projector"
        # v - V V^T v is only a true orthogonal projector when V's columns
        # are orthonormal. A non-orthonormal V would silently shift the
        # returned eigenvalues. Cheap O(k²) check, ~µs at typical k.
        k_proj = V_project.shape[1]
        gram = V_project.T @ V_project
        if not np.allclose( gram, np.eye(k_proj), atol=1e-8 ):
            raise ValueError(
                "V_project columns must be orthonormal (V_project.T @ V_project != I). "
                "Pass an explicitly orthonormalised basis (e.g. via np.linalg.qr)."
            )

    def proj( v: np.ndarray ) -> np.ndarray:
        if V_project is None:
            return v
        return v - V_project @ (V_project.T @ v)

    # Initial vector for the Krylov iteration. Add random noise
    # to avoid direct collapse in case the initial vector is an exact
    # eigenvector A. This way, the initial vector has components along
    # all Krylov vectors.
    v = v0.copy()
    if seed_noise > 0.0:
        rng = np.random.default_rng( random_state )
        v = v + seed_noise * rng.standard_normal(dim)
    v = proj(v)
    nv = np.linalg.norm(v)
    if nv < 1e-12:
        raise ValueError( "Seed vector vanishes after projection; cannot start Lanczos." )
    v /= nv

    # Krylov vectors and Lanczos orthogonolization coeffiencts 
    # `alpha` (diagonal) and `beta` (off-diagonal).
    Q: list[np.ndarray] = [v]
    alphas: list[float] = []
    betas: list[float] = []

    # Run until we hit the zero vector or `max_iter`.
    for j in range(max_iter):
        Aq = proj( matvec(Q[j]) )
        alpha = float( Q[j] @ Aq )
        alphas.append( alpha )
        r = Aq - alpha * Q[j]
        if j > 0:
            r = r - betas[-1] * Q[j - 1]

        # Full reorthogonalisation against the running Krylov basis.
        # Cheap at the problem sizes here and prevents loss of orthogonality.
        for q in Q:
            r = r - float(r @ q) * q
        r = proj(r)
        beta = float( np.linalg.norm(r) )
        if beta < tol:
            break
        betas.append(beta)
        Q.append(r / beta)

    # Compute the eigenvalues of the tridiagonal Lanczos matrix.
    # Scipy has a dedicated routine that takes the diagonal and off-diagonal
    # matrix elements directly. Signals clearer intent than np.linalg.eigh
    k_dim = len(alphas)
    ritz_vals, ritz_vecs = eigh_tridiagonal( d=np.asarray(alphas), e=np.asarray(betas[:k_dim - 1]) )
    lam = float(ritz_vals[0])

    # When the loop completes max_iter iterations without breaking, Q ends up
    # with k_dim + 1 vectors (one extra appended at the end of the last iter
    # before we'd check β at the top of iteration k_dim). Use only the first
    # k_dim vectors, matching the size of the Lanczos tridiagonal (k_dim × k_dim).
    Q_mat = np.stack( Q[:k_dim], axis=1 )
    u = Q_mat @ ritz_vecs[:, 0]
    u = proj(u)
    u /= np.linalg.norm(u)
    return u, lam
