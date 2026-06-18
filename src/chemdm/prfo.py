"""
Partitioned Rational Function Optimization (P-RFO) for single-ended TS search.

Cartesian coordinates with translation/rotation projection. Quasi-Newton Hessian
updated via Bofill — only forces are required, no analytic Hessian.

References:
  Banerjee, Adams, Simons, Shepard, J. Phys. Chem. 89, 52 (1985).
  Bofill, J. Comput. Chem. 15, 1 (1994).
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, Tuple

import numpy as np
import torch as pt

from chemdm.MoleculeGraph import Molecule
from chemdm.MoleculeInformation import _ATOMIC_MASS_TABLE
from chemdm.Constants import _HARTREE_TO_EV, _HARTREE_PER_BOHR2_TO_EV_PER_ANG2
from chemdm.lanczos import lanczos_lowest


def _mass_inv_sqrt_dof( Z: np.ndarray ) -> np.ndarray:
    """
    Per-DOF reciprocal square-root mass vector for a molecule, shape `(3N,)`.

    Used to switch between Cartesian and mass-weighted coordinates:
        y = M^{1/2} x,  H_y = M^{-1/2} H_x M^{-1/2}.
    With `m_inv_sqrt = 1 / sqrt(m)` replicated 3× per atom, elementwise
    multiplication `m_inv_sqrt * v` realises both `M^{-1/2} v` (when `v` is
    a Cartesian quantity becoming mass-weighted) and the conjugate Hessian
    sandwich `M^{-1/2} (H_x (M^{-1/2} v))`.

    Masses in amu (from the standard atomic-weights table in
    `chemdm.MoleculeInformation`). Raises if any `Z` lacks a tabulated mass.
    """
    Z_np = np.asarray(Z, dtype=int).reshape(-1)
    m = _ATOMIC_MASS_TABLE.numpy().astype(np.float64)[Z_np]
    if (m <= 0.0).any():
        bad = np.unique(Z_np[m <= 0.0])
        raise ValueError(
            f"_ATOMIC_MASS_TABLE has no positive entry for Z={bad.tolist()}; "
            f"mass-weighting requires every atom to have a tabulated mass."
        )
    return np.repeat( 1.0 / np.sqrt(m), 3 )


# Evaluator interface
class EnergyForceEvaluator(Protocol):
    # `x` is declared positional-only (the `/`) so that implementers may name
    # the parameter whatever they like (e.g. XTBPotential uses `x_A`).
    def energy_forces(self, x: np.ndarray, /) -> Tuple[float, np.ndarray]:
        ...


# ============================================================
# Helpers
# ============================================================

def _to_numpy( t: Any ) -> np.ndarray:
    """torch.Tensor or array-like -> numpy.ndarray, on CPU."""
    if hasattr( t, "detach" ):
        return t.detach().cpu().numpy()
    return np.asarray(t)


# ============================================================
# Internals
# ============================================================

def _trans_rot_basis( x: np.ndarray ) -> np.ndarray:
    """
    Orthonormal basis (3N, k) spanning rigid-body translations and infinitesimal
    rotations about the current centroid. k = 6 for a generic molecule, 5 for a
    linear one (QR drops the redundant axis).

    x: (n_atoms, 3).
    """
    n = x.shape[0]
    r = x - x.mean(axis=0)

    # Translational basis vectors. Shift the whole molecule
    cols = []
    for i in range(3):
        v = np.zeros((n, 3))
        v[:, i] = 1.0
        cols.append(v.reshape(-1))

    # Infinitesimal rotational basis vectors = unit vector orthogonal to rotation plane.
    for axis in np.eye(3):
        cols.append( np.cross(axis, r).reshape(-1) )

    # Make all vectors orthogonal
    V = np.stack(cols, axis=1)
    Q, R = np.linalg.qr(V)
    diag = np.abs(np.diag(R))
    if diag.max() == 0.0:
        return np.zeros((V.shape[0], 0))
    keep = diag > 1e-10 * diag.max()
    return Q[:, keep]


def _project_vec( v: np.ndarray, V: np.ndarray ) -> np.ndarray:
    """ Project a vector v away from V as (I - V V^T) v"""
    return v - V @ (V.T @ v)


def _project_mat( M: np.ndarray, V: np.ndarray ) -> np.ndarray:
    """ Project all columns of matrix M away from V."""
    M = M - V @ (V.T @ M)
    M = M - (M @ V) @ V.T
    return M


def _bofill_update( H: np.ndarray, # (3N, 3N)
                    dx: np.ndarray, # (3N,)
                    dg: np.ndarray, # (3N,)
                    eps: float = 1e-10 ) -> np.ndarray:
    """
    Bofill update: convex mix of SR1 (preserves indefiniteness, good for saddles)
    and PSB (stable when SR1's denominator is small). Mixing weight
        phi = (dx.r)^2 / (||dx||^2 ||r||^2)
    is the cos^2 between dx and the residual r = \Delta g - H \Delta x.
    """
    r = dg - H @ dx
    dxTdx = float(dx @ dx)
    rTr = float(r @ r)
    if dxTdx < eps or rTr < eps:
        return H

    # PSB update
    dxTr = float( dx @ r )
    psb = (np.outer(r, dx) + np.outer(dx, r)) / dxTdx - dxTr * np.outer(dx, dx) / (dxTdx * dxTdx)

    if abs(dxTr) < eps * np.sqrt(rTr * dxTdx):
        return H + psb

    # Mixing weight phi
    sr1 = np.outer(r, r) / dxTr
    phi = dxTr**2 / (rTr * dxTdx)
    return H + phi * sr1 + (1.0 - phi) * psb


def _prfo_step( H: np.ndarray, # (3N,3N)
                g: np.ndarray, # (3N,)
                follow_vec: Optional[np.ndarray], # (3N,)
                zero_tol: float = 1e-6,
               ) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Solve partitioned RFO equations on the projected Hessian/gradient.

    Returns (step_dx, followed_eigvec, followed_eigval).
    """

    # TODO: We could hot-start the eigenvalue computation for 
    # faster computations. This is not a bottleneck because eigenvalue
    # calculations are drowned by forcefield evaluations. 
    # If we ever apply this to proteins, hot-starting will be necessary.
    eigvals, eigvecs = np.linalg.eigh( H )

    # In Cartesian coordinates, 6 (or 5) eigenvalues will be zero.
    # We need to filter these out before subsequent calculations.
    scale = np.abs(eigvals).max() + 1.0
    nonzero = np.abs(eigvals) > zero_tol * scale
    eigvals = eigvals[nonzero]
    eigvecs = eigvecs[:, nonzero]

    # Projection of the gradient on the eigenbasis.
    f = eigvecs.T @ g

    if follow_vec is None:
        k = int( np.argmin(eigvals) )
    else:
        # find the new eigenvector closest to the old (smallest angle)
        overlaps = np.abs( eigvecs.T @ follow_vec )
        k = int( np.argmax(overlaps) )

    lam_k = float( eigvals[k] )
    f_k = float( f[k] )

    # 2x2 RFO, larger root: ascent direction.
    # Analytic solution is known.
    mu_plus = 0.5 * (lam_k + np.sqrt(lam_k * lam_k + 4.0 * f_k * f_k))
    denom_k = lam_k - mu_plus
    dq_k = -f_k / denom_k if abs( denom_k ) > 1e-14 else 0.0

    # (M+1)x(M+1) RFO on the minimization subspace, lowest root.
    other = np.ones( eigvals.size, dtype=bool )
    other[k] = False
    lam_min = eigvals[other]
    f_min = f[other]
    M = lam_min.size

    if M > 0:
        A = np.zeros( (M + 1, M + 1) )
        idx = np.arange(M)
        A[idx, idx] = lam_min
        A[:M, M] = f_min
        A[M, :M] = f_min
        mu_minus = float( np.linalg.eigvalsh(A)[0] )

        denom = lam_min - mu_minus
        safe = np.where( np.abs(denom) > 1e-14, denom, 1e-14 )
        dq_min = -f_min / safe
    else:
        dq_min = np.zeros( 0 )

    dq = np.empty( eigvals.size )
    dq[k] = dq_k
    dq[other] = dq_min

    dx = eigvecs @ dq
    return dx, eigvecs[:, k].copy(), lam_k


# ============================================================
# Lindh model Hessian
# ============================================================
#
# Lindh, Chem. Phys. Lett. 241, 423 (1995). Stretches + bends; torsions
# still omitted (the chemical effect they'd represent — backbone φ/ψ
# softness — currently shows up via the Lindh-floor null space, which is
# where mass-weighted Lanczos picks them up). Bend contributions use the
# Wilson B-matrix `g_θ = ∂θ/∂x_cart`; near-collinear angles are skipped
# rather than handled with auxiliary linear-bend coordinates.

_LINDH_ALPHA = {  # 1/Å²; indexed by (row_i, row_j)
    (1, 1): 1.0000,
    (1, 2): 0.3949, (2, 1): 0.3949,
    (2, 2): 0.2800,
    (1, 3): 0.3949, (3, 1): 0.3949,
    (2, 3): 0.2800, (3, 2): 0.2800,
    (3, 3): 0.2800,
}

_LINDH_RREF = {  # Å
    (1, 1): 1.35,
    (1, 2): 2.10, (2, 1): 2.10,
    (2, 2): 2.53,
    (1, 3): 2.53, (3, 1): 2.53,
    (2, 3): 2.53, (3, 2): 2.53,
    (3, 3): 2.53,
}

_LINDH_K_STRETCH = 0.45 * _HARTREE_PER_BOHR2_TO_EV_PER_ANG2  # ≈ 43.7 eV/Å²
_LINDH_K_BEND    = 0.15 * _HARTREE_TO_EV                     # ≈  4.08 eV/rad²


def _atom_row( Z: int ) -> int:
    if Z == 1:
        return 1
    if 2 <= Z <= 10:
        return 2
    if 11 <= Z <= 18:
        return 3
    if 19 <= Z <= 36:
        return 3  # 4th row borrows from 3rd; extend later if needed
    raise NotImplementedError( f"Lindh row not defined for Z={Z}" )


def _unique_bonds( edge_index: np.ndarray ) -> np.ndarray:
    """Collapse directed edge_index (n_edges, 2) to unordered (i<j)
    pairs without duplicates. Returns shape (n_unique, 2)."""
    if edge_index.size == 0:
        return np.zeros((0, 2), dtype=int)
    pairs = np.sort( edge_index, axis=1 )
    return np.unique( pairs, axis=0 )


def _bend_rank1_gradient( x: np.ndarray, i: int, j: int, k: int ) -> Optional[np.ndarray]:
    """
    Wilson B-matrix row `g = ∂θ_ijk / ∂x_cart` as a (3N,) sparse-but-dense
    vector with nonzero entries only at atoms i, j, k. Returns `None` when
    the angle is collinear (within numerical tolerance) — Lindh's full model
    handles linear bends with auxiliary coordinates; we just skip them here
    since (a) genuinely collinear i-j-k triples are rare and (b) near-linear
    angles contribute negligibly to the lowest-mode subspace anyway.
    """
    rij = x[i] - x[j]; nij = float( np.linalg.norm(rij) )
    rkj = x[k] - x[j]; nkj = float( np.linalg.norm(rkj) )
    if nij < 1e-8 or nkj < 1e-8:
        return None
    u = rij / nij
    v = rkj / nkj
    cos_t = float( np.clip(u @ v, -1.0, 1.0) )
    sin_t = float( np.sqrt(max(1.0 - cos_t * cos_t, 0.0)) )
    if sin_t < 1e-4:
        return None  # near-collinear — skip
    gi = (cos_t * u - v) / (nij * sin_t)
    gk = (cos_t * v - u) / (nkj * sin_t)
    gj = -(gi + gk)
    g = np.zeros( 3 * x.shape[0] )
    g[3*i : 3*i + 3] = gi
    g[3*j : 3*j + 3] = gj
    g[3*k : 3*k + 3] = gk
    return g


def lindh_model_hessian( molecule: Any ) -> np.ndarray:
    """
    Lindh 1995 pairwise stretch + bend model Hessian (torsions still omitted).

    Two rank-1 contributions per internal coordinate:

    * **Stretches** — for each unique bond (i, j) with current length r_ij
      and the i-j unit vector d, ``H += k_r · ρ_ij · (d_i − d_j)(d_i − d_j)ᵀ``
      with ``k_r = 0.45 Hartree/Bohr² ≈ 43.7 eV/Å²`` and Lindh's row-dependent
      distance damping ``ρ_ij = exp(α (r_ref² − r²))``.
    * **Bends** — for each triple (i, j, k) with j bonded to both i and k,
      ``H += k_θ · ρ_ij · ρ_jk · g_θ g_θᵀ`` with ``k_θ = 0.15 Hartree/rad²
      ≈ 4.08 eV/rad²`` and ``g_θ`` the Wilson B-matrix row for the angle
      i-j-k. Near-collinear triples (sin θ ≲ 1e-4) are skipped — their
      Wilson B-matrix is singular and Lindh's full model would handle them
      with auxiliary linear-bend coordinates we don't bother to set up.

    Parameters
    ----------
    molecule : chemdm.MoleculeGraph.Molecule
        Provides `x` (positions, Å), `Z` (atomic numbers) and `edge_index`
        (bond list, directed or undirected — duplicates are filtered).

    Returns
    -------
    H : (3N, 3N) ndarray, eV/Å²
        Symmetric, PSD up to numerical roundoff. Has 6 zero modes (trans/rot)
        for non-linear molecules, 5 for linear ones. Torsions remain in the
        null space — adding torsion terms is the obvious next extension.
    """
    x = _to_numpy(molecule.x).astype(float)
    Z = _to_numpy(molecule.Z).astype(int).flatten()
    edges = _to_numpy(molecule.edge_index).astype(int) # (E,2)

    n = x.shape[0]
    H = np.zeros((3 * n, 3 * n))

    bonds = _unique_bonds( edges )

    # Cache per-bond ρ and adjacency for the bend loop below.
    rho_bond: dict[tuple[int, int], float] = {}
    adj: list[set[int]] = [set() for _ in range(n)]

    for i, j in bonds:
        v = x[j] - x[i]
        r = float( np.linalg.norm(v) )
        if r < 1e-8:
            continue
        d = v / r

        ri, rj = _atom_row(int(Z[i])), _atom_row(int(Z[j]))
        alpha = _LINDH_ALPHA[ (ri, rj) ]
        r_ref = _LINDH_RREF[ (ri, rj) ]
        rho = float( np.exp(alpha * (r_ref ** 2 - r ** 2)) )
        rho_bond[(int(i), int(j))] = rho
        rho_bond[(int(j), int(i))] = rho
        adj[int(i)].add(int(j))
        adj[int(j)].add(int(i))

        k = _LINDH_K_STRETCH * rho
        dd = np.outer(d, d)
        ii = slice(3 * i, 3 * i + 3)
        jj = slice(3 * j, 3 * j + 3)
        H[ii, ii] += k * dd
        H[jj, jj] += k * dd
        H[ii, jj] -= k * dd
        H[jj, ii] -= k * dd

    # Bend contributions: every (i, j, k) with j as the central atom and
    # i < k drawn from j's bonded neighbours. Each triple appears once.
    for j in range(n):
        nbrs = sorted(adj[j])
        for a in range(len(nbrs)):
            for b in range(a + 1, len(nbrs)):
                i, k = nbrs[a], nbrs[b]
                g = _bend_rank1_gradient(x, i, j, k)
                if g is None:
                    continue
                k_theta = _LINDH_K_BEND * rho_bond[(i, j)] * rho_bond[(j, k)]
                H += k_theta * np.outer(g, g)

    return H


# ============================================================
# Lowest-mode estimation: Lindh seed + dimer rotation
# ============================================================

def _mass_weighted_lowest_mode_of( H_cart: np.ndarray,
                                   x: np.ndarray,
                                   Z: np.ndarray ) -> np.ndarray:
    """
    Cartesian unit vector along the lowest mass-weighted eigenmode of an
    already-built Cartesian Hessian, with translations and rotations
    projected out.

    Shared kernel for :func:`lindh_lowest_mode` and the iter-0 seed inside
    :class:`PRFOOptimizer`. Both callers pass a positive-definite Cartesian
    Hessian (Lindh + isotropic floor) that's already been assembled — this
    helper just does the basis change and eigendecomposition.

    Caller is responsible for ensuring `H_cart` is positive-definite in the
    non-trans/rot subspace; for Lindh, that means adding a small isotropic
    floor before calling so bend/torsion modes lift above the eigenvalue
    filter threshold.
    """
    m_inv_sqrt = _mass_inv_sqrt_dof(Z)
    m_sqrt = 1.0 / m_inv_sqrt

    H_mw = (m_inv_sqrt[:, None] * H_cart) * m_inv_sqrt[None, :]

    V_cart = _trans_rot_basis(x)
    V_proj, _ = np.linalg.qr( m_sqrt[:, None] * V_cart )

    H_proj = _project_mat( H_mw, V_proj )
    eigvals, eigvecs = np.linalg.eigh( H_proj )
    # Filter trans/rot residuals (now exactly zero after projection).
    scale = max( abs(eigvals).max(), 1e-6 )
    nonzero = np.abs(eigvals) > 1e-6 * scale
    k = int( np.argmin(eigvals[nonzero]) )
    v_mw = eigvecs[:, nonzero][:, k]

    u_cart = m_inv_sqrt * v_mw
    u_cart -= V_cart @ (V_cart.T @ u_cart)
    nu = float( np.linalg.norm(u_cart) )
    if nu < 1e-12:
        raise RuntimeError(
            "Mass-weighted lowest mode vanishes after Cartesian trans/rot "
            "projection — should not happen for a well-conditioned Hessian."
        )
    u_cart /= nu
    return u_cart


def lindh_lowest_mode( molecule: Molecule, *, floor: float = 0.01 ) -> np.ndarray:
    """
    Lowest mass-weighted physical mode of the Lindh model Hessian at the
    geometry stored in `molecule.x`. Returns a Cartesian unit vector lying in
    the trans/rot-free subspace.

    Unlike :func:`estimate_lowest_mode`, this does NOT touch the evaluator —
    Lindh is purely a function of bond connectivity and atomic distances, so
    finding its softest mode is a single eigendecomposition with zero force
    calls. The trade-off is that Lindh only knows about stretches; bend and
    torsion modes live in its null space and are differentiated only by
    mass-weighting and the small isotropic floor.

    Used inside :class:`PRFOOptimizer` to seed the mode-following reference
    `_follow_vec` at iteration 0 (so `_prfo_step.argmin` doesn't tie-break
    arbitrarily among the soft modes), and exposed publicly for demo scripts
    that need a kick direction along the slowest physical coordinate.

    Parameters
    ----------
    floor : float, default 0.01
        Isotropic floor (eV/Å²) added to Lindh before mass-weighting. Lifts
        the bend/torsion null space above zero so `eigh`'s eigenvalue filter
        can distinguish physical bends from trans/rot null modes. Same value
        as the floor used inside :class:`PRFOOptimizer`'s initial Hessian.

    Returns
    -------
    u : (3N,) unit vector
        Cartesian unit vector, trans/rot-projected.
    """
    x = _to_numpy( molecule.x ).astype(float)
    dim = 3 * x.shape[0]
    H = lindh_model_hessian( molecule ) + floor * np.eye(dim)
    return _mass_weighted_lowest_mode_of( H, x, _to_numpy(molecule.Z) )


def estimate_lowest_mode( evaluator: EnergyForceEvaluator,
                          molecule: Molecule, *,
                          init_u: Optional[np.ndarray] = None,
                          max_iter: int = 15,
                          eps: float = 1e-3,
                          tol: float = 1e-8,
                          seed_noise: float = 1e-2,
                          random_state: int = 0,
                         ) -> Tuple[np.ndarray, float]:
    """
    Estimate the lowest mass-weighted normal mode at `molecule.x` by Lanczos
    iteration with finite-difference Hessian-vector products.

    The iteration runs on the mass-weighted Hessian `M^{-1/2} H M^{-1/2}`, so
    the lowest eigenvalue is the squared frequency of the slowest *physical*
    normal mode rather than the lowest-Cartesian-curvature direction. On
    biomolecular systems this picks up heavy-atom collective motions (e.g.
    backbone torsions) instead of light hydrogen wags that happen to have
    comparable Cartesian curvature but negligible mass.

    The returned vector is back-converted to Cartesian and unit-normalised;
    `lam` is the Cartesian Rayleigh quotient `uᵀ H u` (eV/Å²) along that
    direction, so downstream callers keep Cartesian semantics. Trans/rot
    modes are always projected out — the mass-weighted basis is built as
    `QR(M^{1/2} V_cart)` so Lanczos's orthonormality check passes.

    Seeds the Lanczos iteration from the lowest eigenmode of the (mass-
    weighted) Lindh model Hessian, or from `init_u` if provided (Cartesian
    input, automatically converted). Each Lanczos step costs one extra
    force call (forward-difference `Hv ≈ (g(x+εv) - g₀)/ε`).

    Returns
    -------
    u : (3N,) unit vector
        Lowest-mode estimate in Cartesian coordinates, lying in the trans/rot-
        free subspace.
    lam : float
        Cartesian Rayleigh quotient `uᵀ H u` along `u`, in eV/Å². NOT in
        general the smallest Cartesian eigenvalue — it can be much larger
        than what a pure-Cartesian Lanczos would report, which is precisely
        the point of mass weighting.
    """
    x = _to_numpy( molecule.x ).astype(float)
    n_atoms = x.shape[0]
    dim = 3 * n_atoms

    V_cart = _trans_rot_basis(x)

    _, F0 = evaluator.energy_forces(x)
    g0 = -np.asarray(F0, dtype=float).reshape(-1)

    def Hv_fd_cart( v_cart: np.ndarray ) -> np.ndarray:
        x_plus = x + eps * v_cart.reshape(n_atoms, 3)
        _, F_plus = evaluator.energy_forces(x_plus)
        g_plus = -np.asarray(F_plus, dtype=float).reshape(-1)
        return (g_plus - g0) / eps

    m_inv_sqrt = _mass_inv_sqrt_dof( _to_numpy(molecule.Z) )       # (3N,)
    m_sqrt = 1.0 / m_inv_sqrt
    # Mass-weighted trans/rot basis: y-space null modes are M^{1/2} times the
    # Cartesian ones. M^{1/2} V_cart is no longer orthonormal (lanczos_lowest
    # validates V_project orthonormality), so QR gives an orthonormal basis
    # of the same span.
    V_proj, _ = np.linalg.qr( m_sqrt[:, None] * V_cart )

    def matvec( v: np.ndarray ) -> np.ndarray:
        return m_inv_sqrt * Hv_fd_cart( m_inv_sqrt * v )

    if init_u is None:
        H_model = lindh_model_hessian( molecule )
        H_model_mw = (m_inv_sqrt[:, None] * H_model) * m_inv_sqrt[None, :]
        H_act = _project_mat( H_model_mw, V_proj )
        eigvals, eigvecs = np.linalg.eigh( H_act )
        scale = max(abs(eigvals).max(), 1e-6)
        nonzero = np.abs(eigvals) > 1e-6 * scale
        if nonzero.any():
            u0 = eigvecs[:, nonzero][:, int(np.argmin(eigvals[nonzero]))]
        else:
            u0 = np.random.default_rng(0).standard_normal(dim)
    else:
        # User-supplied initial ascent direction is given in Cartesian.
        u0 = m_sqrt * np.asarray( init_u, dtype=float ).reshape(-1)

    u, _ = lanczos_lowest(
        matvec, u0,
        max_iter=max_iter,
        tol=tol,
        V_project=V_proj,
        seed_noise=seed_noise,
        random_state=random_state,
    )

    u = m_inv_sqrt * u
    # Mop up any residual Cartesian trans/rot leakage from the back-conversion
    # (mass-weighted trans/rot null space maps to Cartesian trans/rot, but only
    # up to QR's numerical precision).
    u = u - V_cart @ (V_cart.T @ u)
    nu = float(np.linalg.norm(u))
    if nu < 1e-12:
        raise RuntimeError(
            "Mass-weighted Lanczos returned a vector that vanishes after "
            "Cartesian trans/rot projection — should not happen for a "
            "well-conditioned molecular Hessian."
        )
    u /= nu
    lam = float( u @ Hv_fd_cart(u) )
    return u, lam


# ============================================================
# Optimizer
# ============================================================
class PRFOOptimizer:
    """
    P-RFO single-ended transition-state search on a molecular geometry.

    Always operates in Cartesians with translation/rotation projection.
    Quasi-Newton Hessian, updated via Bofill — only forces are required,
    no analytic Hessian.

    Parameters
    ----------
    evaluator
        Object with `energy_forces(x) -> (energy, forces)`. `x` is a numpy array
        of shape `(n_atoms, 3)`; `forces` matches and is the negative gradient.
        :class:`chemdm.xtbSetup.XTBPotential` satisfies this directly.
    molecule : chemdm.MoleculeGraph.Molecule
        Initial geometry as a `Molecule`. Positions are read from `molecule.x`;
        the molecule is stashed as `self.molecule` for later use (e.g.\\
        `lindh_model_hessian`, the auto-seed path).
    init_mode : {"lindh", None}, default "lindh"
        How to seed the mode-following reference `_follow_vec` at iteration 0.

        - "lindh" (default): call :func:`lindh_lowest_mode` internally —
          the lowest mass-weighted eigvec of the Lindh model Hessian plus an
          isotropic floor, all at zero force-call cost. Without the seed,
          `_prfo_step.argmin` would tie-break arbitrarily among the bend/
          torsion modes all sitting at the floor value (= non-deterministic
          mode choice on iteration 0).
        - None: skip the seed. Mode-following starts from whatever
          `_prfo_step.argmin` picks; cheap but the choice is undefined when
          multiple modes share the lowest eigvalue. Use only with mock
          evaluators where determinism isn't critical.

        The *base* initial Hessian is always the stretches-only Lindh model
        plus the same isotropic floor (see `__init__` body). The earlier
        identity-Hessian baseline was retired because it caused Bofill to
        spend the first ~3 iterations washing out wildly-wrong (+1 vs true
        ~+50) curvature priors on stretches. The artificial rank-1 negative-
        curvature overlay of even-earlier versions has also been retired —
        with Lindh+floor as the base, P-RFO's partitioned ascent along the
        seeded soft direction naturally walks the geometry toward the
        inflection point where curvature flips negative on its own, and
        Bofill takes over.
    trust_radius, max_trust, min_trust : float
        Trust-region bounds on the Cartesian step (Angstrom).
    relanczos_every : int or None, default None
        If a positive integer `N`, re-anchor `self._follow_vec` against the
        true lowest mode at the current geometry every `N` iterations (via
        :func:`estimate_lowest_mode`, which Lanczos-iterates against the
        finite-difference Hessian). The Bofill-updated `H` is **left
        untouched** — only the mode-following reference is refreshed.

        What this does NOT cure: smooth rotation of the followed eigenvector
        through a near-degenerate eigenspace (e.g.\\ HCN's bend at the
        linear minimum). Bofill rank-2 updates inside a 2-fold degenerate
        subspace can rotate the eigenbasis by ~45° per step with no energy
        cost, and the followed eigvec can drift through 90° in a few
        iterations even though `overlap_with_prev > 0.8` at every step.
        Refreshing every 10 iterations is too coarse to intervene; by the
        time `N=10` is reached, the rotation has often already completed.

        Effective cadences in practice:
            - `None` or `N=10` (default-ish): essentially inert. Small
              initial trust radius is doing the real stability work.
            - `N ≤ 5`: catches rotation as it happens, makes a noticeable
              difference on hard cases.
            - `N=1` (Lanczos every step): equivalent to ART-nouveau style.
              ~10x more force calls, but eliminates mode-following entirely
              by re-deriving the unstable direction from the true Hessian
              every iteration. The proper fix for systems with degenerate
              softest modes.

        Cost per refresh ≈ 10 force calls (1 reference gradient + 1 per
        Lanczos iteration, default `max_iter=15`).

    Notes
    -----
    Mode discovery is mass-weighted throughout — both the iteration-0 seed
    (via :func:`lindh_lowest_mode`) and any `relanczos_every` refresh (via
    :func:`estimate_lowest_mode`) — so the followed direction is the slowest
    *physical* normal mode rather than the lowest-Cartesian-curvature
    direction. The rest of P-RFO still operates in Cartesian coordinates:
    trust radius, gradient tolerance, and Hessian update semantics are
    unchanged.
    """
    _INIT_MODE_CHOICES = ( "lindh", )

    def __init__(self, evaluator: EnergyForceEvaluator,
                 molecule: Molecule, *,
                 init_mode: Optional[str] = "lindh",
                 trust_radius: float = 0.3,
                 max_trust: float = 0.5,
                 min_trust: float = 0.01,
                 relanczos_every: Optional[int] = None):
        if not isinstance(molecule, Molecule):
            raise TypeError( f"PRFOOptimizer requires a chemdm.MoleculeGraph.Molecule; got {type(molecule).__name__}." )
        if init_mode is not None and init_mode not in self._INIT_MODE_CHOICES:
            raise ValueError( f"init_mode must be None or one of {self._INIT_MODE_CHOICES}; got {init_mode!r}." )

        self.molecule = molecule
        x0 = _to_numpy(molecule.x).astype(float)
        self._shape = x0.shape
        self.evaluator = evaluator
        self.x = x0.flatten().copy()
        self.dim = self.x.size

        # Initial Hessian: stretches-only Lindh model plus a tiny isotropic
        # floor. Lindh gives every covalent-bond direction a physically
        # reasonable positive curvature (~30-70 eV/Å² for typical bonds), so
        # Bofill doesn't have to wash out a wildly-wrong identity prior (+1
        # along every stretch) before it can build a coherent indefinite
        # Hessian. The isotropic floor is there because Lindh assigns
        # *exactly zero* curvature to bend/torsion directions (they lie in
        # the null space of every stretch term); without the floor those
        # modes would be discarded by the `|eigval| > 1e-6 * scale` filter
        # in `_prfo_step`, since `scale` jumps from O(1) to O(50) once Lindh
        # is the base. 0.01 eV/Å² is well below true bend curvatures (~0.04
        # for HCN, ~0.1-0.3 for typical biomolecular torsions) so Bofill
        # refines it correctly on the first few gradient differences.
        _LINDH_FLOOR = 0.01    # eV/Å² — keep-alive seed on Lindh's null space
        self.H = lindh_model_hessian( molecule )
        self.H = self.H + _LINDH_FLOOR * np.eye( self.dim ) # regularization

        self._init_follow_vec: Optional[np.ndarray] = None
        self._init_follow_seed: Optional[np.ndarray] = None
        if init_mode == "lindh":
            # Lindh's lowest mass-weighted eigvec — pure compute, zero force
            # calls (cf. the earlier `estimate_lowest_mode` path which spent
            # ~15 force calls on Lanczos refinement). The Lindh model is what
            # told us the soft directions in the first place; iterating
            # against the true Hessian was redundant once we stopped using
            # the result to inject artificial negative curvature.
            #
            # `self.H` is already Lindh + floor — reuse it instead of going
            # through `lindh_lowest_mode` (which would rebuild Lindh from
            # the molecule's bond list).
            u = _mass_weighted_lowest_mode_of( self.H, _to_numpy(molecule.x).astype(float), _to_numpy(molecule.Z) )
            self._init_follow_vec = u.copy()
            self._init_follow_seed = u.copy()

        # Outer 'safety' trust-region parameters.
        # Enhances numerical stability a lot.
        self.trust = float( trust_radius )
        self.max_trust = float( max_trust )
        self.min_trust = float( min_trust )

        # Periodic Lanczos-refresh cadence (None = disabled).
        if relanczos_every is not None and relanczos_every <= 0:
            raise ValueError( f"relanczos_every must be a positive int or None; got {relanczos_every}." )
        self.relanczos_every = relanczos_every

        # `_follow_vec` is seeded from the Lanczos result (if available) so
        # iteration 0's mode-following picks the slowest physical direction
        # rather than tie-breaking arbitrarily among the bend/torsion modes
        # all sitting at the Lindh floor value.
        self._follow_vec: Optional[np.ndarray] = self._init_follow_seed.copy() if self._init_follow_seed is not None else None
        self._prev_g: Optional[np.ndarray] = None
        self._prev_dx: Optional[np.ndarray] = None
        self._prev_pred_dE: Optional[float] = None
        self._prev_E: Optional[float] = None

        self.history: list[dict] = []

    def _evaluate( self ) -> Tuple[float, np.ndarray]:
        E, F = self.evaluator.energy_forces( self.x.reshape(self._shape) )
        g = -np.asarray( F, dtype=float ).reshape(-1)
        return float(E), g

    def _refresh_unstable_mode(self) -> None:
        """
        Re-anchor the unstable mode at the current geometry.

        Runs Lanczos against the true Hessian (via finite-difference gradient
        probes) to get a fresh lowest-mode estimate `u_fresh`, then sets
        `self._follow_vec = u_fresh` so the next step's mode-following picks
        the eigvec of H closest to u_fresh rather than to the previous step's
        followed vec. The Bofill-updated H is **left untouched** — its
        accumulated curvature estimates persist, only the mode-following
        reference is refreshed.

        Why not also project H + re-inject negative curvature? An earlier
        version did exactly that. The problem: at a minimum's basin, the
        *true* curvature along the soft mode is positive (~+9 eV/Å² for
        HCN's bend). Bofill eventually learns this. Injecting an arbitrary
        -1 along u_fresh overwrites that learned curvature with noise,
        causing P-RFO to converge to spurious "stationary points" of the
        artificially modified Hessian model. The conservative refresh
        below avoids that failure mode.
        """
        x_torch = pt.tensor( self.x.reshape(self._shape), dtype=pt.float64 )
        mol_now = self.molecule.copyWithNewPositions( x_torch )
        u_fresh, _ = estimate_lowest_mode( self.evaluator, mol_now )
        self._follow_vec = u_fresh

    def step( self ) -> dict:
        E, g = self._evaluate()

        if self._prev_g is not None and self._prev_dx is not None:
            dg = g - self._prev_g
            self.H = _bofill_update( self.H, self._prev_dx, dg )

            # Safety trust region method.
            if self._prev_pred_dE is not None and abs(self._prev_pred_dE) > 1e-12:
                rho = (E - self._prev_E) / self._prev_pred_dE
                step_norm_prev = float( np.linalg.norm(self._prev_dx) )
                if rho > 0.75 and step_norm_prev > 0.8 * self.trust:
                    self.trust = min(2.0 * self.trust, self.max_trust)
                elif rho < 0.25:
                    self.trust = max(0.5 * self.trust, self.min_trust)

        # Periodic Lanczos refresh — re-anchor the unstable mode against the
        # true Hessian at the current geometry. Done *after* the Bofill update
        # for the previous step so the refresh overrides the latest H model.
        if ( self.relanczos_every is not None
             and len(self.history) > 0
             and len(self.history) % self.relanczos_every == 0 ):
            self._refresh_unstable_mode()

        # Cartesian workaround.
        V = _trans_rot_basis( self.x.reshape(self._shape) )
        g_act = _project_vec( g, V )
        g_norm = float( np.linalg.norm(g_act) )
        H_act = _project_mat( self.H, V )

        # Main P-RFO step
        dx, follow_vec, lam_followed = _prfo_step( H_act, g_act, self._follow_vec )

        step_norm = float( np.linalg.norm(dx) )
        if step_norm > self.trust:
            dx *= self.trust / step_norm
            step_norm = self.trust

        # Diagnostic overlaps for mode-following stability:
        #   overlap_with_init: how aligned is the currently-followed eigvec
        #     with the original seed `u`? Decreases as the reaction coordinate
        #     bends in configuration space. Sharp drops indicate the optimizer
        #     has switched to a different physical mode.
        #   overlap_with_prev: how aligned is the current followed eigvec with
        #     the one from the *previous* iteration? Stays > 0.9 in a stable
        #     climb; drops below ~0.7 = mode swap on that iteration.
        if self._init_follow_vec is not None:
            overlap_with_init = float( abs(follow_vec @ self._init_follow_vec) )
        else:
            overlap_with_init = None
        if self._follow_vec is not None:
            overlap_with_prev = float( abs(follow_vec @ self._follow_vec) )
        else:
            overlap_with_prev = None

        # Keep history.
        info = dict(
            energy=E,
            grad_norm=g_norm,
            step_norm=step_norm,
            trust=self.trust,
            followed_eigval=float(lam_followed),
            overlap_with_init=overlap_with_init,
            overlap_with_prev=overlap_with_prev,
        )
        self.history.append(info)

        # Maintain state for the next step - specifically the Hessian update.
        pred_dE = float( g_act @ dx + 0.5 * dx @ (H_act @ dx) )
        self._prev_g = g
        self._prev_dx = dx
        self._prev_pred_dE = pred_dE
        self._prev_E = E
        self._follow_vec = follow_vec
        self.x = self.x + dx

        return info

    def run( self, max_iter: int = 200, 
                   tol_g: float = 5e-4,
                   tol_step: float = 1e-4, 
                   verbose: bool = False
            ) -> dict:
        info: dict = {}
        for it in range(max_iter):
            info = self.step()

            if verbose:
                print(f"[{it:4d}] E={info['energy']:+.6f}  "
                      f"|g|={info['grad_norm']:.3e}  "
                      f"|dx|={info['step_norm']:.3e}  "
                      f"trust={info['trust']:.3e}  "
                      f"lam_follow={info['followed_eigval']:+.3e}")
            
            # Check for convergence.
            if info["grad_norm"] < tol_g and info["step_norm"] < tol_step:
                return dict(converged=True, n_iter=it+1, x=self.x.reshape(self._shape), **info)
        
        # Max. iterations reached.
        return dict(converged=False, n_iter=max_iter, x=self.x.reshape(self._shape), **info)
