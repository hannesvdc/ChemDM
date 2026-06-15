"""
Partitioned Rational Function Optimization (P-RFO) for single-ended TS search.

Cartesian coordinates with translation/rotation projection. Quasi-Newton Hessian
updated via Bofill — only forces are required, no analytic Hessian.

References:
  Banerjee, Adams, Simons, Shepard, J. Phys. Chem. 89, 52 (1985).
  Bofill, J. Comput. Chem. 15, 1 (1994).
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, Tuple, Union

import numpy as np

from chemdm.MoleculeGraph import Molecule
from chemdm.Constants import _HARTREE_PER_BOHR2_TO_EV_PER_ANG2
from chemdm.lanczos import lanczos_lowest


# Evaluator interface
class EnergyForceEvaluator(Protocol):
    def energy_forces(self, x: np.ndarray) -> Tuple[float, np.ndarray]:
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

    # Translational basis vectors
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
# Lindh, Chem. Phys. Lett. 241, 423 (1995). Pairwise stretch contributions
# only; angle bends and torsions are omitted. The bend coordinate's Wilson B
# matrix is singular at linear angles (HCN, CO2, ...) and recovering bend
# stiffness there requires auxiliary linear-bend coordinates — extra
# machinery not worth it for a seed Hessian. `estimate_lowest_mode` refines
# the lowest mode via dimer rotation regardless of the seed quality.

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


def lindh_model_hessian( molecule: Any ) -> np.ndarray:
    """
    Lindh 1995 pairwise stretch model Hessian. Essentially populates the Hessian
    with bond-length force constants, multiplied by Cartesian -> internal bond
    length Jacobians (if using Cartesian coordinates).

    Parameters
    ----------
    molecule : chemdm.MoleculeGraph.Molecule
        Provides `x` (positions, Å), `Z` (atomic numbers) and `edge_index`
        (bond list, directed or undirected — duplicates are filtered).

    Returns
    -------
    H : (3N, 3N) ndarray, eV/Å²
        Symmetric, PSD up to numerical roundoff. Has 6 zero modes (trans/rot)
        for non-linear molecules, 5 for linear ones.

    Notes
    -----
    Includes only bond-stretch contributions. Bend modes are *missing* —
    the lowest eigenmode of this matrix will be a low-frequency stretch, not
    a bend, for systems where bends are softest (e.g. HCN). Use
    `estimate_lowest_mode` to refine to the true lowest mode of the real
    potential via dimer rotation.
    """
    x = _to_numpy(molecule.x).astype(float)
    Z = _to_numpy(molecule.Z).astype(int).flatten()
    edges = _to_numpy(molecule.edge_index).astype(int) # (E,2)

    n = x.shape[0]
    H = np.zeros((3 * n, 3 * n))

    bonds = _unique_bonds( edges )
    for i, j in bonds:
        v = x[j] - x[i]
        r = float( np.linalg.norm(v) )
        if r < 1e-8:
            continue
        d = v / r

        ri, rj = _atom_row(int(Z[i])), _atom_row(int(Z[j]))
        alpha = _LINDH_ALPHA[ (ri, rj) ]
        r_ref = _LINDH_RREF[ (ri, rj) ]
        rho = np.exp(alpha * (r_ref ** 2 - r ** 2))
        k = _LINDH_K_STRETCH * rho

        dd = np.outer(d, d)
        ii = slice(3 * i, 3 * i + 3)
        jj = slice(3 * j, 3 * j + 3)
        H[ii, ii] += k * dd
        H[jj, jj] += k * dd
        H[ii, jj] -= k * dd
        H[jj, ii] -= k * dd

    return H


# ============================================================
# Lowest-mode estimation: Lindh seed + dimer rotation
# ============================================================

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
    Estimate the lowest-curvature mode of the true Hessian at `molecule.x` by
    Lanczos iteration with finite-difference Hessian-vector products. Trans/rot
    modes are always projected out.

    Seeds the Lanczos iteration from `lindh_model_hessian(molecule)`'s lowest
    eigenmode (or from `init_u` if provided). Each Lanczos step costs one
    extra force call (forward-difference `Hv ≈ (g(x+εv) - g₀)/ε`).

    Returns
    -------
    u : (3N,) unit vector
        Lowest-mode estimate, lying in the trans/rot-free subspace.
    lam : float
        Smallest Ritz value (eV/Å²).
    """
    x = _to_numpy( molecule.x ).astype(float)
    n_atoms = x.shape[0]
    dim = 3 * n_atoms

    V = _trans_rot_basis(x)

    _, F0 = evaluator.energy_forces(x)
    g0 = -np.asarray(F0, dtype=float).reshape(-1)

    def Hv_fd( v: np.ndarray ) -> np.ndarray:
        x_plus = x + eps * v.reshape(n_atoms, 3)
        _, F_plus = evaluator.energy_forces(x_plus)
        g_plus = -np.asarray(F_plus, dtype=float).reshape(-1)
        return (g_plus - g0) / eps

    if init_u is None:
        H_model = lindh_model_hessian( molecule )
        H_act = _project_mat( H_model, V )
        eigvals, eigvecs = np.linalg.eigh( H_act )
        scale = max(abs(eigvals).max(), 1e-6)
        nonzero = np.abs(eigvals) > 1e-6 * scale
        if nonzero.any():
            u0 = eigvecs[:, nonzero][:, int(np.argmin(eigvals[nonzero]))]
        else:
            u0 = np.random.default_rng(0).standard_normal(dim)
    else:
        # Start from the initial ascent direction provided by the user.
        # Should not really happen often.
        u0 = np.asarray( init_u, dtype=float ).reshape(-1)

    return lanczos_lowest(
        Hv_fd, u0,
        max_iter=max_iter,
        tol=tol,
        V_project=V,
        seed_noise=seed_noise,
        random_state=random_state,
    )


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
    init_mode : {"lanczos", None}, default "lanczos"
        How to seed the unstable direction in the initial quasi-Newton Hessian.

        - "lanczos" (default): call :func:`estimate_lowest_mode` internally
          (Lindh-seeded Lanczos against the true Hessian via finite-difference
          gradient probes), then add a rank-1 `-2 u uᵀ` overlay so iteration 1
          ascends along the discovered mode. Costs ~10 extra force calls upfront.
        - None: skip the auto-seed. Initial Hessian is `I`; the optimizer must
          discover the unstable mode by accident via Bofill. Cheap but often
          fragile — use only when you have other reasons to skip the seed
          (e.g.\\ the evaluator is a stand-in / mock).
    trust_radius, max_trust, min_trust : float
        Trust-region bounds on the Cartesian step (Angstrom).
    """
    _INIT_MODE_CHOICES = ( "lanczos", )

    def __init__(self, evaluator: EnergyForceEvaluator,
                 molecule: Molecule, *,
                 init_mode: Optional[str] = "lanczos",
                 trust_radius: float = 0.3,
                 max_trust: float = 0.5,
                 min_trust: float = 0.01):
        if not isinstance(molecule, Molecule):
            raise TypeError(
                f"PRFOOptimizer requires a chemdm.MoleculeGraph.Molecule; got {type(molecule).__name__}."
            )
        if init_mode is not None and init_mode not in self._INIT_MODE_CHOICES:
            raise ValueError( f"init_mode must be None or one of {self._INIT_MODE_CHOICES}; got {init_mode!r}." )

        self.molecule = molecule
        x0 = _to_numpy(molecule.x).astype(float)
        self._shape = x0.shape
        self.evaluator = evaluator
        self.x = x0.flatten().copy()
        self.dim = self.x.size

        # Identity Hessian; the rank-1 negative-curvature overlay is what gives
        # P-RFO an ascent direction on iteration 1. Without it, Bofill has to
        # discover the unstable mode from gradient differences — slow and
        # sometimes fragile.
        self.H = np.eye( self.dim )
        if init_mode == "lanczos":
            u, _ = estimate_lowest_mode( evaluator, molecule )
            # `estimate_lowest_mode` already projected u onto the trans/rot-free
            # subspace and normalised it.
            self.H = self.H - 2.0 * np.outer(u, u)

        # Outer 'safety' trust-region parameters.
        # Not strictly necessary but enhances stability.
        self.trust = float(trust_radius)
        self.max_trust = float(max_trust)
        self.min_trust = float(min_trust)

        # Bookkeeping for state maintenance and logging.
        self._follow_vec: Optional[np.ndarray] = None
        self._prev_g: Optional[np.ndarray] = None
        self._prev_dx: Optional[np.ndarray] = None
        self._prev_pred_dE: Optional[float] = None
        self._prev_E: Optional[float] = None

        self.history: list[dict] = []

    def _evaluate(self) -> Tuple[float, np.ndarray]:
        E, F = self.evaluator.energy_forces( self.x.reshape(self._shape) )
        g = -np.asarray( F, dtype=float ).reshape(-1)
        return float(E), g

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

        # Cartesian workaround.
        V = _trans_rot_basis( self.x.reshape(self._shape) )
        g_act = _project_vec( g, V )
        g_norm = float( np.linalg.norm(g_act) )
        H_act = _project_mat( self.H, V )

        # Main P-RFO step
        dx, follow_vec, lam_followed = _prfo_step(H_act, g_act, self._follow_vec)

        step_norm = float( np.linalg.norm(dx) )
        if step_norm > self.trust:
            dx *= self.trust / step_norm
            step_norm = self.trust

        # Keep history.
        info = dict(
            energy=E,
            grad_norm=g_norm,
            step_norm=step_norm,
            trust=self.trust,
            followed_eigval=float(lam_followed),
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
