"""Refine a transition state from a converged NEB path.

After NEB gives a minimum-energy band, the highest interior image is a good
guess for the saddle. This module polishes that guess into a true first-order
saddle point with P-RFO (chemdm.prfo), seeding the uphill mode-following with
the NEB reaction-coordinate tangent at that image, and (optionally) validates
the result with a finite-difference Hessian index check.

Units: the evaluator must return energies/forces in eV and eV/A (xTB via
chemdm.xtbSetup.XTBPotential does), matching P-RFO and the Lindh model Hessian.
The `energies` argument is used only to pick the seed image and weight the
tangent, both of which are invariant to the overall energy scale, so it may be
in any consistent unit.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch as pt

from chemdm.MoleculeGraph import MoleculeGraph
from chemdm.nebXtbDirect import improved_tangents
from chemdm.geometry import kabsch_align_numpy
from chemdm.prfo import PRFOOptimizer, EnergyForceEvaluator, _trans_rot_basis, _project_mat


def _fd_hessian( evaluator: EnergyForceEvaluator, 
                 x: np.ndarray, 
                 h: float ) -> np.ndarray:
    """Symmetric finite-difference Hessian (eV/A^2) via central force differences.
    Costs 2*(3N) gradient evaluations."""
    shape = x.shape
    xf = x.reshape(-1).astype(float)
    n = xf.size
    H = np.zeros((n, n))
    for i in range(n):
        xp = xf.copy(); xp[i] += h
        xm = xf.copy(); xm[i] -= h
        _, Fp = evaluator.energy_forces( xp.reshape(shape) )
        _, Fm = evaluator.energy_forces( xm.reshape(shape) )
        H[i, :] = -( np.asarray(Fp).reshape(-1) - np.asarray(Fm).reshape(-1) ) / (2.0 * h)
    return 0.5 * (H + H.T)


def is_transition_state( evaluator: EnergyForceEvaluator, 
                         x_ts : np.ndarray, 
                         fd_step : float,
                         eig_tol: float = 1e-3 ) -> bool:
    """First-order-saddle check: a TS has exactly one negative Hessian
    eigenvalue. Build the finite-difference Hessian, project out the 6 rigid-body
    (translation/rotation) null modes, and count negative eigenvalues among the
    physical ones (|lambda| > eig_tol eV/A^2 to ignore numerical near-zero modes)."""
    H = _fd_hessian( evaluator, x_ts, fd_step )
    H = _project_mat( H, _trans_rot_basis( x_ts ) )   # remove trans/rot null modes
    eigvals = np.linalg.eigvalsh( H )
    physical = eigvals[ np.abs(eigvals) > eig_tol ]
    n_negative = int( np.sum(physical < 0.0) )
    
    return n_negative == 1


def locate_on_path( path: np.ndarray,
                    x: np.ndarray,
                    tol: float
                  ) -> tuple[bool, int]:
    """Locate a geometry on a path and decide whether it actually lies on it.

    Find the segment `x` projects onto, then test its perpendicular distance to
    that segment against `tol`. A query that genuinely sits on the band (e.g. a
    TS that P-RFO climbed to from *this* reaction) projects onto an interior
    segment at small distance; a foreign saddle the optimizer wandered off to
    lands far from every segment. General-purpose -- works for any query
    geometry, not just a TS.

    Parameters
    ----------
    path : (M, n_atoms, 3) images.
    x    : (n_atoms, 3) query geometry, assumed already in the same rigid frame
           as `path` (the caller aligns it if needed).
    tol  : per-atom RMSD (Angstrom) below which `x` is considered on the path.

    Returns
    -------
    on_path : bool
        True if x lies within `tol` (per-atom RMSD) of its nearest segment.
    index : int
        Index of the path segment x projects onto.

    """
    M, n_atoms, _ = path.shape
    flat = path.reshape(M, -1)
    xf = x.reshape(-1)

    # Projection: closest point on each segment, then the nearest segment.
    A = flat[:-1]      # (M-1, 3N) segment starts
    AB = flat[1:] - A  # (M-1, 3N) segment vectors a->b
    L2 = np.einsum( "ij,ij->i", AB, AB )  # (M-1,) squared segment lengths
    
    d = xf - A                                                     # (M-1, 3N)
    # t_i = clip((x - A_i) . AB_i / |AB_i|^2, 0, 1); zero-length segments -> t=0.
    t = np.divide( np.einsum( "ij,ij->i", d, AB ), L2, out=np.zeros_like(L2), where=L2 > 0 )
    t = np.clip( t, 0.0, 1.0 )

    # Distance from x to that closest point: x - proj = (x - A) - t*AB = d - t*AB.
    diff = d - t[:, None] * AB
    d2 = np.einsum( "ij,ij->i", diff, diff )  # (M-1,)
    i = int( np.argmin(d2) )

    # d2[i] is the summed squared Cartesian deviation over all atoms; convert to
    # a per-atom RMSD so `tol` is independent of molecule size, then test whether
    # x is close enough to the band to belong to it.
    rmsd = float( np.sqrt( d2[i] / n_atoms ) )
    on_path = rmsd <= tol

    return on_path, i


def refine_ts_from_path( evaluator: EnergyForceEvaluator,
                         Z: np.ndarray,
                         path: np.ndarray,        # (M, n_atoms, 3), Angstrom
                         energies: np.ndarray,    # (M,), seed selection only
                         bonds: np.ndarray,       # (E, 2), for the Lindh H0 model. Typically union of reactant and product bonds
                         *,
                         max_iter: int = 200,
                         tol_g: float = 5e-4,
                         tol_step: float = 1e-4,
                         trust_radius: float = 1e-4,
                         relanczos_every: Optional[int] = None,
                         fd_step: float = 1e-3,
                         on_path_tol: float = 0.5,
                         verbose: bool = False
                        ) -> tuple[np.ndarray, float, np.ndarray, int]:
    """
    Refine the transition state from a converged NEB path.

    The initial guess is the highest-energy *interior* image (endpoints excluded, since they are
    the relaxed reactant/product, not the barrier). The NEB improved tangent at
    that image seeds P-RFO's uphill mode-following. P-RFO then climbs to the
    first-order saddle.
    """
    path = np.asarray(path, dtype=np.float64)
    E = np.asarray(energies, dtype=np.float64)
    M = path.shape[0]
    if path.ndim != 3 or M < 3:
        raise ValueError( f"need at least 3 images (2 endpoints + 1 interior); got {M}." )

    # Seed: highest-energy interior image (exclude endpoints).
    ts_idx = 1 + int( np.argmax(E[1:-1]) )
    x_seed = path[ts_idx]

    # NEB reaction-coordinate tangent at the seed image -> uphill follow mode.
    tau = improved_tangents( path, E )[ts_idx - 1].reshape(-1)   # interior-indexed

    molecule = MoleculeGraph(
        Z=pt.as_tensor( np.asarray(Z, dtype=np.int64) ),
        x=pt.as_tensor( x_seed, dtype=pt.float64 ),
        bonds=pt.as_tensor( np.asarray(bonds, dtype=np.int64) ),
    )
    opt = PRFOOptimizer( evaluator, molecule, init_follow_vec=tau,
                         trust_radius=trust_radius, relanczos_every=relanczos_every,
                         min_trust=trust_radius, max_trust=0.01 )
    res = opt.run( max_iter=max_iter, tol_g=tol_g, tol_step=tol_step, verbose=verbose )

    # Check if the computed point is a transition state.
    x_ts = np.asarray( res["x"], dtype=np.float64 )
    E_ts, F_ts = evaluator.energy_forces( x_ts )
    if not is_transition_state( evaluator, x_ts, fd_step ):
        return x_ts, E_ts, F_ts, -1

    # Where the refined TS sits on the band. Kabsch-align it onto its seed
    # image first to strip any residual rigid drift (seed and TS are
    # near-identical shapes, so the all-atom fit is well posed and robust for
    # tiny molecules with <3 heavy atoms). A TS that belongs to *this*
    # reaction lands in the interior at small distance; one P-RFO wandered
    # off to shows up far from the band.
    fits_on_path, ts_index = locate_on_path( path, kabsch_align_numpy( x_ts, x_seed ), on_path_tol )
    if not fits_on_path:
        ts_index = -1
    return x_ts, E_ts, F_ts, ts_index
