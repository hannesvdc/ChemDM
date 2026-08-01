"""
P-RFO demo: HCN -> HNC isomerization transition state with xTB.

1. Build an HCN MoleculeGraph and relax to the GFN2-xTB minimum.
2. From that minimum, estimate the lowest-curvature mode via Lindh seed +
   dimer rotation (no chemical intuition required).
3. Perturb along that mode and run P-RFO to the saddle.
4. Verify: exactly one negative Hessian eigenvalue, report the barrier.

Run:
    python examples/prfo/hcn_to_hnc.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch as pt

from chemdm.MoleculeGraph import MoleculeGraph
from chemdm.prfo import PRFOOptimizer, lindh_lowest_mode
from chemdm.relaxMolecule import relaxMolecule
from chemdm.potentials.TBLitePotential import TBLitePotential
from chemdm.Constants import EV_TO_KCAL_PER_MOL
from chemdm.prfo import _project_mat, _trans_rot_basis


def plot_prfo_trajectory( history: list[dict] ) -> None:
    """
    Two-panel diagnostic plot of a P-RFO run.

    Top panel — *followed eigenvalue per iteration*.
        A clean climb shows λ_followed staying negative throughout (early
        iterations report the artificial -1 from the rank-1 overlay; only
        after ~5 Bofill updates does the value become physically meaningful).
        Excursions to positive values indicate the mode-following heuristic
        briefly lost the unstable direction.

    Bottom panel — *eigenvector overlaps* with the initial seed `u` and with
    the previous iteration's followed vector.
        `overlap_with_prev` is the direct diagnostic for mode-following
        stability: drops below ~0.7 indicate a mode swap between consecutive
        iterations. `overlap_with_init` shows cumulative drift away from the
        original Lanczos direction; for a clean climb along a single reaction
        coordinate it should decrease smoothly as the eigenvector rotates
        with the geometry.
    """
    iters = np.arange( len(history) )
    lam_followed = np.array( [info["followed_eigval"] for info in history] )

    # Overlaps may be None on iter 0 or when init_mode=None; filter and align.
    init_xy = [(t, info["overlap_with_init"]) for t, info in enumerate(history)
               if info.get("overlap_with_init") is not None]
    prev_xy = [(t, info["overlap_with_prev"]) for t, info in enumerate(history)
               if info.get("overlap_with_prev") is not None]

    fig, (ax_top, ax_bot) = plt.subplots( 2, 1, figsize=(7, 6.5), sharex=True )

    # Top: followed eigenvalue
    ax_top.plot( iters, lam_followed, marker="o", markersize=4, linewidth=1.0,
                 color="tab:blue", label=r"$\lambda_{\rm follow}$" )
    ax_top.axhline( 0.0, color="black", linestyle="--", linewidth=0.7, alpha=0.6 )
    ax_top.set_ylabel( r"followed eigenvalue (eV/Å²)" )
    ax_top.set_title( f"P-RFO trajectory ({len(history)} steps)" )
    ax_top.grid( True, alpha=0.3 )
    ax_top.legend( loc="best", fontsize=9 )

    # Bottom: overlaps
    if init_xy:
        ts, vals = zip(*init_xy)
        ax_bot.plot( ts, vals, marker="o", markersize=4, linewidth=1.0,
                     color="tab:orange", label=r"$|u_t \cdot u_0|$  (vs initial)" )
    if prev_xy:
        ts, vals = zip(*prev_xy)
        ax_bot.plot( ts, vals, marker="s", markersize=4, linewidth=1.0,
                     color="tab:green", label=r"$|u_t \cdot u_{t-1}|$  (vs previous)" )
    ax_bot.axhline( 0.7, color="black", linestyle="--", linewidth=0.7, alpha=0.6,
                    label="0.7 (mode-swap threshold)" )
    ax_bot.set_ylim( -0.05, 1.05 )
    ax_bot.set_xlabel( "P-RFO iteration" )
    ax_bot.set_ylabel( "eigenvector overlap" )
    ax_bot.grid( True, alpha=0.3 )
    ax_bot.legend( loc="best", fontsize=9 )

    fig.tight_layout()
    plt.show()


def finite_difference_hessian( potential: TBLitePotential, 
                               x: np.ndarray,
                               h: float = 1e-3
                             ) -> np.ndarray:
    """Central-difference Cartesian Hessian (eV/Å²) for diagnostics only."""
    n = x.size
    H = np.zeros( (n, n) )
    x_flat = x.reshape(-1).copy()
    for i in range(n):
        xp = x_flat.copy(); xp[i] += h
        xm = x_flat.copy(); xm[i] -= h
        _, Fp = potential.energy_forces(xp.reshape(x.shape))
        _, Fm = potential.energy_forces(xm.reshape(x.shape))
        # H_ij = -dF_j/dx_i; use central diff in x_i.
        H[i, :] = -(Fp.reshape(-1) - Fm.reshape(-1)) / (2.0 * h)
    return 0.5 * (H + H.T)


def main() -> None:
    Z = np.array([1, 6, 7]) # H-C=N
    potential = TBLitePotential(Z=Z)

    # HCN guess, atoms collinear along x. H-C ≈ 1.07 Å, C-N ≈ 1.16 Å.
    x_init = np.array([
        [-1.07, 0.0, 0.0],
        [ 0.00, 0.0, 0.0],
        [ 1.16, 0.0, 0.0],
    ])

    print("=== Relaxing HCN minimum (GFN2-xTB) ===")
    x_min = relaxMolecule( potential, x_init, minimizer="Adam", force_tol=1e-3, max_steps=2000, verbose=False )
    E_min, F_min = potential.energy_forces( x_min ) # type: ignore
    print( f"E_min  = {E_min:+.6f} eV" )
    print( f"|F|max = {np.abs(F_min).max():.2e} eV/Å" )
    print( "HCN minimum geometry (Å):" )
    for sym, p in zip("HCN", x_min):
        print(f"  {sym}  {p[0]:+.4f}  {p[1]:+.4f}  {p[2]:+.4f}")

    # Build MoleculeGraph at the minimum (H-C and C-N bonds).
    hcn_min = MoleculeGraph(
        Z=pt.tensor(Z),
        x=pt.tensor(x_min, dtype=pt.float64),
        bonds=pt.tensor([[0, 1], [1, 0], [1, 2], [2, 1]], dtype=pt.long),
    )

    print("\n=== Estimating lowest mode (mass-weighted Lindh) ===")
    u = lindh_lowest_mode( hcn_min )
    u_3d = u.reshape(3, 3)
    print( "mode (per-atom components):" )
    for sym, c in zip("HCN", u_3d):
        print(f"  {sym}  {c[0]:+.3f}  {c[1]:+.3f}  {c[2]:+.3f}")

    # Perturb off the minimum along the discovered mode. At the minimum
    # |g| is below tol and P-RFO would declare convergence immediately; a
    # finite kick gives the optimizer real gradient to work with from step 1.
    kick = 0.5
    x_start = x_min #+ kick * u_3d
    hcn_start = hcn_min.copyWithNewPositions( pt.tensor(x_start, dtype=pt.float64) )
    print( f"\nPerturbing by {kick:.2f} Å along the lowest mode before P-RFO." )

    print( "\n=== P-RFO ascent toward TS ===" )
    # PRFOOptimizer's `init_mode="lindh"` default re-runs lindh_lowest_mode
    # internally at the perturbed start to seed mode-following — zero force
    # calls, no double-compute concern.
    min_trust = 1e-3
    max_trust = 0.1
    opt = PRFOOptimizer( potential, hcn_start, trust_radius=min_trust, max_trust=max_trust, min_trust=min_trust, relanczos_every=5 )

    # Wrap opt.run() so the diagnostic trajectory is still plotted even when
    # the evaluator (e.g. xTB) crashes mid-run. On crash we synthesise a
    # partial `result` from `opt.history` so the rest of main() can react.
    try:
        result = opt.run( max_iter=200, tol_g=1e-3, tol_step=1e-4, verbose=True )
    except Exception as e:
        n_done = len(opt.history)
        print( f"\nOptimizer crashed after {n_done} iterations: {type(e).__name__}: {e}" )
        last_info = opt.history[-1] if opt.history else {}
        result = {
            "converged": False,
            "crashed":   True,
            "exception": f"{type(e).__name__}: {e}",
            "n_iter":    n_done,
            "x":         opt.x.reshape(opt._shape).copy(),
            "energy":    last_info.get("energy"),
            "grad_norm": last_info.get("grad_norm"),
        }


    if not result["converged"]:
        if result.get("crashed"):
            print( f"\nOptimization did NOT converge — crash at iter {result['n_iter']}." )
        else:
            print( "\nOptimization did NOT converge within max_iter." )
        return
    x_ts = result["x"]
    E_ts = result["energy"]

    print( "\n=== TS candidate ===" )
    print( f"converged in {result['n_iter']} iterations" )
    print( f"E_ts   = {E_ts:+.6f} eV" )
    print( f"barrier = {(E_ts - E_min) * EV_TO_KCAL_PER_MOL:+.2f} kcal/mol" )
    print( "TS geometry (Å):" )
    for sym, p in zip("HCN", x_ts):
        print(f"  {sym}  {p[0]:+.4f}  {p[1]:+.4f}  {p[2]:+.4f}")

    print( "\n=== Verifying saddle character (finite-difference Hessian) ===" )
    H_fd = finite_difference_hessian( potential, x_ts, h=1e-3 )
    V = _trans_rot_basis( x_ts ) # Project trans/rot out before checking eigenvalues.
    H_proj = _project_mat( H_fd, V )
    eigvals = np.linalg.eigvalsh( H_proj )
    # Drop the (3) numerical zeros from projection.
    physical = eigvals[np.abs(eigvals) > 1e-3]
    n_neg = int( np.sum(physical < 0.0) )
    print(f"physical eigenvalues (eV/Å²): {np.array2string(physical, precision=3)}")
    print(f"# negative = {n_neg}  -> {'first-order saddle ✓' if n_neg == 1 else 'NOT a clean saddle ✗'}")

    # Plot the trajectory whether or not we converged --- it is most
    # diagnostic precisely when mode-following struggled.
    plot_prfo_trajectory( opt.history )

if __name__ == "__main__":
    main()
