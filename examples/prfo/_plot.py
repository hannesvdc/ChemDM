"""
Shared plotting helper for the P-RFO example scripts.

Pulled out of `hcn_to_hnc.py` so multiple demos can share the same
diagnostic two-panel plot without duplicating ~40 lines of matplotlib.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_prfo_trajectory( history: list[dict], title_suffix: str = "" ) -> None:
    """
    Two-panel diagnostic plot of a P-RFO run.

    Top: followed eigenvalue per iteration, with reference line at 0.
    Bottom: eigenvector overlaps with the initial seed `u` (orange) and
            with the previous iteration's followed vector (green), plus a
            0.7 mode-swap-threshold reference line.

    Overlap fields may be `None` on iteration 0 or when `init_mode=None`;
    those iterations are filtered out of the overlap panel.
    """
    iters = np.arange( len(history) )
    lam_followed = np.array( [info["followed_eigval"] for info in history] )

    init_xy = [(t, info["overlap_with_init"]) for t, info in enumerate(history)
               if info.get("overlap_with_init") is not None]
    prev_xy = [(t, info["overlap_with_prev"]) for t, info in enumerate(history)
               if info.get("overlap_with_prev") is not None]

    fig, (ax_top, ax_bot) = plt.subplots( 2, 1, figsize=(7, 6.5), sharex=True )

    ax_top.plot( iters, lam_followed, marker="o", markersize=4, linewidth=1.0,
                 color="tab:blue", label=r"$\lambda_{\rm follow}$" )
    ax_top.axhline( 0.0, color="black", linestyle="--", linewidth=0.7, alpha=0.6 )
    ax_top.set_ylabel( r"followed eigenvalue (eV/Å²)" )
    title = f"P-RFO trajectory ({len(history)} steps)"
    if title_suffix:
        title += f"  —  {title_suffix}"
    ax_top.set_title( title )
    ax_top.grid( True, alpha=0.3 )
    ax_top.legend( loc="best", fontsize=9 )

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
