"""Path smoothing utilities for transition-path initial guesses.

A predicted transition path (..., n_images, n_atoms, 3) can be wiggly
image-to-image; penalized-least-squares smoothing damps the second difference
along the image axis (an acceleration penalty) while keeping the endpoints
fixed, which tends to give NEB a cleaner starting band.
"""
from __future__ import annotations

import sys

import numpy as np
import scipy.linalg as la


def smooth_path_penalized_least_squares( Y: np.ndarray,  # (..., n_images, n_atoms, 3)
                                         alpha: float = 1.0,
                                        ) -> np.ndarray:
    """
    Smooth a path using penalized least squares:

        min_X  0.5 ||X - Y||^2 + 0.5 alpha ||D2 X||^2

    with fixed endpoints. D2 is the second difference along the image axis.

    Parameters
    ----------
    Y: ndarray
        Array of shape (..., n_images, n_atoms, 3). The last two dimensions are
        the per-image geometry (n_atoms, 3); the third-from-last axis is the
        image/path axis that gets smoothed. Any leading dimensions are treated
        as a batch and smoothed independently in a single vectorized solve, e.g.
        a plain path (n_images, n_atoms, 3) or a batch (n_paths, n_images,
        n_atoms, 3).
    alpha: float
        Smoothing strength. Larger means smoother.

    Returns
    -------
    X:
        Smoothed array with the same shape as Y.
    """
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim < 3 or Y.shape[-1] != 3:
        raise ValueError(
            f"Y must have shape (..., n_images, n_atoms, 3) with the image axis "
            f"third-from-last; got ndim={Y.ndim}."
        )
    if alpha < 0.0:
        raise ValueError( "alpha must be non-negative." )
    sqrt_alpha = np.sqrt( alpha )

    # Image axis is third-from-last; (n_atoms, 3) are the last two.
    img_axis = -3
    n_images = Y.shape[img_axis]
    if n_images <= 2 or alpha == 0.0:
        return Y.copy()

    # Move the image axis to the front and flatten everything else (batch dims
    # plus n_atoms and 3) into columns. The penalized-least-squares system G
    # depends only on (n_images, alpha), so every column shares it and a single
    # lstsq smooths the whole batch at once.
    Ym = np.moveaxis( Y, img_axis, 0 )      # (n_images, *rest)
    rest_shape = Ym.shape[1:]
    Y_flat = Ym.reshape( n_images, -1 )     # (n_images, n_cols)
    n_internal = n_images - 2

    # Unknowns are X[1:-1].
    #
    # D2 rows are:
    #
    #   X[0] - 2 X[1] + X[2]
    #   X[1] - 2 X[2] + X[3]
    #   ...
    #   X[-3] - 2 X[-2] + X[-1]
    #
    # After eliminating fixed endpoints:
    #
    #   D2 X_full = A X_internal + b
    #
    # A has shape (n_images - 2, n_images - 2).
    #
    # For internal unknowns, the stencil is [1, -2, 1], but the first
    # and last rows are missing one contribution because those are endpoints.
    A = -2.0 * np.eye( n_internal, dtype=np.float64 )
    if n_internal > 1:
        idx = np.arange(n_internal - 1)
        A[idx, idx + 1] = 1.0
        A[idx + 1, idx] = 1.0

    # Endpoint contribution b.
    # Only first and last second-difference rows involve fixed endpoints.
    b = np.zeros_like( Y_flat[1:-1] )
    b[0] += Y_flat[0]
    b[-1] += Y_flat[-1]

    # Direct augmented least-squares form:
    #
    #   min || [I; sqrt(alpha) A] X_internal - [Y_internal; -sqrt(alpha) b] ||^2
    G = np.vstack( [ np.eye(n_internal), sqrt_alpha * A ] )
    rhs = np.vstack( [ Y_flat[1:-1], -sqrt_alpha * b ] )
    try:
        lstsq_result = la.lstsq(G, rhs, lapack_driver="gelsy")
        X_internal = np.asarray( lstsq_result[0], dtype=np.float64 ) # type: ignore
    except Exception:
        print( "Smoothing solver did not converge, continuing with unfiltered ML path", file=sys.stderr )
        return Y.copy()

    # Reassemble: internal images smoothed, endpoints untouched.
    X_flat = Y_flat.copy()
    X_flat[1:-1] = X_internal
    X_flat[0] = Y_flat[0]
    X_flat[-1] = Y_flat[-1]

    Xm = X_flat.reshape( (n_images,) + rest_shape )
    return np.moveaxis( Xm, 0, img_axis )
