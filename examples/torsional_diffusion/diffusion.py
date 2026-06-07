"""
Diffusion-side utilities for torsional-diffusion training and sampling.

Two small helpers, both stateless:

    sigma_schedule(t, sigma_min, sigma_max)
        Geometric noise schedule σ(t) = σ_min^(1−t) σ_max^t. Default bounds
        from the paper: σ_min = 0.01π, σ_max = π.

    wrapped_normal_score(delta, sigma, num_terms)
        Closed-form ∇_{τ_t} log p_{t|0}(τ_t | τ_0) where p_{t|0} is the
        wrapped normal on the unit circle. Acts coordinate-wise — the
        wrapped-normal kernel on the m-torus factorises across coordinates,
        so per-bond evaluation is exact.

The trunk's radius graph is now built via
`chemdm.MoleculeGraph.findAllDistanceNeighbors`, which uses scipy KDTree and
handles batched-molecule separation natively via `molecule_id`. See
`train.py` for the call site.
"""

from __future__ import annotations

import math

import torch as pt


def sigma_schedule( t: pt.Tensor, sigma_min: float = 0.01 * math.pi, sigma_max: float = math.pi ) -> pt.Tensor:
    """
    Geometric noise schedule: σ(t) = σ_min^(1−t) σ_max^t.

    t ∈ [0, 1].  σ(0) = σ_min  (near-data),  σ(1) = σ_max  (uniform-prior end).

    Returns a tensor matching the shape of `t`.
    """
    return sigma_min ** (1.0 - t) * sigma_max ** t


def wrapped_normal_score( delta: pt.Tensor,  sigma: pt.Tensor, num_terms: int = 5 ) -> pt.Tensor:
    """
    Closed-form ∇log p_{t|0}(τ_t | τ_0) for the wrapped-normal kernel on the
    unit circle, evaluated at the (already-known) increment

        delta = τ_t - τ_0  ∈ ℝ.

    Wrapped-normal density (up to a normalising constant in τ_0):

        p_{t|0}(τ_t | τ_0)  ∝  Σ_{d ∈ ℤ}  exp(-(δ + 2π d)² / 2σ²)

    Gradient w.r.t. τ_t:

        ∇log p  =  -(1/σ²) ·  Σ_d (δ + 2π d) · w_d   /   Σ_d w_d

    where w_d = exp(-(δ + 2π d)² / 2σ²).

    The sum is truncated to d ∈ {−K, …, +K}. K = num_terms (default 5)
    is enough for σ up to about π — at larger σ the truncation error is
    bounded by exp(−(2π(K+1))² / 2σ²) ≲ 1e-14 at K=5, σ=π.

    Parameters
    ----------
    delta : float tensor
        Any shape; typically (m,) per-bond increments.
    sigma : float tensor
        Same shape as `delta` (already broadcast per-bond), or scalar.
    num_terms : int
        K in the truncated sum.

    Returns
    -------
    score : same shape as `delta`
    """
    K = num_terms
    d = pt.arange(-K, K + 1, device=delta.device, dtype=delta.dtype)            # (2K+1,)

    # Broadcast over the trailing wrap-index dim:
    shifted = delta.unsqueeze(-1) + 2.0 * math.pi * d                           # (..., 2K+1)
    sigma_b = sigma.unsqueeze(-1) if sigma.ndim == delta.ndim else sigma        # (..., 1) or scalar
    log_w = -0.5 * (shifted / sigma_b) ** 2                                   # (..., 2K+1)

    # `num / denom` over the unnormalised weights is just the softmax-weighted
    # average of `shifted` with logits `log_w`. F.softmax handles the stability
    # (max-subtract) internally.
    weights = pt.nn.functional.softmax( log_w, dim=-1 )                           # (..., 2K+1)
    return -(shifted * weights).sum(dim=-1) / sigma ** 2
