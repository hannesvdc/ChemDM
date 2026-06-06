"""
Smoke test for TorsionalScoreNetwork.

Run from this directory so the relative imports in score_network.py resolve:

    /opt/homebrew/anaconda3/envs/py311/bin/python test_score_network.py

What it checks:
    1. Forward pass returns the expected shape and finite values.
    2. Translation invariance: δτ(x + t) = δτ(x).
    3. Rotation invariance: δτ(Rx) = δτ(x) for R ∈ SO(3).
    4. Parity equivariance: δτ(−x) = −δτ(x).

The parity test is the architecturally interesting one: torsional diffusion
requires the score to flip sign under reflection (the molecular energy is
parity-invariant, so log p is parity-invariant, so its derivative is parity-
odd). All four checks are run in float32 and again in float64 — the float32
deltas reflect accumulated roundoff in the equivariant TPs, while the float64
deltas should be at the level of machine epsilon for a correctly built model.
"""

from __future__ import annotations

import math
import sys

import torch as pt

# Allow this script to be invoked from anywhere; ensure the example dir is on
# sys.path so the bare `from attention import ...` style imports in
# score_network.py resolve.
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from score_network import TorsionalScoreNetwork


def build_toy_batch(dtype: pt.dtype) -> dict:
    """
    Two tiny molecules of 8 atoms each, fully-connected attention graph
    within each molecule, 2 rotatable bonds per molecule.
    """
    pt.manual_seed(0)

    N1, N2 = 8, 8
    N = N1 + N2

    Z = pt.randint(1, 10, (N,))
    x = pt.randn(N, 3, dtype=dtype)
    sigma = pt.tensor([0.3, 1.5], dtype=dtype)

    def fully_connected(offset: int, n: int) -> pt.Tensor:
        idx = pt.arange(n) + offset
        src = idx.repeat_interleave(n)
        dst = idx.repeat(n)
        keep = src != dst
        return pt.stack([src[keep], dst[keep]], dim=0)

    edge_index = pt.cat([fully_connected(0, N1), fully_connected(N1, N2)], dim=1)

    bonds = pt.tensor([[0, 1], [2, 3], [N1 + 0, N1 + 1], [N1 + 2, N1 + 3]])
    atom_batch = pt.cat([pt.zeros(N1, dtype=pt.long), pt.ones(N2, dtype=pt.long)])
    bond_batch = pt.tensor([0, 0, 1, 1])

    return dict(
        Z=Z, x=x, sigma=sigma,
        edge_index=edge_index, bonds=bonds,
        atom_batch=atom_batch, bond_batch=bond_batch,
    )


def random_rotation(dtype: pt.dtype) -> pt.Tensor:
    # Rotation around z by 0.7 rad, kept simple for traceability.
    theta = 0.7
    c, s = math.cos(theta), math.sin(theta)
    return pt.tensor(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=dtype,
    )


def run_checks(dtype: pt.dtype) -> None:
    print(f"\n=== dtype = {dtype} ===")

    batch = build_toy_batch(dtype)
    model = TorsionalScoreNetwork().to(dtype)

    def call(x):
        return model(
            Z=batch["Z"], x=x, sigma=batch["sigma"],
            edge_index=batch["edge_index"], bonds=batch["bonds"],
            atom_batch=batch["atom_batch"], bond_batch=batch["bond_batch"],
        )

    out = call(batch["x"])
    print("output shape :", tuple(out.shape))
    print("output values:", [float(v) for v in out.detach()])
    assert out.shape == (4,), f"expected shape (4,), got {tuple(out.shape)}"
    assert pt.isfinite(out).all(), "non-finite output"

    # Tolerances: parity holds exactly (integer ±1 sign on each block, no
    # numerical accumulation); translation cancels exactly in (x[dst] - x[src])
    # but still picks up a tiny roundoff from |x|≫|edge|; rotation accumulates
    # the most error because Wigner D matrices and SH evaluations are not
    # algebraically exact.
    eps_exact = 0.0   if dtype == pt.float64 else 1.0e-10
    eps_trans = 1.0e-18 if dtype == pt.float64 else 1.0e-10
    eps_rot   = 1.0e-10 if dtype == pt.float64 else 1.0e-5

    t = pt.tensor([0.7, -1.3, 0.4], dtype=dtype)
    diff_t = (call(batch["x"] + t) - out).abs().max().item()
    print(f"translation max |Δ| : {diff_t:.3e}")
    assert diff_t < eps_trans, f"translation invariance violated: {diff_t:.3e} >= {eps_trans:.0e}"

    R = random_rotation(dtype)
    diff_r = (call(batch["x"] @ R.T) - out).abs().max().item()
    print(f"rotation    max |Δ| : {diff_r:.3e}")
    assert diff_r < eps_rot, f"rotation invariance violated: {diff_r:.3e} >= {eps_rot:.0e}"

    diff_p = (call(-batch["x"]) + out).abs().max().item()
    print(f"parity      max |δτ(-x) + δτ(x)| : {diff_p:.3e}")
    assert diff_p <= eps_exact, f"parity equivariance violated: {diff_p:.3e} > {eps_exact:.0e}"

    n_params = sum(p.numel() for p in model.parameters())
    print(f"total params: {n_params:,}")


if __name__ == "__main__":
    run_checks(pt.float32)
    run_checks(pt.float64)
