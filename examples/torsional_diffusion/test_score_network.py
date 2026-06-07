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

from chemdm.MoleculeGraph import MoleculeGraph, batchMolecules


def build_toy_batch(dtype: pt.dtype) -> dict:
    """
    Two tiny molecules of 8 atoms each, fully-connected attention graph
    within each molecule, 2 rotatable bonds per molecule. Each molecule is
    given a small set of covalent bonds so `is_bond` is a non-trivial mix.
    """
    pt.manual_seed(0)

    N1, N2 = 8, 8

    Z1 = pt.randint( 1, 10, (N1,) )
    Z2 = pt.randint( 1, 10, (N2,) )
    x1 = pt.randn( N1, 3, dtype=dtype )
    x2 = pt.randn( N2, 3, dtype=dtype )

    # Arbitrary covalent bonds (both directions). Only their identity matters
    # for the test — they exercise the is_bond code path in the attention.
    def both_dirs(pairs):
        out = []
        for u, v in pairs:
            out += [[u, v], [v, u]]
        return pt.tensor(out, dtype=pt.long)
    cov1 = both_dirs( [(0, 1), (1, 2), (2, 3)] )
    cov2 = both_dirs( [(0, 1), (1, 2), (2, 3)] )

    mol = batchMolecules([
        MoleculeGraph( Z=Z1, x=x1, bonds=cov1 ),
        MoleculeGraph( Z=Z2, x=x2, bonds=cov2 ),
    ])

    t = pt.tensor([0.2, 0.8], dtype=dtype)   # diffusion time in [0, 1]

    def fully_connected(offset: int, n: int) -> pt.Tensor:
        idx = pt.arange(n) + offset
        src = idx.repeat_interleave(n)
        dst = idx.repeat(n)
        keep = src != dst
        return pt.stack([src[keep], dst[keep]], dim=0)

    neighbors = pt.cat([fully_connected(0, N1), fully_connected(N1, N2)], dim=1)
    # Mark a few edges as covalent bonds; the rest are distance-only. The
    # specific assignment doesn't matter for the equivariance probes, but
    # using a non-trivial pattern catches accidental dependence on is_bond's
    # value (e.g. if it were used as a position-dependent quantity).
    is_bond = (pt.rand( neighbors.shape[1] ) < 0.3).to(dtype)

    bonds      = pt.tensor([[0, 1], [2, 3], [N1 + 0, N1 + 1], [N1 + 2, N1 + 3]])
    bond_batch = pt.tensor([0, 0, 1, 1])

    return dict(
        mol=mol, t=t,
        neighbors=neighbors, is_bond=is_bond,
        bonds=bonds, bond_batch=bond_batch,
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
        # Equivariance probes vary x; rebuild the molecule with the new positions.
        mol_at_x = batch["mol"].copyWithNewPositions(x)
        return model(
            mol=mol_at_x, t=batch["t"],
            neighbors=batch["neighbors"], is_bond=batch["is_bond"],
            bonds=batch["bonds"], bond_batch=batch["bond_batch"],
        )

    out = call(batch["mol"].x)
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

    x0 = batch["mol"].x

    t = pt.tensor([0.7, -1.3, 0.4], dtype=dtype)
    diff_t = (call(x0 + t) - out).abs().max().item()
    print(f"translation max |Δ| : {diff_t:.3e}")
    assert diff_t < eps_trans, f"translation invariance violated: {diff_t:.3e} >= {eps_trans:.0e}"

    R = random_rotation(dtype)
    diff_r = (call(x0 @ R.T) - out).abs().max().item()
    print(f"rotation    max |Δ| : {diff_r:.3e}")
    assert diff_r < eps_rot, f"rotation invariance violated: {diff_r:.3e} >= {eps_rot:.0e}"

    diff_p = (call(-x0) + out).abs().max().item()
    print(f"parity      max |δτ(-x) + δτ(x)| : {diff_p:.3e}")
    assert diff_p <= eps_exact, f"parity equivariance violated: {diff_p:.3e} > {eps_exact:.0e}"

    n_params = sum(p.numel() for p in model.parameters())
    print(f"total params: {n_params:,}")


if __name__ == "__main__":
    run_checks(pt.float32)
    run_checks(pt.float64)
