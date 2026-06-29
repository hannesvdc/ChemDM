"""
Reverse-SDE sampler for torsional diffusion on the m-torus.

Given a trained score model and a starting molecule whose torsions we want to
re-sample, we:

    1. Initialise at the uniform prior — add a Uniform([0, 2π)) increment to
       each rotatable bond's torsion angle (equivalently, apply a random Δτ
       to x via `apply_torsion_update`).

    2. Run K reverse-SDE steps from t = 1 down to t = 0:

           Δτ_step = g²(t) · s_θ(x_t, t) · Δt   +   g(t) · √Δt · ε

       where g²(t) = dσ²/dt = 2 σ²(t) · log(σ_max/σ_min) for the geometric
       noise schedule σ(t) = σ_min^(1-t) σ_max^t, and ε ~ N(0, I_m).

    3. On the final step, the noise term is dropped — common practice for
       score-based sampling, removes the last bit of jitter.

The molecule's bond lengths, bond angles, and ring conformations are
invariant under every `apply_torsion_update`, so the sampled geometry stays
chemically sensible by construction. Only torsion angles change.

Batched sampling
----------------
The caller can pass a `BatchedMoleculeGraph` containing K independent copies
of the same molecule (or different molecules entirely). The sampler operates
on the flat batched layout — each copy gets its own random initialisation
and noise per step. This makes it cheap to draw many conformers in parallel.
"""

from __future__ import annotations

import math

import torch as pt

from chemdm.MoleculeGraph import Molecule, findAllNeighbors_gpu
from chemdm.geometry import apply_torsion_update

from score_network import TorsionalScoreNetwork


# Defaults match the noise schedule in `diffusion.py`.
SIGMA_MIN_DEFAULT = 0.01 * math.pi
SIGMA_MAX_DEFAULT = math.pi


@pt.no_grad()
def sample_torsional_diffusion( model: TorsionalScoreNetwork,
                                mol: Molecule,               # BatchedMoleculeGraph carrying Z, x_init, molecule_id
                                rotatable_bonds: pt.Tensor,              # (m_total, 2)   rotatable bond endpoints (global atom indices)
                                side_atom_idx: pt.Tensor,              # (P_total,)     atoms on the c-side
                                side_bond_idx: pt.Tensor,              # (P_total,)     bond index per side-atom
                                bond_batch: pt.Tensor,              # (m_total,)     molecule index per bond
                                *,
                                n_steps:       int   = 20,
                                sigma_min:     float = SIGMA_MIN_DEFAULT,
                                sigma_max:     float = SIGMA_MAX_DEFAULT,
                                cutoff:        float = 5.0,
                                dtype:         pt.dtype = pt.float32,
    ) -> pt.Tensor:
    """
    Run the reverse SDE and return the sampled atomic positions.

    Returns
    -------
    x_sampled : (N_total, 3) float — atomic positions after K reverse steps.
                Same dtype/device as `mol.x`.
    """
    device = mol.x.device

    B = int( bond_batch.max().item() ) + 1
    m = rotatable_bonds.shape[0]

    # Uniform-on-torus initialization. Adding Uniform([0, 2π)) to whatever
    # torsion the input molecule has gives Uniform([0, 2π)) mod 2π.
    delta_tau_init = pt.rand( m, device=device, dtype=dtype ) * (2.0 * math.pi)
    x = apply_torsion_update( mol.x, rotatable_bonds, side_atom_idx, side_bond_idx, delta_tau_init )

    # Reverse SDE loop. Steps go from t = 1 down to t = 0 in dt = 1/K
    # increments. For the geometric noise schedule, g²(t) = 2 σ²(t) log(σmax/σmin).
    log_sigma_ratio = math.log( sigma_max / sigma_min )
    dt = 1.0 / n_steps

    for k in range( n_steps, 0, -1 ):
        t_scalar = k * dt
        t_per_mol = pt.full( (B,), t_scalar, device=device, dtype=dtype )

        sigma = sigma_min ** (1.0 - t_scalar) * sigma_max ** t_scalar
        g_sq = 2.0 * (sigma ** 2) * log_sigma_ratio

        # Score from the model: rebuild union neighbor graph from current x.
        mol_t = mol.copyWithNewPositions(x)
        neighbors_E2, is_bond = findAllNeighbors_gpu( mol_t, cutoff )
        neighbors = neighbors_E2.T.contiguous()
        is_bond = is_bond.to( dtype=dtype )

        score = model( mol=mol_t, t=t_per_mol,
                       neighbors=neighbors, is_bond=is_bond,
                       rotatable_bonds=rotatable_bonds, bond_batch=bond_batch )

        # Reverse SDE step. Drop the noise on the last step (k == 1) for a
        # deterministic finish — standard score-based-sampling practice.
        drift = g_sq * score * dt
        if k > 1:
            noise = math.sqrt( g_sq * dt ) * pt.randn( m, device=device, dtype=dtype )
        else:
            noise = pt.zeros( m, device=device, dtype=dtype )
        delta_tau_step = drift + noise

        # Apply the torsions for the next step.
        x = apply_torsion_update( x, rotatable_bonds, side_atom_idx, side_bond_idx, delta_tau_step )

    return x
