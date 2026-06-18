"""
Gentlest Ascent Dynamics (GAD) — single-ended walker for transition-state search.

Discrete-time approximation of the continuous GAD flow

    dx/dt = -∇V(x) + 2 (∇V · u) u
    du/dt = -(I - u uᵀ) H u,        ||u|| = 1

whose attracting fixed points are first-order saddles of V. The gradient
along the unstable direction `u` is *reversed* at every step (ascent), while
the gradient orthogonal to `u` is preserved (descent). Starting from a
slightly displaced minimum, the flow walks gently uphill along the softest
mode while staying in the floor of the orthogonal valley, until the geometry
enters the basin of attraction of a saddle — i.e. the true Hessian acquires
a negative eigenvalue along `u`.

Designed to be the *first stage* of a two-stage TS finder; once GAD reports
``transitioned=True`` (lam_min has stayed below threshold for a few
consecutive steps), hand off to :class:`chemdm.prfo.PRFOOptimizer` for fast
refinement.

References
----------
W. E and X. Zhou, Nonlinearity 24, 1831 (2011).
W. Gao, J. Leng and X. Zhou, J. Chem. Phys. 142, 154109 (2015).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch as pt

from chemdm.MoleculeGraph import Molecule
from chemdm.prfo import (
    EnergyForceEvaluator,
    estimate_lowest_mode,
    _project_vec,
    _to_numpy,
    _trans_rot_basis,
)


class GentlestAscentDynamics:
    """
    Walk gently away from a local minimum along the softest mode until the
    true Hessian acquires a negative eigenvalue along that direction — i.e.
    we leave the minimum's basin and enter a saddle's basin.

    Each step

        u, lam_min = estimate_lowest_mode(...)         # mass-weighted Lanczos against true H
        g_mod = g - 2 (g · u) u                        # u-component of g reversed
        dx = - α g_mod,  capped at `max_step`          # explicit Euler

    is exactly the GAD ODE discretized. The Lanczos call from the previous
    step's `u` is reused as a seed, so mode-following is continuous.

    Parameters
    ----------
    evaluator
        Same `energy_forces(x) -> (E, F)` protocol as :class:`PRFOOptimizer`.
    molecule : chemdm.MoleculeGraph.Molecule
        Starting geometry. Should be *slightly* displaced from the relaxed
        minimum — at the exact minimum ``g = 0`` and GAD makes no progress.
        Use a Lindh-lowest-mode kick of e.g. 0.05-0.1 Å (much smaller than
        P-RFO's kick) to give GAD a starting gradient.
    step_alpha : float, default 0.1
        Multiplier on the modified-gradient step:  ``dx = -step_alpha * g_mod``.
        Smaller values walk more slowly but more accurately.
    max_step : float, default 0.05
        Hard upper bound on ``||dx||`` per step. Mirrors P-RFO's trust radius.
    lam_threshold : float, default -0.05
        Walk is "in a saddle's basin" once the followed mode's curvature stays
        below this for `stable_for` consecutive steps. eV/Å² units, matching
        :func:`estimate_lowest_mode`'s reported Rayleigh quotient.
    stable_for : int, default 3
        Required consecutive-step count under the threshold.
    lanczos_max_iter : int, default 15
        Forwarded to :func:`estimate_lowest_mode`.
    """

    def __init__(self,
                 evaluator: EnergyForceEvaluator,
                 molecule: Molecule, *,
                 step_alpha: float = 0.1,
                 max_step: float = 0.05,
                 lam_threshold: float = -0.05,
                 stable_for: int = 3,
                 lanczos_max_iter: int = 15):
        if not isinstance(molecule, Molecule):
            raise TypeError(
                f"GentlestAscentDynamics requires a chemdm.MoleculeGraph.Molecule; "
                f"got {type(molecule).__name__}."
            )
        self.evaluator = evaluator
        self.molecule = molecule
        x0 = _to_numpy(molecule.x).astype(float)
        self._shape = x0.shape
        self.x = x0.flatten().copy()
        self.dim = self.x.size

        self.step_alpha = float(step_alpha)
        self.max_step = float(max_step)
        self.lam_threshold = float(lam_threshold)
        self.stable_for = int(stable_for)
        self.lanczos_max_iter = int(lanczos_max_iter)

        # Mode-following seed reused across steps so Lanczos converges fast and
        # the chosen direction evolves smoothly rather than jumping each iter.
        self._u: Optional[np.ndarray] = None
        self._consecutive_unstable = 0
        self.history: list[dict] = []

    def step( self ) -> dict:
        E, F = self.evaluator.energy_forces( self.x.reshape(self._shape) )
        g = -np.asarray(F, dtype=float).reshape(-1)

        # Project trans/rot out of g — they're not meaningful directions.
        V = _trans_rot_basis( self.x.reshape(self._shape) )
        g_proj = _project_vec(g, V)
        g_norm = float(np.linalg.norm(g_proj))

        # True-Hessian lowest mode at current geometry.
        mol_now = self.molecule.copyWithNewPositions(
            pt.tensor(self.x.reshape(self._shape), dtype=pt.float64)
        )
        u, lam = estimate_lowest_mode(
            self.evaluator, mol_now,
            init_u=self._u,
            max_iter=self.lanczos_max_iter,
            eps=1e-3, tol=1e-3,
        )
        self._u = u.copy()

        # Modified gradient: flip the u-component. Ascend along u, descend ⊥.
        g_dot_u = float(u @ g_proj)
        g_mod = g_proj - 2.0 * g_dot_u * u

        # Euler step, capped at max_step.
        dx = -self.step_alpha * g_mod
        step_norm = float(np.linalg.norm(dx))
        if step_norm > self.max_step:
            dx *= self.max_step / step_norm
            step_norm = self.max_step

        # Transition detector.
        if lam < self.lam_threshold:
            self._consecutive_unstable += 1
        else:
            self._consecutive_unstable = 0
        transitioned = self._consecutive_unstable >= self.stable_for

        info = dict(
            energy             = float(E),
            grad_norm          = g_norm,
            step_norm          = step_norm,
            lam_min            = float(lam),
            g_dot_u            = g_dot_u,
            consecutive_unstable = self._consecutive_unstable,
            transitioned       = transitioned,
        )
        self.history.append(info)

        self.x = self.x + dx
        return info

    def run( self, max_iter: int = 200, verbose: bool = False ) -> dict:
        """
        Walk until `transitioned=True` or `max_iter` is reached. Returns a
        summary dict; the caller can hand `result['x']` (wrapped back into a
        Molecule) directly to :class:`PRFOOptimizer`.
        """
        for it in range(max_iter):
            info = self.step()
            if verbose:
                print( f"[{it:4d}] E={info['energy']:+.6f}  "
                       f"|g|={info['grad_norm']:.3e}  "
                       f"|dx|={info['step_norm']:.3e}  "
                       f"lam_min={info['lam_min']:+.4f}  "
                       f"unstable={info['consecutive_unstable']}" )
            if info['transitioned']:
                return dict(
                    transitioned = True,
                    n_iter       = it + 1,
                    x            = self.x.reshape(self._shape).copy(),
                    energy       = info['energy'],
                    lam_min      = info['lam_min'],
                    u            = self._u.copy() if self._u is not None else None,
                )
        last = self.history[-1] if self.history else {}
        return dict(
            transitioned = False,
            n_iter       = max_iter,
            x            = self.x.reshape(self._shape).copy(),
            energy       = last.get('energy'),
            lam_min      = last.get('lam_min'),
            u            = self._u.copy() if self._u is not None else None,
        )
