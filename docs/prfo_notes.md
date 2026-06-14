# P-RFO Notes

A conceptual reference for partitioned rational-function optimization as
implemented in `src/chemdm/prfo.py`. Captures the why, not just the what — in
particular why BFGS fails for saddle search, where the "shift" in P-RFO comes
from, and why this is a saddle *finder* and not a reaction *path* method.

---

## 1. The problem

Find a first-order saddle point of a potential energy surface `E(x)` using
**only forces** `−∇E`, no analytic Hessian. The first-order saddle has

```
∇E = 0          (stationary point)
H = ∇²E         has exactly one negative eigenvalue
```

Two challenges: (a) the gradient alone is zero at every minimum *and* every
saddle, so gradient-following can't distinguish them, and (b) the Hessian is
expensive (the very thing we want to avoid computing).

---

## 2. Why BFGS specifically fails

The BFGS quasi-Newton update

```
H_{k+1} = H_k + (yyᵀ)/(yᵀs) − (H_k s sᵀ H_k)/(sᵀ H_k s)
```

with `s = x_{k+1} − x_k` and `y = g_{k+1} − g_k` **provably preserves positive
definiteness** when the curvature condition `yᵀs > 0` holds. Line search
enforces this condition for minimization.

The consequence for saddle search is fatal: if `H₀ = I` (PD), every subsequent
`H_k` is PD by induction. The lowest eigenvalue can never cross zero. The
Newton step `−H_k⁻¹ g` always points downhill. **You converge to a minimum, no
matter how close you start to a saddle.**

This is not a corner case — it's a fundamental structural mismatch. BFGS was
designed for minimization and bakes that intent into its update rule.

The fix is to use a quasi-Newton update that **doesn't preserve PD**:

| Update | PD-preserving | Used for |
|---|---|---|
| BFGS | yes | minimization |
| SR1 (symmetric rank-1) | no | saddle search (unstable if `rᵀs` small) |
| PSB (Powell-symmetric-Broyden) | no | saddle search (stable, slower) |
| **Bofill** (SR1/PSB convex mix) | no | saddle search (best of both) |

`prfo.py` uses Bofill with weight `φ = (rᵀs)² / (‖r‖²‖s‖²)` (cosine² between
the residual `r = y − H s` and `s`): SR1-heavy when those vectors align,
PSB-heavy when they don't.

---

## 3. Newton with an indefinite Hessian — almost works

Suppose we *had* the true indefinite Hessian at a saddle. Decompose the Newton
step `Δx = −H⁻¹ g` in the eigenbasis:

```
Δx_i = −g_i / λ_i
```

For positive `λ_i`, `Δx_i` opposes `g_i` — descent. For *negative* `λ_i`,
`Δx_i` aligns with `g_i` — **ascent**. So Newton with the true Hessian
automatically climbs the unstable mode and descends the others. This is
eigenvector-following before we even introduce RFO.

But two problems remain:

1. **Step blows up near singular Hessians.** Any `λ_i ≈ 0` makes `|Δx_i| =
   |g_i / λ_i|` enormous. We need regularization.
2. **Wrong-mode following.** Early in the search, the "naturally" most negative
   eigenvalue may not be the one we actually want to ascend (multiple soft
   modes, eigenvalues crossing zero during the search). We need explicit
   *control* over which mode to follow.

RFO solves both at once.

---

## 4. The RFO step as shift-invert

Replace the quadratic energy model with a **rational function**:

```
ΔE(Δx) ≈  (gᵀ Δx + ½ Δxᵀ H Δx) / (1 + Δxᵀ Δx)
```

Stationary points of this rational function are solutions of the **augmented
eigenvalue problem**:

```
⎡ H  g ⎤ ⎡Δx⎤        ⎡Δx⎤
⎣ gᵀ 0 ⎦ ⎣ 1⎦  =  μ  ⎣ 1⎦
```

Expanding the first block row: `H Δx + g = μ Δx`, so

```
Δx = −(H − μI)⁻¹ g
```

**That's shift-invert.** The Hessian gets shifted by `μ`, inverted, applied to
`−g`. The same operator that appears in:

- **Levenberg-Marquardt** for nonlinear least squares (shift = damping `α`)
- **Trust-region Newton (Moré-Sorensen)** (shift chosen to satisfy
  `‖Δx‖ = trust_radius`)
- **Tikhonov regularization** for ill-posed problems
- **Inverse power iteration / shift-invert Lanczos** for eigenvalues

What makes RFO distinctive among shift-invert methods is the **rule for picking
`μ`**: it's an eigenvalue of the augmented `(N+1) × (N+1)` matrix. Equivalently,
substituting `Δx` into the second row gives the **secular equation**

```
∑_i  f_i² / (μ − λ_i)  =  μ
```

where `λ_i` are eigenvalues of `H` and `f_i = (eigvec_i)ᵀ g`. This scalar
equation has `N+1` roots, interleaved with the `λ_i`'s plus one below the
lowest and one above the highest eigenvalue.

**Each root is a valid shift; each gives a different step type:**

- **Lowest root** `μ < λ_min` → all `(λ_i − μ) > 0` → step descends in every
  mode → minimization RFO.
- **Largest root** `μ > λ_max` → all `(λ_i − μ) < 0` → step ascends in every
  mode → not usually what you want.
- **Intermediate roots** → mixed: ascends in some modes, descends in others.

The choice of root encodes the method's intent.

The rational denominator `1 + ‖Δx‖²` is what gives RFO its **automatic step
bounding**. As `‖g‖` grows or `H` becomes singular, `μ` grows in magnitude so
`(λ_i − μ)` stays bounded away from zero. The step `Δx` cannot blow up by
construction — no trust-region constraint required. (We still impose one as a
second safety net in `prfo.py`, used adaptively via the `ρ = ΔE_actual /
ΔE_predicted` ratio.)

---

## 5. The partitioning (the "P" in P-RFO)

For TS search we want one mode to ascend and the rest to descend. **Use two
different shifts**:

- **Followed (ascent) mode** `k`: solve the 2×2 RFO subproblem in mode `k`
  alone:
  ```
  ⎡ λ_k  f_k ⎤      ⎡v⎤
  ⎣ f_k   0  ⎦   =  ⎣1⎦ · μ⁺
  ```
  Take the **larger** root: `μ⁺ = ½(λ_k + √(λ_k² + 4f_k²)) > λ_k`. Then
  `Δq_k = −f_k / (λ_k − μ⁺)` ascends along mode `k`.

- **Descent subspace** (the other `N − 1` modes): solve the `M × M` RFO
  subproblem on the descent eigenvalues. Take the **lowest** root
  `μ⁻ < min_{j ≠ k} λ_j`. Then `Δq_j = −f_j / (λ_j − μ⁻)` descends in each
  remaining mode.

Reassemble `Δq` in the eigenbasis and rotate back to get `Δx`. **One ascent
mode, all-others descent.** That's the P in P-RFO.

The choice of which mode `k` to follow is the **mode-following heuristic**:
on iteration 1, pick the lowest eigenvalue (most negative if you have one,
otherwise softest). Subsequent iterations: pick the eigenvector with maximum
overlap with the previously-followed eigenvector. Failure mode: when the
followed eigenvalue crosses zero or near-degenerate eigenvalues swap, overlap
can pick the wrong mode and the optimizer briefly walks the wrong way. Our
HCN demo's iters 19–31 show this in action — the optimizer recovers, but it's
the main source of fragility in current `prfo.py`.

---

## 6. Coordinate systems are orthogonal

RFO is just linear algebra on `(H, g)`. It works in any coordinate system.
What changes between coordinate systems is the **plumbing around RFO**, not
RFO itself:

| Coords | Dim | Zero modes | Back-transform | Comments |
|---|---|---|---|---|
| Cartesian + trans/rot projection | `3N` | 6 (project out) | none | what we use today |
| Non-redundant internals | `3N − 6` | 0 | iterative Wilson B | smaller matrix, more code |
| Redundant internals (Pulay) | `M > 3N−6` | `M − (3N−6)` | iterative Wilson B + pseudoinverse | best step quality, most code |

The **three-axis design space** for any TS optimizer is:

```
Step generator    : {Newton, trust-region Newton, RFO/P-RFO, dimer rotation}
Hessian model     : {analytic, Bofill, SR1, PSB, BFGS (minimization only)}
Coordinate system : {Cartesian + projection, internal, redundant internal}
```

Most published "methods" are just specific points in this cube. Our `prfo.py`
is `{P-RFO, Bofill, Cartesian + projection}`. Gaussian's TS optimizer is
`{P-RFO, Bofill, redundant internals}`. Sella is roughly `{trust-region
Newton, exact Hessian via finite differences, Cartesian + projection}`. Dimer
codes are `{dimer rotation step, none — they don't store a Hessian, Cartesian}`.

These choices are independent — you can swap any axis without changing the
others.

---

## 7. P-RFO finds *points*, not *paths*

The optimizer's iterates trace a discrete trajectory through configuration
space, but **that trajectory has no chemical meaning.** It depends on initial
guess, Hessian model, trust radius, mode-following heuristic — none of which
are properties of the molecule. Two runs from different starts can take very
different optimizer paths to the same saddle.

The "eigenvector-following" name refers to following an *eigenvector at each
iteration*, not to following a *curve through configuration space*. These are
different objects.

The reaction path is a separate computation, defined intrinsically by the PES:

- **IRC** (Intrinsic Reaction Coordinate, Fukui): from the converged saddle,
  integrate `dx/dt = −g(x)` in mass-weighted Cartesians, ±along the unstable
  mode. One direction → reactant minimum, the other → product minimum. This
  curve is geometry-defined; different optimizers that find the same TS yield
  the same IRC.

This cleanly separates concerns:

| Step | Method | Object computed | Cost |
|---|---|---|---|
| 1 | P-RFO + Bofill | Saddle point (single geometry) | ~20–100 force calls |
| 2 | IRC (steepest descent ±) | Reaction path (a curve) | ~50–200 force calls |

Both are needed for a complete reaction characterization. P-RFO alone gives
you a saddle that might connect anywhere; IRC without a converged TS has no
starting point.

---

## 8. A note on gradient extremals (and why we don't build them)

**Gradient extremal curves** (Hoffman, Nord, Ruedenberg) are an alternative
class of methods that *do* try to trace a curve directly from the reactant
basin. A gradient extremal is the locus of points where `∇E` is parallel to a
specific Hessian eigenvector. There's a connection: the discrete trajectory of
P-RFO with mode-following is approximately a gradient extremal of the
followed mode. So in a loose sense, P-RFO is discretely walking *along* a
gradient extremal toward its endpoint at the saddle.

Gradient extremal *following* methods (continuous ODE integration of the curve,
adaptive parametrization, branch tracking) make the *curve itself* the
object of interest — you walk from the reactant minimum along the curve,
through any saddles you encounter, and into the product basin. This is elegant
and chemically motivated, but practically harder than the "find TS then IRC"
approach for several reasons:

- Gradient extremals branch and recombine — choosing the right branch at every
  bifurcation requires bookkeeping that IRC doesn't.
- They can leave the chemically relevant region of the PES, especially in
  high-dimensional systems.
- They don't necessarily pass through every saddle, and a saddle they don't
  pass through is invisible to the method.
- Numerically: ODE integration with branch tracking is heavier than
  "find a stationary point, then steepest-descent."

**Our practical choice is the simpler decomposition:**

```
relaxed reactant  →  estimate_lowest_mode  →  perturb + P-RFO  →  IRC
                     (Lindh + Lanczos)         (find TS)         (trace path)
```

The IRC step is cheap enough (gradient descent, ~150 lines of code) that we
don't gain anything from trying to construct the reaction path directly during
the saddle search. The gradient-extremal connection is a real mathematical
fact, but it doesn't pay off in code.

---

## 9. Summary

P-RFO is **shifted Newton with self-determined shifts**, where the shifts come
from an augmented eigenvalue problem and partition configuration space into
one ascent mode and a descent subspace. The Hessian is approximated by Bofill
(an indefinite-capable quasi-Newton update — BFGS does not work here). The
step is implicitly bounded by the rational-function model's denominator; we
add an explicit trust radius as a second safety net with adaptive shrink/grow.

This finds a single saddle point per run. To trace the reaction path that
connects the saddle to its endpoints, run IRC from the converged TS — a
separate, cheap, well-defined computation.

For posterity: the three-axis design space (step generator × Hessian model ×
coordinate system) is what makes the TS-optimization literature navigable.
Once you see it that way, "method X vs method Y" is mostly a question of
which point in the cube each method occupies.

---

## References

The reading list from earlier in the conversation, condensed:

- Schlegel, *WIREs Comput. Mol. Sci.* **1**, 790 (2011) — best tutorial review.
- Banerjee, Adams, Simons, Shepard, *J. Phys. Chem.* **89**, 52 (1985) — the
  original P-RFO paper.
- Baker, *J. Comput. Chem.* **7**, 385 (1986) — the practical recipe (mode
  following by max overlap) most QC codes follow.
- Bofill, *J. Comput. Chem.* **15**, 1 (1994) — the Bofill update.
- Besalú & Bofill, *Theor. Chem. Acc.* **100**, 265 (1998) — proper
  trust-radius treatment for RFO.
- Wales, *Energy Landscapes* (Cambridge, 2003), ch. 6–7 — the wide-angle view
  connecting RFO to dimer, GAD, and gradient extremals.
- Hoffman, Nord, Ruedenberg, *Theor. Chim. Acta* **69**, 265 (1986) — gradient
  extremal curves (for reference; we don't implement these).
