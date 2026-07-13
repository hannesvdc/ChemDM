# Torsional-Diffusion Conformer Generation — Initial Evaluation Report

**Date:** 2026-07-12  ·  **Source:** `examples/torsional_diffusion/eval_results/report_1000.md`

## TL;DR

Pipeline: **RDKit ETKDG backbones → torsional-diffusion torsion sampling → GFN2-xTB
relaxation**, with a **flat conformer budget per molecule** (the production setting —
no CREST reference is available at deploy time to size the sample). Evaluated against
CREST on 100 molecules per split.

**After relaxation, recall is at CREST parity across train/val/test.** The method
recovers essentially the full reference ensemble; the only remaining daylight is
tight-threshold *precision*. This makes the flat-budget configuration viable for
production.

## Setup

- **Runs:** `crest` (reference / oracle), `rdkit_t1` (method under test).
- **Level of theory:** GFN2-xTB, L-BFGS relaxation, force tolerance 0.02 eV/Å.
- **Scope:** 100 molecules per split (train / val / test).
- **Sampling:** flat per-molecule conformer budget (production mode). Contrast with a
  CREST-sized budget `K = 2·n_crest`, which requires knowing the answer in advance and
  is therefore evaluation-only.
- **Metrics:** AMR-R / AMR-P (mean min heavy-atom RMSD, recall / precision, Å);
  Cov-R / Cov-P at δ ∈ {0.5, 0.75, 1.0, 1.25} Å; `relax_shift` (raw→relaxed RMSD, Å);
  `init_force` (eV/Å); `n_iter` (L-BFGS steps); `prune_factor` (raw samples per retained
  conformer); `frac_converged`. Gap = method − crest.

## Headline: post-relaxation recall parity

Relaxed (GFN2-xTB) recall, per split:

| split | AMR-R crest | AMR-R rdkit_t1 | **gap** | Cov-R@0.5 gap | Cov-R@0.75 gap |
| --- | --- | --- | --- | --- | --- |
| train | 0.027 | 0.045 | **+0.018** | −0.011 | −0.000 |
| val   | 0.020 | 0.035 | **+0.015** | −0.008 | +0.000 |
| test  | 0.020 | 0.029 | **+0.009** | −0.005 | +0.000 |

Relaxed AMR-R is within 0.01–0.02 Å of CREST, and Cov-R ≥ 0.99 at every threshold
(including the tight 0.5 Å). The method lands in the same basins CREST finds — it is
recovering the reference ensemble, not a subset.

## Relaxation is the equalizer

Raw (pre-relaxation) geometry looks far behind CREST, but xTB relaxation collapses
almost the entire recall gap:

| split | raw AMR-R gap | relaxed AMR-R gap |
| --- | --- | --- |
| train | +0.087 | **+0.018** |
| val   | +0.096 | **+0.015** |
| test  | +0.086 | **+0.009** |

**Mechanism.** `relax_shift` for the method is ~0.19 Å vs ~0.04 Å for CREST, and
`init_force` is ~4.0 vs ~0.2–0.5 eV/Å. The raw conformers start far from the xTB
minimum and move a lot — but they move *into* the right place (tiny relaxed AMR). This
is the expected fingerprint of torsional diffusion: it sets the **torsions** correctly
(right basin) while the **local geometry** (bond lengths/angles) is ETKDG-limited, and
xTB cleanly repairs the latter. Torsion sampling is doing its job; relaxation is
mandatory, not optional.

## Residual gap: tight-threshold precision

The one axis that still lags after relaxation:

| split | AMR-P crest | AMR-P rdkit_t1 | gap | Cov-P@0.5 gap |
| --- | --- | --- | --- | --- |
| train | 0.087 | 0.139 | +0.052 | −0.054 |
| val   | 0.042 | 0.111 | +0.069 | −0.055 |
| test  | 0.047 | 0.087 | +0.039 | −0.040 |

At 0.75 Å precision coverage is already ≥ 0.97, so nothing is wildly off — the method
**over-generates** modestly: ~7–12% of its conformers sit > 0.5 Å from any CREST
minimum even after relaxation. Whether those are genuine extra minima CREST missed or
near-duplicates is the open question for the next iteration.

## Cost and validity

| metric | crest | rdkit_t1 | note |
| --- | --- | --- | --- |
| init_force (eV/Å) | 0.16–0.51 | 3.85–4.01 | ETKDG raw geometry is ~10–20× further from the minimum |
| n_iter (L-BFGS)   | 13–17 | 59–61 | ~4× more relaxation steps per conformer |
| prune_factor      | 81–94 | 22–24 | CREST samples are far more redundant; method keeps more distinct raw structures |
| frac_converged    | 1.000 | 1.000 | every conformer relaxes to a valid minimum |

The cost premium (~4× relaxation steps) is the price of classical ETKDG raw geometry
versus CREST's already-semiempirical structures. All conformers converge.

## Production implications

- **Flat budget works.** Fixing a per-molecule conformer count — the only option in
  production, where `n_crest` is unknown — still reaches CREST-parity relaxed recall.
- **Relaxation is part of the method, not post-processing.** Raw output is ETKDG-limited;
  the GFN2-xTB step is what delivers the accuracy and must ship with the pipeline.
- **Next lever is precision.** The residual `Cov-P@0.5` gap (~0.88–0.93) is the target.
  One concrete, principled contributor: symmetry-aware clustering that merges
  **conformational enantiomers** of achiral molecules (reflection-gated on achirality —
  see `docs/chirality_notes/`). These currently count as precision "misses" and inflate
  the over-generation.

## Caveats

- Single method run (`rdkit_t1`) against the `crest` reference; no `rdkit_t2` this round.
- Cross-report comparisons are not clean: the evaluation molecule set and the CREST
  reference both change between runs, so only **within-report gaps vs CREST** are a fair
  lens. This document reports the current run's standing, not a method-level delta over
  prior runs.
