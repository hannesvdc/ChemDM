# Distributional shift: pre-retrain baseline

**Date:** 2026-06-13
**Model:** current score network (pre-retrain)
**Eval set:** QM9 test split (~12k molecules)
**Ground truth:** CREST conformer ensemble

This document records the three-way comparison of the score model's behavior
under different starting-geometry distributions, taken just before retraining
on RDKit-augmented data.

## Setup

Three inference configurations were evaluated against the CREST ground truth:

1. **CREST start.** Score model is given a CREST conformer as starting
   geometry. No distributional shift — the inputs match the training
   distribution. This is the "ceiling" measurement.
2. **RDKit + MMFF94 start.** ETKDG-generated conformer relaxed with MMFF94,
   then handed to the score model. The realistic deployment setting.
3. **RDKit only.** ETKDG-generated conformer, no MMFF94 relaxation. The
   weakest preprocessing.

All three use the same trained score network. Differences between configurations
isolate the cost of distributional mismatch in the starting geometry.

## Numbers

| Metric                  | CREST start | RDKit + MMFF94 | RDKit only |
| ----------------------- | ----------- | -------------- | ---------- |
| Molecules               | 12,197      | 11,695         | 11,695     |
| **AMR-R** mean          | 0.317 Å     | 0.358 Å        | 0.393 Å    |
| **AMR-R** median        | 0.255 Å     | 0.306 Å        | 0.362 Å    |
| **AMR-P** mean          | **0.043 Å** | 0.194 Å        | 0.277 Å    |
| **AMR-P** median        | **0.014 Å** | 0.138 Å        | 0.226 Å    |
| Cov-R @ 0.5 Å mean      | 68.2 %      | 67.9 %         | 65.9 %     |
| Cov-R @ 0.5 Å median    | 80 %        | 80 %           | 75 %       |
| Cov-P @ 0.5 Å mean      | 98.0 %      | 90.6 %         | 84.5 %     |
| Cov-P @ 0.5 Å median    | 100 %       | 100 %          | 100 %      |
| Cov-R @ 0.75 Å mean     | 83.4 %      | 84.8 %         | 85.4 %     |
| Cov-R @ 1.00 Å mean     | 92.3 %      | 94.4 %         | 95.3 %     |
| Cov-P @ 0.75 Å mean     | 99.2 %      | 97.2 %         | 96.1 %     |

(AMR-R = recall direction: avg over CREST conformers of min RMSD to a generated
conformer. AMR-P = precision direction: avg over generated conformers of min
RMSD to a CREST conformer. Coverage: fraction within RMSD threshold δ.)

## Headline finding

**Distributional shift hits precision sharply but barely touches recall.**

- AMR-P degrades by ~10× from CREST start (0.014 Å) to MMFF start (0.138 Å)
  and ~16× to raw RDKit (0.226 Å).
- AMR-R degrades modestly: 0.255 → 0.306 → 0.362 Å (+20%, +42%).
- Coverage-R at 0.5 Å is essentially unchanged across all three modes (68.2 /
  67.9 / 65.9 mean). The model's *ability to span the CREST ensemble's modes*
  is preserved under shift; the *quality of alignment to those modes* is not.

## Interpretation

The most likely mechanism, consistent with the numbers:

The score model is operating as a **near-identity** on CREST inputs (AMR-P
median 0.014 Å — the conformers basically don't move) and as a **structurally
mismatched updater** on RDKit inputs. The torsional updates the model learned
are correct for the CREST manifold; applied to RDKit conformers, those updates
correctly rotate the dihedrals but leave the non-torsional differences (bond
lengths, bond angles, ring puckers) untouched. The generated conformer ends up
torsionally CREST-like but geometrically RDKit-like — close to a CREST mode in
torsional space but offset from it in bond-length / angle space. Hence sharp
precision degradation, preserved recall.

MMFF94 partially mitigates this by pulling RDKit conformers toward MMFF
minima, which themselves sit closer to CREST minima than raw ETKDG. The MMFF
column shows this mitigation closes about one-third of the AMR-P gap
(0.226 → 0.138, against a 0.014 ideal).

## Predicted retraining outcomes

After retraining with RDKit-augmented data, the diagnostic comparison is:

- **AMR-P, RDKit mode**: should move from 0.138 toward 0.014 (the CREST-mode
  ceiling). Large movement = shift is closed.
- **AMR-P, CREST mode**: should hold near 0.014. If it degrades meaningfully,
  augmentation taught the model a new joint distribution rather than closing
  the shift — different story, different implications.
- **AMR-R, both modes**: smaller expected movement; recall isn't the bottleneck
  here.
- **Coverage-R at 0.5 Å**: tracks AMR-R; modest gains expected.

The cleanest pre/post comparison: **AMR-P RDKit mode** is the single number
that summarizes how much of the shift the retraining closed.

## Open question worth resolving before reading too much into the retrain

Where, geometrically, does the AMR-P gap in RDKit mode actually live? The
hypothesis above is "bond lengths and angles, because the model only adjusts
torsions" — but it's a hypothesis, not a measurement.

Concretely: pick ~10 RDKit-mode molecules with high AMR-P, align the closest
generated conformer to the closest CREST conformer, and decompose the residual
RMSD into:

1. Bond-length contribution
2. Bond-angle contribution
3. Dihedral / torsional contribution

If (1) + (2) dominate, the interpretation above is confirmed and RDKit-augmented
retraining is well-targeted. If (3) dominates, the model is failing to find
the correct torsional minima from RDKit starts, and a different fix is needed
(e.g., more sampling steps, lower-temperature sampling, or training objective
changes).

This decomposition is cheap to run (no extra inference, just RMSD analysis on
existing matched pairs) and would significantly de-risk the interpretation of
retrained-model results.
