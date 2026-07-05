# xtb on GPU via dxtb

Evaluating whether [`dxtb`](https://github.com/grimme-lab/dxtb) (grimme-lab's
fully-differentiable PyTorch implementation of xTB) can serve as a GPU-accelerated
replacement for the CPU `xtb`-python (tblite) calculator used elsewhere in ChemDM.

Two priorities:
1. **Correspondence** — does dxtb reproduce the production `XTBPotential` energies
   and forces?
2. **Speed** — at what batch size does dxtb (batched, on GPU) beat xtb-python
   evaluating molecules one at a time?

Everything here is self-contained; no existing ChemDM files are modified.

## Platform split (important)

|            | macOS (this repo's dev machine) | Linux cluster |
|------------|---------------------------------|---------------|
| GFN1-xTB   | ✅ pure-PyTorch backend         | ✅            |
| GFN2-xTB   | ❌ needs `tad-libcint` (Linux-only) | ✅        |
| GPU        | ❌ MPS only, and dxtb breaks on MPS (`torch.unique` device bug) | ✅ CUDA |

So the plan is two-pronged: validate **GFN1 on CPU (f64)** locally, then run
**GFN2 and the CUDA speed crossover on the cluster**. The scripts are
method/device-agnostic (constants at the top of each) so the same code covers
both.

## Install

**Local (macOS), GFN1 only:**
```
/opt/homebrew/anaconda3/envs/py311/bin/pip install dxtb
```

**Cluster (Linux + CUDA), GFN1 + GFN2:**
```
pip install torch --index-url https://download.pytorch.org/whl/cu121   # match your CUDA
pip install "dxtb[libcint]"
```
Install a CUDA-matched torch *first*, or pip may pull a CPU-only build. The
`[libcint]` extra (which enables GFN2) has Linux-only wheels and will fail on macOS.

## Files

| file | purpose |
|------|---------|
| `dxtb_potential.py` | `DxtbPotential` — drop-in analogue of `chemdm.xtbSetup.XTBPotential` (`energy_forces(x_A) -> (E_eV, F_eV_A)`), backed by dxtb. Configurable method/device/dtype; converts units with `ase.units` so comparisons measure only algorithmic differences. |
| `molecules.py` | 10-molecule test suite, H2 (2 atoms) → C60 alkane (182 atoms), elements H/C/N/O/F/S, built deterministically from SMILES via RDKit. |
| `check_correspondence.py` | dxtb vs xtb-python energies + forces, per molecule and aggregate. |
| `benchmark_speed.py` | 2-D sweep (molecule size × batch size B) of dxtb-batched vs xtb-python multiprocess; reports the crossover B per molecule. Defaults to GFN2 + CUDA for the cluster. |
| `reference_gradient_bug.py` | Three-way comparison (xtb-python analytical vs dxtb autograd vs finite differences) of the gradient discrepancy found below. |
| `ISSUE.md` | Draft post for the **dxtb** GitHub asking whether the xtb↔dxtb force discrepancy is a known incompatibility. |

Run any of them with the `py311` interpreter, e.g.:
```
/opt/homebrew/anaconda3/envs/py311/bin/python check_correspondence.py
```
No CLI flags — edit the constants at the top of each script (method, device, dtype,
batch sizes).

## Findings

### 1. Correspondence (GFN1, CPU, float64) — excellent

dxtb reproduces xtb-python across the full 2→182-atom suite:

| quantity | agreement |
|----------|-----------|
| absolute energy | mean ~2×10⁻⁶ Ha, worst 1.1×10⁻⁵ Ha (182 atoms) |
| forces | max error ≤ 1.8×10⁻⁵ eV/Å, cosine = 1.000000000 |

Float64 matters here: in float32 the ~10⁻⁷ relative roundoff would swamp the
genuine (algorithmic) differences. Correspondence needs no GPU — it runs on CPU.

### 2. A gradient bug in the *reference* (not dxtb)

At the symmetric equilibrium geometries of methanol and methanethiol, xtb-python's
**analytical force disagrees with the finite-difference gradient of its own
energy** by ~0.1–0.22 eV/Å on one carbon component. The gap is invariant to SCF
accuracy (1.0→1e-4), electronic temperature, and FD step size — so it is not a
convergence artifact. dxtb's autograd force matches the finite-difference gradient
to <10⁻⁶, i.e. dxtb is the self-consistent/correct one. See
`reference_gradient_bug.py` (three-way comparison) and `ISSUE.md` (draft dxtb post).

Practical implication: the production `XTBPotential` occasionally returns forces
that are not ∇E; dxtb would fix this.

### 3. Speed (local, CPU) — dxtb loses on CPU; batching only pays off on GPU

Four-way comparison for `aspirin` × B (energy+forces, GFN1), 8-core machine.
**B is the batch size** (number of molecules per call); the molecule is fixed at
21 atoms — only B is swept.

| B | xtb-mp (8 proc) | dxtb-mp (8 proc, non-batched) | xtb-seq (1 proc) | dxtb-cpu (batched) |
|---|-----------------|-------------------------------|------------------|--------------------|
| 1   | 9 ms   | 69 ms   | 9 ms    | 152 ms   |
| 8   | 22 ms  | 112 ms  | 70 ms   | 711 ms   |
| 32  | 87 ms  | 447 ms  | 280 ms  | 2154 ms  |
| 128 | 340 ms | 1710 ms | 1120 ms | 10044 ms |

- `xtb-mp` (spawned `ProcessPoolExecutor`, one `XTBPotential`/worker) is the pattern
  ChemDM runs now — the baseline to beat.
- **On CPU, parallelizing dxtb beats batching it by ~6×** (dxtb-mp 1710 ms vs
  dxtb-cpu-batched 10044 ms at B=128): batching is one big single-threaded autograd
  graph + a larger `eigh`, while process-parallel spreads small evals across cores.
  Batched dxtb-CPU per-molecule cost is roughly flat in B (~78–152 ms/mol) — batching
  amortizes fixed overhead but gives no CPU speedup.
- dxtb is still ~5× slower per molecule than compiled-Fortran xtb even when both are
  process-parallel (13 vs 2.7 ms/mol at B=128).

So **batching only pays off when the batched linear algebra runs massively parallel —
i.e. CUDA**. On CPU there is no configuration where dxtb wins. NB: the dxtb-mp workers
must have BLAS/LAPACK pinned to 1 thread each (env vars before torch import), else the
per-worker Accelerate/LAPACK threads oversubscribe the cores and it thrashes (~150×
slower). The benchmark handles this.

## Remaining work — on the cluster

1. **GFN2 correspondence**: set `METHOD="GFN2-xTB"` in `check_correspondence.py`
   and run on Linux (CPU or CUDA, f64). Same expectation as GFN1.
2. **GFN2 CUDA speed crossover**: `benchmark_speed.py` already defaults to
   `METHOD="GFN2-xTB"` and `DXTB_DEVICES=[("cuda", torch.double), ("cpu", ...)]`, and
   sweeps a 2-D grid of `MOLECULES` (size axis) × `BATCH_SIZES`. It prints, per
   molecule, the smallest batch size B at which dxtb-GPU-batched beats the xtb-python
   multiprocess baseline (`xtb-mp`, which uses all node cores), plus a speedup-vs-B
   plot. Push `BATCH_SIZES` higher to find the crossover (watch GPU memory). GFN2 uses
   libcint by default — not overridden. `RUN_XTB_SEQ` / `RUN_DXTB_MP` add context
   columns. **Because libcint integrals are CPU-bound and looped per molecule, expect
   the crossover only for larger molecules and/or large B — the whole point of the
   sweep is to find where (if anywhere) it lands.**
