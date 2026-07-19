# ChemDM — Infrastructure & Setup

Operational reference for setting up ChemDM on a new machine or the Linux
production server. For the *why* behind the environment design (the OpenMP
packaging story, the tblite migration, benchmarks), see
[`docs/xtb_tblite_notes/xtb_tblite_notes.pdf`](docs/xtb_tblite_notes/xtb_tblite_notes.pdf).

---

## TL;DR

One command (detects your conda tool, creates the env, installs the package,
validates torch+tblite):
```bash
./install_env.sh        # CHEMDM_RECREATE=1 to rebuild an existing env
conda activate chemdm
pytest                  # expect all pass
```
Equivalently, by hand:
```bash
conda env create -f environment.yml     # all conda-forge, single OpenMP
conda activate chemdm
pip install -e . --no-deps
```

That's the whole install. The rest of this document explains the two-environment
layout, the hard rules that keep it working, and the Linux/cluster specifics.

---

## Two environments (by necessity)

| | **`chemdm`** — production & dev | **verification** (e.g. `py311`) |
|---|---|---|
| Purpose | run CLI commands, tests, the ML pipeline | run the `examples/xtb_gpu/` comparison scripts |
| Channels | **conda-forge only** | mixed: conda `xtb-python` + **pip** `tblite` |
| Backend | tblite 0.7.0 (conda-forge) | tblite 0.7.0 (pip) **and** xtb-python 22.1 (conda) |
| OpenMP | one shared `libomp` → torch+tblite coexist | two `libomp` → **subprocess-isolated** in the harness |
| Reproducible via | `environment.yml` | ad-hoc (dev only) |

**Why two?** `xtb-python` (frozen at 22.1) needs the `xtb` program, which pins
`tblite 0.6.x` / `dftd4 3.7.x` — incompatible with the `tblite 0.7.0` / `dftd4 4.x`
production uses. They cannot be conda-solved together (on any Python). Production
doesn't need `xtb-python`; only the comparison scripts (which compare tblite
*against* the deprecated reference) do. So production stays clean and verification
lives in a separate mixed env.

---

## The hard rules (do not break these)

1. **Everything OpenMP-linked comes from conda-forge, nothing from `defaults` or
   the `pytorch` channel.** Mixing channels loads two copies of `libomp` → the
   process aborts at runtime (`OMP: Error #15`). `environment.yml` enforces this
   with `channels: [conda-forge, nodefaults]`.
2. **Install the package with `pip install -e . --no-deps`.** `--no-deps` prevents
   pip from pulling a second (pip-wheel) copy of torch/numpy/etc. that would
   reintroduce the OpenMP clash. All deps are already satisfied by conda-forge.
3. **`xtb-python` never goes in the `chemdm` env** (it can't solve with tblite
   0.7.0, and production doesn't need it).

---

## Production / development env: `chemdm`

Created from `environment.yml` (repo root). Resolved contents:

| package | version | role |
|---|---|---|
| python | 3.11 | interpreter |
| pytorch | 2.12.1 | ML models (EquivariantTransformer), LBFGS relaxation |
| tblite-python | 0.7.0 | **production xTB backend** (GFN1/GFN2), via `TBLitePotential` |
| dftd4 | 4.x (lib) | D4 dispersion, linked by tblite for GFN2 |
| openmm | 8.5.2 | MD / system utilities |
| numpy | 2.4.6 | arrays |
| scipy | 1.17.1 | numerics |
| rdkit | 2026.3.4 | cheminformatics, conformers |
| ase | 3.29.0 | atomistic interface (drives tblite) |
| e3nn | 0.6.0 | equivariant NN layers |
| matplotlib | 3.11.0 | plotting |
| python-dotenv | 1.2.2 | `.env` config loading |
| pytest | 9.1.1 | tests |
| chemdm | 0.0.1 | this package (editable, `pip install -e .`) |

**Validate after install:**
```bash
pytest                                  # full suite; expect all pass
python -c "import torch, numpy as np; \
  from chemdm.TBLitePotential import TBLitePotential; \
  print(TBLitePotential(Z=np.array([1,1])).energy_forces(np.array([[0,0,0],[0,0,.74]]))[0])"
# ^ torch + tblite in one process; prints ~-26.7 eV, no abort => env is correct
```

---

## Linux production cluster (CPU-only)

The cluster has conda and is **CPU-only**, so the install is identical to a dev
machine — no GPU/CUDA handling, same `environment.yml`:
```bash
# on the cluster, in the repo checkout:
module load conda        # or `module load miniforge`, if the site uses modules
./install_env.sh         # or: conda env create -f environment.yml && conda run -n chemdm pip install -e . --no-deps
```
Then point that host's ReactionStudio at `.../envs/chemdm/bin/chemdm`.

- **dxtb.** On Linux, dxtb GFN2 works (Linux-only `tad-libcint`), so the
  *comparison* scripts can run the full three-way (`RUN_DXTB` auto-enables). It is
  a verification-only dependency, not part of the production env.
- **If a GPU node is ever added** (not the case today): install a **conda-forge**
  CUDA torch build (`pytorch=2.12.1=*cuda*`), never a pip/pytorch-channel wheel,
  or two `libomp` copies collide.

---

## Verification env (comparison scripts only)

Needed only to run `examples/xtb_gpu/check_correspondence.py` and
`reference_gradient_bug.py`, which compare tblite against the deprecated
`xtb-python` reference (and dxtb on Linux). This is a **mixed** env — conda
`xtb-python` next to **pip** `tblite` (pip bypasses conda's solver so both
coexist on disk); the harness isolates each backend in its own `subprocess.run`
worker to dodge the OpenMP clash.

```bash
conda create -n chemdm-verify -c conda-forge python=3.11 xtb-python ase numpy rdkit
conda activate chemdm-verify
pip install tblite            # pip build (coexists with conda xtb-python on disk)
pip install dxtb tad-libcint  # Linux only, for the 3-way comparison
pip install -e . --no-deps
```
Run the comparison scripts here, **not** in `chemdm`. (The existing `py311` env
already serves this role.)

---

## Application integration (ReactionStudio)

The ReactionStudio app shells out to the ChemDM CLI, so it must point at the
**`chemdm` env's executable**, not a system/base Python:

```
/opt/homebrew/anaconda3/envs/chemdm/bin/chemdm      # (or .../bin/python -m chemdm)
```

Validated locally: **conformer generation** and **transition-path calculation**
both run through the app against this env. On the Linux server, point the app at
that host's `.../envs/chemdm/bin/chemdm`.

## Handy paths & commands

```bash
conda activate chemdm                          # production / dev
chemdm --help                                  # CLI entry point (chemdm.cli:main)
python -m pytest                               # tests

# regenerate the environment spec after adding a conda-forge package:
#   1) conda install -n chemdm -c conda-forge <pkg>
#   2) add "<pkg>=<version>" to environment.yml (conda-forge only!)
```

**Reproducibility:** `environment.yml` pins versions but not build strings, so it
re-solves per platform (macOS dev ↔ linux-64 prod) at the same versions. For
byte-identical builds across machines, generate a `conda-lock` file from it.
