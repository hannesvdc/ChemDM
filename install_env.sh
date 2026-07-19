#!/usr/bin/env bash
#
# ChemDM environment installer.
#
# Creates the `chemdm` conda environment from environment.yml (all conda-forge,
# single libomp -> torch + tblite coexist), installs the package, and validates.
#
# The env is CPU-based and identical on macOS and linux-64 (conda re-solves the
# same spec per platform), so there is nothing machine-specific to choose. The
# script just:
#   1. finds conda (or mamba),
#   2. conda env create -f environment.yml,
#   3. pip install -e . --no-deps,
#   4. validates torch + tblite coexistence.
#
# Usage:  ./install_env.sh
# Overrides:
#   CHEMDM_ENV_NAME=name   env name (default: chemdm)
#   CHEMDM_RECREATE=1      remove an existing env of that name first
#
# (No GPU handling: the production server is CPU-only and macOS has no CUDA. If a
#  GPU node is ever added, install a conda-forge CUDA torch build --
#  `pytorch=2.12.1=*cuda*` -- never a pip/pytorch-channel wheel, or two libomp
#  copies collide; see SETUP.md.)
set -euo pipefail

ENV_NAME="${CHEMDM_ENV_NAME:-chemdm}"
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
YML="$REPO_DIR/environment.yml"
[ -f "$YML" ] || { echo "[install] ERROR: $YML not found"; exit 1; }

echo "[install] $(uname -s)/$(uname -m)  env=$ENV_NAME"

# --- 1. find conda (or mamba) ------------------------------------------------
CONDA=""
for c in mamba conda; do
    if command -v "$c" >/dev/null 2>&1; then CONDA="$c"; break; fi
done
if [ -z "$CONDA" ]; then
    echo "[install] ERROR: no conda/mamba on PATH."
    echo "          Install Miniforge (https://github.com/conda-forge/miniforge)"
    echo "          or 'module load' the cluster's conda, then re-run."
    exit 1
fi
echo "[install] conda tool: $CONDA"

# --- 2. create the env from environment.yml (channels: conda-forge only) -----
if "$CONDA" env list 2>/dev/null | grep -qiE "(^|/)$ENV_NAME([[:space:]]|/|\$)"; then
    if [ "${CHEMDM_RECREATE:-0}" = "1" ]; then
        echo "[install] removing existing env '$ENV_NAME'"
        "$CONDA" env remove -y -n "$ENV_NAME"
    else
        echo "[install] ERROR: env '$ENV_NAME' exists (set CHEMDM_RECREATE=1 to replace)"; exit 1
    fi
fi
echo "[install] creating '$ENV_NAME' from $YML ..."
"$CONDA" env create -f "$YML" -n "$ENV_NAME"

# --- 3. install the chemdm package (editable, no deps) -----------------------
echo "[install] pip install -e . --no-deps"
"$CONDA" run -n "$ENV_NAME" pip install -e "$REPO_DIR" --no-deps

# --- 4. validate torch + tblite coexistence ----------------------------------
echo "[install] validating (torch + tblite in one process)..."
"$CONDA" run -n "$ENV_NAME" python - <<'PY'
import numpy as np, torch  # noqa: F401
from chemdm.TBLitePotential import TBLitePotential
e, _ = TBLitePotential(Z=np.array([1, 1])).energy_forces(np.array([[0, 0, 0], [0, 0, 0.74]]))
assert abs(e - (-26.72)) < 0.5, f"unexpected H2 GFN2 energy {e}"
print(f"[install] OK: torch+tblite coexist; H2 GFN2 E={e:.4f} eV")
PY

echo "[install] DONE. Activate:  $CONDA activate $ENV_NAME   (then: pytest)"
