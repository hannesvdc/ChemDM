"""
Discover Stable Conformer from 2D Molecular Graph: RDKIT initial guess + xTB-optim refinement.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_XTB_DIR = _REPO_ROOT / "examples" / "xtb"
if str(_XTB_DIR) not in sys.path:
    sys.path.insert(0, str(_XTB_DIR))


def run(input_data: dict) -> dict:
    """
    Empty implementation for now.
    """

    return input_data