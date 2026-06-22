from __future__ import annotations

from typing import Protocol

import numpy as np

# Evaluator interface
class EnergyForceEvaluator(Protocol):
    """
    Protocol for duck-typing energy and force evaluators.

    The only convention is that positions must be passed in Angstrom,
    and the returned energy has units of eV. The forces have 
    units eV / A.
    """
    # `x` is declared positional-only (the `/`) so that implementers may name
    # the parameter whatever they like (e.g. XTBPotential uses `x_A`).
    def energy_forces(self, x: np.ndarray, /) -> tuple[float, np.ndarray]:
        ...