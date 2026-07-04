"""
DxtbPotential
=============

A drop-in analogue of ``chemdm.xtbSetup.XTBPotential`` backed by `dxtb`
(grimme-lab), the fully-differentiable PyTorch implementation of xTB.

The public contract is deliberately identical to ``XTBPotential`` so the two can
be swapped and compared directly:

    pot = DxtbPotential(Z, method="GFN1-xTB")
    energy_eV, forces_eV_A = pot.energy_forces(x_A)      # x_A in Angstrom

Unit convention (matching XTBPotential):
    - input positions ``x_A``: Angstrom, shape (n_atoms, 3)
    - returned energy       : eV (float)
    - returned forces       : eV / Angstrom, shape (n_atoms, 3), F = -dE/dx

dxtb works natively in atomic units (Bohr, Hartree). We convert with the *same*
CODATA constants ASE/xtb-python use (``ase.units.Hartree`` and ``ase.units.Bohr``)
so that a correspondence test measures only *algorithmic* differences between the
two implementations, not a mismatch in unit constants.

Notes
-----
* ``device`` selects cpu / cuda / mps. Use ``dtype=torch.double`` for the
  correspondence test; MPS supports only float32.
* ``GFN2-xTB`` requires the libcint backend (``tad-libcint``), which ships
  Linux-only wheels. On macOS only ``GFN1-xTB`` is available.
"""

from __future__ import annotations

import numpy as np
import torch
from ase.units import Bohr, Hartree

import dxtb

# Silence dxtb's per-call "Total Energy: ... Hartree." banner.
dxtb.OutputHandler.verbosity = 0

# Map the reference method strings onto dxtb parametrizations.
_PARAM = {
    "GFN1-xTB": dxtb.GFN1_XTB,
    "GFN2-xTB": dxtb.GFN2_XTB,
}


class DxtbPotential:
    """xTB potential backed by dxtb; mirrors XTBPotential.energy_forces."""

    def __init__( self,
                  Z: np.ndarray,
                  charge: int = 0,
                  uhf: int = 0,
                  method: str = "GFN1-xTB",
                  accuracy: float = 1.0,
                  electronic_temperature: float = 300.0,
                  max_iterations: int = 250,
                  solvent: str | None = None,
                  device: str | torch.device = "cpu",
                  dtype: torch.dtype = torch.double,
                ):
        if method not in _PARAM:
            raise ValueError( f"Unknown xTB method {method!r}; expected one of {list(_PARAM)}" )
        if solvent is not None:
            raise NotImplementedError( "Implicit solvation is not wired up in DxtbPotential yet." )

        self.Z = np.asarray( Z, dtype=int )
        self.method = method
        self.accuracy = accuracy
        self.electronic_temperature = electronic_temperature
        self.max_iterations = max_iterations
        self.device = torch.device(device)
        self.dtype = dtype

        # dxtb takes total charge and number of unpaired electrons per call.
        self.charge = float( charge )
        self.spin = int( uhf )

        self.atomic_numbers = torch.tensor(self.Z, dtype=torch.long, device=self.device)

        # Options merged into dxtb's Config. fermi_etemp default is already
        # 300 K; maxiter default is 100, so bump it to match the reference.
        self._opts = {
            "maxiter": int(max_iterations),
            "fermi_etemp": float(electronic_temperature),
            "verbosity": 0,
        }
        self.calc = dxtb.Calculator(
            self.atomic_numbers,
            _PARAM[method],
            opts=self._opts,
            device=self.device,
            dtype=self.dtype,
        )

    def energy_forces( self, x_A: np.ndarray ) -> tuple[float, np.ndarray]:
        """
        x_A: positions in Angstrom, shape (n_atoms, 3)

        returns:
            energy_eV: float
            forces_eV_per_A: shape (n_atoms, 3)
        """
        pos = torch.tensor( np.asarray(x_A, dtype=float) / Bohr,  # Angstrom -> Bohr
            device=self.device,
            dtype=self.dtype,
            requires_grad=True,
        )

        # Fresh SCF each call (positions change); clears cached tensors.
        self.calc.reset()

        energy_au = self.calc.get_energy( pos, chrg=self.charge, spin=self.spin )
        (grad_au,) = torch.autograd.grad( energy_au.sum(), pos )
        forces_au = -grad_au  # F = -dE/dx, matching ASE convention

        energy_eV = float( energy_au.detach().cpu() ) * Hartree
        forces_eV_A = forces_au.detach().cpu().numpy() * (Hartree / Bohr)

        return energy_eV, forces_eV_A
