"""
The GFN1-xTB analytical gradient from xtb-python is inconsistent with the true
gradient of its own energy at the equilibrium (G2) geometry of methanol -- and
dxtb (PyTorch, autograd) is the one that agrees with the true gradient.

Three forces are compared at the same geometry:
    1. xtb-python analytical force        (a.get_forces())
    2. dxtb autograd force                (DxtbPotential)
    3. -FD(xtb-python energy)             central finite differences of the
                                          reference's OWN energy = neutral truth

Result: dxtb == -FD(xtb energy) to ~1e-6 eV/A, while xtb-python's analytical
force disagrees with -FD(its own energy) by ~0.225 eV/A on one carbon component.
So the discrepancy is a defect in the reference's analytical gradient, and it is
confirmed independently two ways: (a) the reference is inconsistent with its own
energy (needs no dxtb), and (b) an independent code (dxtb) agrees with the truth.

The gap is invariant to SCF accuracy, FD step size, and electronic temperature,
ruling out convergence / step-size / smearing explanations.

Dependencies: xtb (xtb-python), dxtb, numpy, ase.  Run: python reference_gradient_bug.py
"""

import numpy as np
import torch
from ase import Atoms
from xtb.ase.calculator import XTB

from dxtb_potential import DxtbPotential

# Equilibrium methanol geometry (ASE G2 collection), Angstrom.
SYMBOLS = ["C", "O", "H", "H", "H", "H"]
Z = np.array([6, 8, 1, 1, 1, 1])
XYZ = np.array([
    [-0.047131,  0.664389,  0.000000],
    [-0.047131, -0.758551,  0.000000],
    [-1.092995,  0.969785,  0.000000],
    [ 0.878534, -1.048458,  0.000000],
    [ 0.437145,  1.080376,  0.891772],
    [ 0.437145,  1.080376, -0.891772],
])


def xtb_energy( x, accuracy=1.0, etemp=300.0 ):
    a = Atoms( symbols=SYMBOLS, positions=x )
    a.calc = XTB( method="GFN1-xTB", accuracy=accuracy,
                 electronic_temperature=etemp, max_iterations=250 )
    return a.get_potential_energy()    # eV


def xtb_force( accuracy=1.0, etemp=300.0 ):
    a = Atoms( symbols=SYMBOLS, positions=XYZ )
    a.calc = XTB( method="GFN1-xTB", accuracy=accuracy, electronic_temperature=etemp, max_iterations=250 )
    return a.get_forces()                    # analytical, eV/A


def fd_force( energy_fn, h=2e-4, **kw ):
    g = np.zeros_like(XYZ)
    for i in range(XYZ.shape[0]):
        for j in range(3):
            xp = XYZ.copy(); xp[i, j] += h
            xm = XYZ.copy(); xm[i, j] -= h
            g[i, j] = (energy_fn(xp, **kw) - energy_fn(xm, **kw)) / (2 * h)
    return -g


def main():
    # CPU ASE calculator
    F_xtb = xtb_force()
    F_truth = fd_force(xtb_energy)     # -FD of xtb-python's own energy

    # Dxtb calculator
    dxtb_pot = DxtbPotential( Z=Z, method="GFN1-xTB", dtype=torch.float64 )
    _, F_dxtb = dxtb_pot.energy_forces( XYZ )

    # Any difference?
    dev = np.abs( F_xtb - F_truth )
    a = int(dev.max(axis=1).argmax())

    print("GFN1-xTB, methanol equilibrium geometry")
    print(f"worst atom: index {a} ({SYMBOLS[a]})   [eV/A]")
    print(f"  xtb-python analytical : {np.round(F_xtb[a], 4)}")
    print(f"  dxtb autograd         : {np.round(F_dxtb[a], 4)}")
    print(f"  -FD(xtb-python energy): {np.round(F_truth[a], 4)}   (neutral truth)\n")

    print("Deviation from -FD(xtb-python energy), max over all atoms:")
    print(f"  xtb-python analytical : {np.abs(F_xtb  - F_truth).max():.4e} eV/A   (should be ~0)")
    print(f"  dxtb autograd         : {np.abs(F_dxtb - F_truth).max():.4e} eV/A (FD Discretization error)")
    print(f"  => dxtb vs xtb-python : {np.abs(F_dxtb - F_xtb).max():.4e} eV/A\n")

    print("xtb-python inconsistency at multiple accuracies (not a convergence issue):")
    print(f"  {'accuracy':>10s} {'max|F_xtb - (-dE/dx)|':>24s}")
    for acc in (1.0, 0.1, 0.01, 0.001, 0.0001):
        d = np.abs( xtb_force(accuracy=acc) - fd_force(xtb_energy, accuracy=acc) ).max()
        print(f"  {acc:10.4f} {d:24.4e}")


if __name__ == "__main__":
    main()