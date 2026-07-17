"""
P-RFO demo: alanine dipeptide on the GFN2-xTB potential.

Alanine dipeptide (Ac-Ala-NMe) is a canonical 22-atom test system with two
backbone dihedrals (φ, ψ) and several Ramachandran basins. Pipeline:

    1. Pull the starting structure (Z, positions, bond connectivity) from
       openmmtools.AlanineDipeptideImplicit. The AMBER `System` is discarded;
       openmmtools is used here only as a convenient source of a sensible
       PDB-style geometry.
    2. Build a chemdm.TBLitePotential (ASE-backed GFN2-xTB) and relax to a
       local minimum via :func:`chemdm.relaxMolecule.relaxMolecule`.
    3. Estimate the lowest mass-weighted Lindh mode at the relaxed minimum.
    4. Perturb along that mode.
    5. Run P-RFO.
    6. Verify saddle character via finite-difference Hessian.

The pipeline matches the HCN and vinyl-alcohol demos exactly — same xTB
evaluator, same `lindh_lowest_mode` seed, same P-RFO machinery. Only the
structure source differs (openmmtools instead of RDKit ETKDG, because
constructing a peptide via SMILES + ETKDG is finicky for backbone
geometry).

Run:
    python examples/prfo/alanine_dipeptide.py
"""

from __future__ import annotations

import numpy as np
import openmm.unit as unit
import torch as pt

from openmmtools import testsystems

from chemdm.MoleculeGraph import MoleculeGraph
from chemdm.prfo import (
    PRFOOptimizer, lindh_lowest_mode, _project_mat, _trans_rot_basis,
)
from chemdm.relaxMolecule import relaxMolecule
from chemdm.TBLitePotential import TBLitePotential
from chemdm.Constants import EV_TO_KCAL_PER_MOL

from _plot import plot_prfo_trajectory


NM_TO_A = 10.0


def directed_bond_tensor( bond_pairs: list[tuple[int, int]] ) -> pt.Tensor:
    edges = []
    for i, j in bond_pairs:
        edges.append((i, j)); edges.append((j, i))
    return pt.tensor(edges, dtype=pt.long)


def finite_difference_hessian( potential: TBLitePotential,
                               x: np.ndarray,
                               h: float = 1e-3 ) -> np.ndarray:
    """Central-difference Cartesian Hessian (eV/Å²)."""
    n = x.size
    H = np.zeros( (n, n) )
    x_flat = x.reshape(-1).copy()
    for i in range(n):
        xp = x_flat.copy(); xp[i] += h
        xm = x_flat.copy(); xm[i] -= h
        _, Fp = potential.energy_forces(xp.reshape(x.shape))
        _, Fm = potential.energy_forces(xm.reshape(x.shape))
        H[i, :] = -(Fp.reshape(-1) - Fm.reshape(-1)) / (2.0 * h)
    return 0.5 * (H + H.T)


def main() -> None:
    print( "=== Building alanine dipeptide structure (openmmtools, AMBER ignored) ===" )
    ad = testsystems.AlanineDipeptideImplicit()
    topology = ad.topology
    n_atoms = topology.getNumAtoms()
    Z = np.array( [atom.element.atomic_number for atom in topology.atoms()], dtype=int )
    bond_pairs = [(b[0].index, b[1].index) for b in topology.bonds()]
    x_init = np.asarray( ad.positions.value_in_unit(unit.nanometer) ) * NM_TO_A
    print( f"  {n_atoms} atoms, {len(bond_pairs)} bonds" )

    potential = TBLitePotential( Z=Z )

    print( "\n=== Relaxing alanine dipeptide minimum (GFN2-xTB) ===" )
    x_min = relaxMolecule( potential, x_init, minimizer="Adam",
                           force_tol=1e-3, max_steps=2000, verbose=False )
    E_min, F_min = potential.energy_forces( x_min )  # type: ignore
    print( f"  E_min  = {E_min:+.6f} eV" )
    print( f"  |F|max = {np.abs(F_min).max():.2e} eV/Å" )

    mol_min = MoleculeGraph(
        Z=pt.tensor(Z, dtype=pt.long),
        x=pt.tensor(x_min, dtype=pt.float64),
        bonds=directed_bond_tensor( bond_pairs ),
    )

    print( "\n=== Estimating lowest mode at minimum (mass-weighted Lindh) ===" )
    u = lindh_lowest_mode( mol_min )
    u_3d = u.reshape(-1, 3)
    norms = np.linalg.norm(u_3d, axis=1)
    top5 = np.argsort(norms)[::-1][:5]
    print( "  top-5 atoms in lowest mode (|u_i|):" )
    for atom_idx in top5:
        print( f"    atom {atom_idx:3d}  Z={Z[atom_idx]:2d}  |u_i|={norms[atom_idx]:.3f}" )

    kick = 0.5
    x_perturbed = x_min #+ kick * u_3d
    E_perturbed, F_perturbed = potential.energy_forces(x_perturbed)
    print( f"\nPerturbing by {kick:.2f} Å along the lowest mode before P-RFO." )
    print( f"  E_perturbed = {E_perturbed:+.4f} eV "
           f"(ΔE = {(E_perturbed - E_min) * EV_TO_KCAL_PER_MOL:+.3f} kcal/mol)" )
    print( f"  |F|max      = {np.abs(F_perturbed).max():.2e} eV/Å" )

    mol_perturbed = mol_min.copyWithNewPositions( pt.tensor(x_perturbed, dtype=pt.float64) )

    print( "\n=== P-RFO ascent toward TS ===" )
    min_radius = 1e-4
    max_radius = 1e-2
    opt = PRFOOptimizer(
        potential, mol_perturbed,
        trust_radius=min_radius,
        max_trust=max_radius,
        min_trust=min_radius,
        relanczos_every=5,
    )

    max_iter = 1000
    grad_tol = 1e-3
    step_tol = 1e-4
    try:
        result = opt.run( max_iter=max_iter, tol_g=grad_tol, tol_step=step_tol, verbose=True )
    except Exception as e:
        n_done = len(opt.history)
        print( f"\nOptimizer crashed after {n_done} iterations: {type(e).__name__}: {e}" )
        last_info = opt.history[-1] if opt.history else {}
        result = {
            "converged": False, "crashed": True,
            "exception": f"{type(e).__name__}: {e}",
            "n_iter": n_done,
            "x": opt.x.reshape(opt._shape).copy(),
            "energy": last_info.get("energy"),
        }

    if result["converged"]:
        x_ts = result["x"]
        E_ts = result["energy"]
        print( "\n=== TS candidate ===" )
        print( f"converged in {result['n_iter']} iterations" )
        print( f"E_ts    = {E_ts:+.6f} eV" )
        print( f"barrier = {(E_ts - E_min) * EV_TO_KCAL_PER_MOL:+.2f} kcal/mol" )

        print( "\n=== Verifying saddle character (finite-difference Hessian) ===" )
        H_fd = finite_difference_hessian( potential, x_ts )
        V = _trans_rot_basis( x_ts )
        H_proj = _project_mat( H_fd, V )
        eigvals = np.linalg.eigvalsh( H_proj )
        physical = eigvals[np.abs(eigvals) > 1e-3]
        n_neg = int( np.sum(physical < 0.0) )
        print( f"physical eigenvalues (eV/Å²): "
               f"smallest 5 = {np.array2string(np.sort(physical)[:5], precision=2)}  "
               f"largest 5 = {np.array2string(np.sort(physical)[-5:], precision=2)}" )
        print( f"# negative = {n_neg}  -> "
               f"{'first-order saddle ✓' if n_neg == 1 else 'NOT a clean first-order saddle ✗'}" )
    else:
        if result.get("crashed"):
            print( f"\nOptimization did NOT converge — crash at iter {result['n_iter']}." )
        else:
            print( "\nOptimization did NOT converge within max_iter." )

    # Plot the trajectory whether or not we converged — it is most
    # diagnostic precisely when mode-following struggled.
    plot_prfo_trajectory( opt.history, title_suffix="alanine dipeptide" )


if __name__ == "__main__":
    main()
