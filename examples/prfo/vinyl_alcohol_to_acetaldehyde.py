"""
P-RFO demo: vinyl alcohol → acetaldehyde 1,3-H shift with xTB.

The keto-enol tautomerisation of vinyl alcohol (H2C=CH-OH) → acetaldehyde
(CH3-CHO) proceeds via a 4-membered cyclic transition state where the
hydroxyl hydrogen migrates from O to the terminal CH2 carbon while the
C=C / C=O double-bond pattern reorganises.

Pipeline (HCN-style, no chemistry-informed TS guess):
    1. Build vinyl alcohol with RDKit ETKDG.
    2. Relax to the GFN2-xTB minimum (the enol form).
    3. Estimate the lowest mass-weighted mode at the relaxed minimum. The
       slowest physical mode at the enol minimum is dominated by the OH
       hydrogen and a CH₂ hydrogen on Cα moving in opposite directions —
       exactly the 1,3-H-shift reaction coordinate.
    4. Perturb by 0.5 Å along that mode and run P-RFO with a small trust
       radius (max_trust=0.01 — vinyl alcohol's PES has steeper gradients
       than HCN's; with larger trust the optimiser overshoots).
    5. Verify: exactly one negative Hessian eigenvalue, report barrier.

Earlier versions used a chemistry-informed TS-like starting geometry
(migrating H placed at the O-Cα midpoint with an out-of-plane offset).
That approach landed +347 kcal/mol above the enol minimum because the
carbon framework (C-C, C-O) was left in pure-enol geometry, leaving Cα
pentavalent — xTB rejected it and P-RFO cascaded downhill at max trust
without locating any saddle. The Lanczos-mode kick avoids the issue by
starting near the minimum.

Run:
    python examples/prfo/vinyl_alcohol_to_acetaldehyde.py
"""

from __future__ import annotations

import numpy as np
import torch as pt

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

from chemdm.MoleculeGraph import MoleculeGraph
from chemdm.prfo import PRFOOptimizer, lindh_lowest_mode, _project_mat, _trans_rot_basis
from chemdm.relaxMolecule import relaxMolecule
from chemdm.TBLitePotential import TBLitePotential
from chemdm.Constants import EV_TO_KCAL_PER_MOL

from _plot import plot_prfo_trajectory

RDLogger.DisableLog( "rdApp.*" )

VINYL_ALCOHOL_SMILES = "C=CO"     # H2C=CH-OH


def build_vinyl_alcohol() -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]], dict[str, int]]:
    """
    Generate a 3D vinyl alcohol structure via RDKit ETKDG.

    Returns
    -------
    Z           : (N,) atomic numbers
    positions   : (N, 3) Å
    bond_pairs  : list of (i, j) undirected covalent-bond endpoints
    key_atoms   : dict mapping {"O", "H_OH", "C_alpha"} → atom index. The
                  "H_OH" is the migrating hydrogen; "C_alpha" is the terminal
                  =CH2 carbon that the H needs to reach.
    """
    mol = Chem.MolFromSmiles(VINYL_ALCOHOL_SMILES)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule( mol, randomSeed=42 )
    AllChem.MMFFOptimizeMolecule( mol, maxIters=200 )

    Z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=int)
    positions = mol.GetConformer().GetPositions()      # already in Å

    bond_pairs = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]

    # Identify key atoms by chemistry, not by index — robust to RDKit's
    # internal atom ordering.
    o_idx = next( i for i, z in enumerate(Z) if z == 8 )
    o_atom = mol.GetAtomWithIdx(o_idx)
    h_oh = next( n.GetIdx() for n in o_atom.GetNeighbors() if n.GetAtomicNum() == 1 )
    c_with_o = next( n.GetIdx() for n in o_atom.GetNeighbors() if n.GetAtomicNum() == 6 )
    c_with_o_atom = mol.GetAtomWithIdx(c_with_o)
    # The terminal =CH2 carbon is the OTHER carbon (bonded via C=C).
    c_alpha = next( n.GetIdx() for n in c_with_o_atom.GetNeighbors()
                    if n.GetAtomicNum() == 6 )

    key_atoms = {"O": o_idx, "H_OH": h_oh, "C_alpha": c_alpha, "C_O": c_with_o}
    return Z, positions, bond_pairs, key_atoms


def directed_bond_tensor( bond_pairs: list[tuple[int, int]] ) -> pt.Tensor:
    """Convert undirected (i, j) pairs to directed (n_edges, 2) tensor for MoleculeGraph."""
    edges = []
    for i, j in bond_pairs:
        edges.append((i, j)); edges.append((j, i))
    return pt.tensor(edges, dtype=pt.long)


def finite_difference_hessian( potential: TBLitePotential, x: np.ndarray, h: float = 1e-3 ) -> np.ndarray:
    """Central-difference Cartesian Hessian (eV/Å²) for the saddle-character check."""
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
    Z, x_init, bond_pairs, key = build_vinyl_alcohol()
    potential = TBLitePotential(Z=Z)
    print(f"=== Vinyl alcohol: {len(Z)} atoms, {len(bond_pairs)} bonds ===")
    print(f"key atoms: O={key['O']}, H_OH={key['H_OH']}, C_O={key['C_O']}, C_alpha={key['C_alpha']}")

    print("\n=== Relaxing vinyl alcohol minimum (GFN2-xTB) ===")
    x_min = relaxMolecule( potential, x_init, minimizer="Adam",
                           force_tol=1e-3, max_steps=2000, verbose=False )
    E_min, F_min = potential.energy_forces( x_min )  # type: ignore
    print( f"E_min  = {E_min:+.6f} eV" )
    print( f"|F|max = {np.abs(F_min).max():.2e} eV/Å" )

    mol_min = MoleculeGraph(
        Z=pt.tensor(Z, dtype=pt.long),
        x=pt.tensor(x_min, dtype=pt.float64),
        bonds=directed_bond_tensor( bond_pairs ),
    )

    print( "\n=== Estimating lowest mode at enol minimum (mass-weighted Lindh) ===" )
    u = lindh_lowest_mode( mol_min )
    u_3d = u.reshape(-1, 3)
    norms = np.linalg.norm(u_3d, axis=1)
    sym = {1: "H", 6: "C", 8: "O"}
    top5 = np.argsort(norms)[::-1][:5]
    print(  "  top-5 atoms by displacement:" )
    for a in top5:
        marker = ""
        if a == key["H_OH"]:    marker = "  (migrating OH hydrogen)"
        elif a == key["C_alpha"]: marker = "  (Cα — accepts the migrating H)"
        elif a == key["O"]:       marker = "  (O — loses the migrating H)"
        elif a == key["C_O"]:     marker = "  (C bonded to O)"
        print( f"    atom {a:2d}  {sym[int(Z[a])]}  |u_i|={norms[a]:.3f}{marker}" )

    # Vinyl alcohol's lowest mode at the enol minimum has *positive* curvature
    # (it's a minimum). A small kick along it leaves the harmonic region but
    # the followed mode is still in positive-curvature territory; Bofill then
    # overwrites the rank-1 overlay with the true positive value and P-RFO
    # has no unstable direction to climb. Empirically a 1.0 Å kick gets the
    # geometry far enough that the followed mode crosses to negative curvature.
    kick = 1.0
    x_perturbed = x_min #+ kick * u_3d
    print( f"\nPerturbing by {kick:.2f} Å along the lowest mode before P-RFO." )

    mol_perturbed = mol_min.copyWithNewPositions( pt.tensor(x_perturbed, dtype=pt.float64) )

    print( "\n=== P-RFO ascent toward TS ===" )
    # Smaller max_trust than HCN: vinyl alcohol's gradient magnitudes near the
    # 4-membered-ring TS are larger and the surface is steeper. With HCN's
    # 0.1 Å max trust the optimiser overshoots the saddle.
    # relanczos_every=5 re-anchors the unstable mode against the true Hessian
    # every 5 steps — important here because the reaction coordinate's direction
    # rotates substantially as C-C and C-O reorganise around the migrating H.
    min_radius = 1e-4
    max_radius = 1e-1
    opt = PRFOOptimizer(
        potential, mol_perturbed,
        trust_radius=min_radius,
        max_trust=max_radius,
        min_trust=min_radius,
        relanczos_every=1,
    )

    max_iter = 1000
    try:
        result = opt.run( max_iter=max_iter, tol_g=1e-3, tol_step=1e-4, verbose=True )
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

    if not result["converged"]:
        if result.get("crashed"):
            print( f"\nOptimization did NOT converge — crash at iter {result['n_iter']}." )
        else:
            print( "\nOptimization did NOT converge within max_iter." )
        return

    x_ts = result["x"]
    E_ts = result["energy"]

    print( "\n=== TS candidate ===" )
    print( f"converged in {result['n_iter']} iterations" )
    print( f"E_ts   = {E_ts:+.6f} eV" )
    print( f"barrier = {(E_ts - E_min) * EV_TO_KCAL_PER_MOL:+.2f} kcal/mol" )

    # Key bond lengths at the TS — diagnostic for the 1,3-H shift.
    r_OH = np.linalg.norm( x_ts[key["O"]]      - x_ts[key["H_OH"]] )
    r_CH = np.linalg.norm( x_ts[key["C_alpha"]] - x_ts[key["H_OH"]] )
    r_CC = np.linalg.norm( x_ts[key["C_alpha"]] - x_ts[key["C_O"]] )
    print( f"\nKey TS bond lengths (Å):" )
    print( f"  O—H_OH      = {r_OH:.3f}   (was ~0.96 in enol minimum)" )
    print( f"  C_alpha—H   = {r_CH:.3f}   (forming bond)" )
    print( f"  C_alpha—C_O = {r_CC:.3f}   (~1.34 enol → ~1.50 keto over reaction)" )

    print( "\n=== Verifying saddle character (finite-difference Hessian) ===" )
    H_fd = finite_difference_hessian( potential, x_ts )
    V = _trans_rot_basis( x_ts )
    H_proj = _project_mat( H_fd, V )
    eigvals = np.linalg.eigvalsh( H_proj )
    physical = eigvals[np.abs(eigvals) > 1e-3]
    n_neg = int( np.sum(physical < 0.0) )
    print( f"physical eigenvalues (eV/Å²): {np.array2string(np.sort(physical)[:5], precision=2)} ... ({len(physical)} total)" )
    print( f"# negative = {n_neg}  -> {'first-order saddle ✓' if n_neg == 1 else 'NOT a clean saddle ✗'}" )

    plot_prfo_trajectory( opt.history, title_suffix="vinyl alcohol → acetaldehyde" )


if __name__ == "__main__":
    main()
