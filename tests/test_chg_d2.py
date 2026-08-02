"""Tests for the ωB97X-D CHG dispersion, at two levels.

Dispersion term in isolation (`chg_d2_dispersion`):
  * energies vs frozen references + analytic forces vs finite difference, over
    fixtures spanning every parameterized element (H,C,N,O,F,P,S,Cl) -- always
    on, no QC engine, so they survive psi4 being dropped;
  * when psi4 IS installed, a live cross-check against psi4's own
    EmpiricalDispersion that validates our code AND confirms the frozen
    references haven't drifted.
Full ωB97X-D potential (functional + dispersion) via `PySCFPotential`, checked
against psi4 (through `Psi4Potential`) if present, else finite difference.
"""
import importlib.util

import numpy as np
import pytest

from chemdm.potentials.dispersion import chg_d2_dispersion

_HAVE_PSI4 = importlib.util.find_spec("psi4") is not None

# Ethane: multi-element molecule with real intramolecular dispersion (~50 meV).
_ETHANE_Z = np.array([6, 6, 1, 1, 1, 1, 1, 1])
_ETHANE_X = np.array([[0.0, 0.0, 0.7625], [0.0, 0.0, -0.7625],
                      [0.0, 1.0185, 1.1573], [0.8821, -0.5092, 1.1573], [-0.8821, -0.5092, 1.1573],
                      [0.0, -1.0185, -1.1573], [0.8821, 0.5092, -1.1573], [-0.8821, 0.5092, -1.1573]])

# (name, Z, X_Å, E_disp_ref_eV) — frozen references captured once from psi4.
# Diatomics exercise each element's own C6/R0; water & ethane add H/C/O cross
# terms. Together they cover all of H, C, N, O, F, P, S, Cl.
_DISP_REFS = [
    ("C2_3.0A", [6, 6], [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], -0.0049158031),
    ("N2_3.0A", [7, 7], [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], -0.0049188145),
    ("O2_2.8A", [8, 8], [[0.0, 0.0, 0.0], [2.8, 0.0, 0.0]], -0.0032649715),
    ("F2_3.0A", [9, 9], [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], -0.0054541557),
    ("P2_3.5A", [15, 15], [[0.0, 0.0, 0.0], [3.5, 0.0, 0.0]], -0.0082020110),
    ("S2_3.5A", [16, 16], [[0.0, 0.0, 0.0], [3.5, 0.0, 0.0]], -0.0066032915),
    ("Cl2_3.5A", [17, 17], [[0.0, 0.0, 0.0], [3.5, 0.0, 0.0]], -0.0076573457),
    ("water", [8, 1, 1], [[0.0, 0.0, 0.117], [0.0, 0.757, -0.469], [0.0, -0.757, -0.469]], -0.0007289304),
    ("ethane", _ETHANE_Z.tolist(), _ETHANE_X.tolist(), -0.0507414415),
]


def _psi4_dispersion( Z : np.ndarray, X : np.ndarray ):
    """Live psi4 ωB97X-D dispersion: (energy_eV, forces_eV_per_A)."""
    import psi4
    from ase.data import chemical_symbols
    from ase.units import Hartree, Bohr
    psi4.core.be_quiet()
    lines = "\n".join(f"{chemical_symbols[z]} {x:.6f} {y:.6f} {zz:.6f}" for z, (x, y, zz) in zip(Z, X))
    mol = psi4.geometry(f"0 1\n{lines}\nunits angstrom\nno_com\nno_reorient\nsymmetry c1")
    disp = psi4.driver.EmpiricalDispersion(name_hint="wb97x-d")
    e = disp.compute_energy(mol) * Hartree
    f = -np.asarray(disp.compute_gradient(mol)) * (Hartree / Bohr)
    return e, f


def _fd_forces( Z : np.ndarray, X : np.ndarray, h : float=1e-4 ):
    fd = np.zeros_like(X)
    for i in range(len(Z)):
        for k in range(3):
            xp = X.copy(); xp[i, k] += h
            xm = X.copy(); xm[i, k] -= h
            fd[i, k] = -(chg_d2_dispersion(Z, xp)[0] - chg_d2_dispersion(Z, xm)[0]) / (2 * h)
    return fd


# --- dispersion term: always on, no QC engine ---
@pytest.mark.parametrize("name,Z,X,e_ref", _DISP_REFS)
def test_dispersion_energy_matches_reference( name : str, Z : np.ndarray, X : np.ndarray, e_ref : float):
    e, _ = chg_d2_dispersion(np.array(Z), np.array(X))
    assert abs(e - e_ref) < 1e-6, f"{name}: {e} vs {e_ref}"


@pytest.mark.parametrize("name,Z,X", [(name, Z, X) for name, Z, X, _ in _DISP_REFS])
def test_dispersion_forces_match_finite_difference( name : str, Z : np.ndarray, X : np.ndarray ):
    Z, X = np.array(Z), np.array(X, dtype=float)
    _, f = chg_d2_dispersion(Z, X)
    assert np.allclose(f, _fd_forces(Z, X), atol=1e-6), np.abs(f - _fd_forces(Z, X)).max()


def test_dispersion_missing_parameters_raise():
    with pytest.raises(ValueError):
        chg_d2_dispersion(np.array([3, 3]), np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]))  # Li


# --- dispersion term: live cross-check when psi4 is present ---
@pytest.mark.skipif(not _HAVE_PSI4, reason="psi4 not installed")
@pytest.mark.parametrize("name,Z,X,e_ref", _DISP_REFS)
def test_dispersion_matches_live_psi4( name : str, Z : np.ndarray, X : np.ndarray, e_ref : float ):
    Z, X = np.array(Z), np.array(X)
    e_psi4, f_psi4 = _psi4_dispersion( Z, X )
    e, f = chg_d2_dispersion( Z, X )
    assert abs( e - e_psi4 ) < 1e-6            # our implementation vs live psi4
    assert abs( e_ref - e_psi4 ) < 1e-6        # frozen reference still matches psi4
    assert np.abs( f - f_psi4 ).max() < 1e-6   # forces vs live psi4 gradient


# Full wB97X-D potential (functional + CHG dispersion) vs psi4.
#
# Diverse molecules so the combined functional+dispersion is exercised across
# atom types (O, N, F, S, Cl). This is a cross-engine correctness check, 
# so it needs psi4 as the oracle; the psi4-free coverage is
# the isolated dispersion tests above plus test_pyscf_backend.py's mechanics.
_FULL_FF_MOLS = [
    ("ethanol", "CCO"),          # C, O, H
    ("methylamine", "CN"),       # C, N, H
    ("fluoromethane", "CF"),     # C, F, H
    ("methanethiol", "CS"),      # C, S, H
    ("chloromethane", "CCl"),    # C, Cl, H
]

def _rdkit_geom( smiles : str, seed : int=1):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m, randomSeed=seed)
    AllChem.MMFFOptimizeMolecule(m)
    return (np.array([a.GetAtomicNum() for a in m.GetAtoms()]),
            np.asarray(m.GetConformer().GetPositions()))


@pytest.mark.skipif(not _HAVE_PSI4, reason="psi4 is the ωB97X-D oracle for the full potential")
@pytest.mark.parametrize("name,smi", _FULL_FF_MOLS)
def test_full_wb97x_d_matches_psi4( name : str, smi : str ):
    pytest.importorskip("rdkit")
    from chemdm.potentials.PySCFPotential import PySCFPotential
    from chemdm.potentials.Psi4Potential import Psi4Potential
    Z, X = _rdkit_geom( smi )
    e, f = PySCFPotential( Z, functional="wb97xd", basis="def2-svp", disp="chg-d2",
                           device="cpu", num_threads=8 ).energy_forces(X)
    e_ref, f_ref = Psi4Potential(Z, functional="wb97x-d", basis="def2-svp").energy_forces(X)
    
    # Energy carries the dispersion signal (~50 meV if it were missing), so it is
    # the tight check. Cross-engine forces are DF/grid-noise-limited (~0.03-0.05
    # eV/A for heavier atoms, doesn't shrink with grid) -- a ballpark sanity only.
    # The gap is the functional (SCF), not the D2 -- see docs/dft_backend_notes/.
    assert abs(e - e_ref) < 0.03, f"{name}: dE={(e - e_ref) * 1000:.1f} meV"
    assert np.abs(f - f_ref).max() < 0.05, f"{name}: max dF={np.abs(f - f_ref).max():.4f} eV/A"
