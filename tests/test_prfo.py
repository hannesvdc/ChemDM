"""
Tests for chemdm.prfo: trans/rot projection, Bofill update, and a 2D Mueller-Brown
end-to-end convergence to a known saddle.
"""

import numpy as np
import pytest
import torch as pt

from chemdm.MoleculeGraph import MoleculeGraph
from chemdm.prfo import (
    PRFOOptimizer,
    _bofill_update,
    _project_mat,
    _project_vec,
    _trans_rot_basis,
    estimate_lowest_mode,
    lindh_model_hessian,
)


# ============================================================
# Trans/rot projector
# ============================================================

def test_trans_rot_basis_shape_generic():
    rng = np.random.default_rng(0)
    x = rng.standard_normal( (6, 3) )
    V = _trans_rot_basis(x)
    assert V.shape == (18, 6)
    np.testing.assert_allclose( V.T @ V, np.eye(6), atol=1e-12 )


def test_trans_rot_basis_drops_linear_redundant_axis():
    # Atoms strictly along z: rotations about z are degenerate -> 5 modes survive.
    x = np.zeros( (4, 3) )
    x[:, 2] = np.arange(4, dtype=float)
    V = _trans_rot_basis( x )
    assert V.shape[1] == 5


def test_project_kills_pure_translation_and_rotation():
    rng = np.random.default_rng(1)
    x = rng.standard_normal( (5, 3) )
    V = _trans_rot_basis( x )

    # Pure x-translation.
    trans = np.zeros_like(x)
    trans[:, 0] = 0.37
    assert np.linalg.norm( _project_vec(trans.reshape(-1), V) ) < 1e-10

    # Infinitesimal rotation about z, applied to centered coords.
    c = x.mean( axis=0 )
    r = x - c
    rot = np.cross( np.array([0.0, 0.0, 1.0]), r ) # the first is the z-unit vector
    assert np.linalg.norm( _project_vec(rot.reshape(-1), V) ) < 1e-10


def test_project_preserves_internal_direction():
    # A stretch of two atoms along x (purely internal) must survive projection.
    x = np.array([[-1.0, 0.0, 0.0],
                  [+1.0, 0.0, 0.0],
                  [ 0.0, 1.0, 0.0]])
    V = _trans_rot_basis(x)
    stretch = np.array([[-1.0, 0.0, 0.0],
                        [+1.0, 0.0, 0.0],
                        [ 0.0, 0.0, 0.0]]).reshape(-1)
    projected = _project_vec( stretch, V )

    # For this symmetric configuration the stretch is *exactly* orthogonal to
    # every trans/rot mode:
    #   - x-trans (1,0,0, 1,0,0, 1,0,0) · stretch = -1 + 1 + 0 = 0.
    #   - y/z-trans live in y/z slots; stretch is all-x.
    #   - z-rotation about centroid (0, 1/3, 0) sends atoms 0 and 1 with
    #     EQUAL x-components (each +1/3), cancelling against stretch's
    #     opposite signs.
    #   - x/y-rotations only generate z-displacements for atoms in this
    #     xy-plane configuration, trivially orthogonal to the all-x stretch.
    # So (I - V V^T) @ stretch must equal stretch to machine precision. A real
    # bug in the projector (e.g. `P = V V^T` instead of `I - V V^T`, missing
    # rotation generator, wrong centroid) would fail this immediately.
    np.testing.assert_allclose( projected, stretch, atol=1e-12 )
    np.testing.assert_allclose( np.linalg.norm(projected), np.linalg.norm(stretch), rtol=1e-12 )


def test_project_mat_idempotent_on_projected_subspace():
    rng = np.random.default_rng(2)
    x = rng.standard_normal( (4, 3) )
    V = _trans_rot_basis(x)
    M = rng.standard_normal( (12, 12) )
    M = 0.5 * (M + M.T)
    Mp = _project_mat(M, V)
    Mpp = _project_mat(Mp, V)
    np.testing.assert_allclose(Mp, Mpp, atol=1e-10)
    # Projected matrix annihilates trans/rot subspace from both sides.
    np.testing.assert_allclose(Mp @ V, 0.0, atol=1e-10)
    np.testing.assert_allclose(V.T @ Mp, 0.0, atol=1e-10)


# ============================================================
# Bofill update
# ============================================================

def test_bofill_satisfies_secant_condition():
    # For any quasi-Newton update of SR1/PSB family, H_new @ dx = dg should hold.
    rng = np.random.default_rng(3)
    n = 8
    H = np.eye(n)
    dx = rng.standard_normal(n)
    dg = rng.standard_normal(n)
    H_new = _bofill_update(H, dx, dg)
    np.testing.assert_allclose(H_new @ dx, dg, atol=1e-10)


def test_bofill_preserves_symmetry():
    rng = np.random.default_rng(4)
    n = 6
    H = rng.standard_normal((n, n))
    H = 0.5 * (H + H.T)
    dx = rng.standard_normal(n)
    dg = rng.standard_normal(n)
    H_new = _bofill_update(H, dx, dg)
    np.testing.assert_allclose(H_new, H_new.T, atol=1e-10)


def test_bofill_can_introduce_negative_eigenvalue():
    # Start from H = I; feed a gradient difference consistent with negative curvature
    # along x_0. After update, the smallest eigenvalue should be negative.
    n = 5
    H = np.eye(n)
    dx = np.zeros(n); dx[0] = 0.1
    dg = np.zeros(n); dg[0] = -0.2  # implies curvature -2 along x_0
    H_new = _bofill_update(H, dx, dg)
    assert np.min( np.linalg.eigvalsh(H_new) ) < 0.0


# ============================================================
# Lindh model Hessian
# ============================================================

def _directed( edges: list[tuple[int, int]] ) -> pt.Tensor:
    out = []
    for i, j in edges:
        out.append((i, j)); out.append((j, i))
    return pt.tensor(out, dtype=pt.long) if out else pt.zeros((0, 2), dtype=pt.long)


def _hcn_molecule() -> MoleculeGraph:
    Z = pt.tensor([1, 6, 7])
    x = pt.tensor([[-1.07, 0.0, 0.0],
                   [ 0.00, 0.0, 0.0],
                   [ 1.16, 0.0, 0.0]], dtype=pt.float64)
    return MoleculeGraph(Z=Z, x=x, bonds=_directed([(0, 1), (1, 2)]))


def test_lindh_hessian_shape_and_symmetry():
    mol = _hcn_molecule()
    H = lindh_model_hessian(mol)
    assert H.shape == (9, 9)
    np.testing.assert_allclose(H, H.T, atol=1e-12)


def test_lindh_hessian_psd_and_zero_modes():
    mol = _hcn_molecule()
    H = lindh_model_hessian( mol )
    eigvals = np.linalg.eigvalsh(H)
    # PSD up to roundoff
    assert eigvals.min() > -1e-8
    # HCN is linear: 5 rigid-body zero modes (3 trans + 2 rot).
    # With stretches only the bend modes are *also* zero — for HCN that's
    # another 2 (degenerate bend), so total zero modes = 7, nonzero = 2.
    n_zero = int(np.sum(eigvals < 1e-6))
    assert n_zero == 7, f"eigvals={eigvals}"


def test_lindh_hessian_bond_stiffness_along_axis():
    # Stretching the H-C bond (atoms 0 and 1) along +x should cost energy:
    # u·H·u must be positive for the displacement vector "atom 0 moves -x,
    # atom 1 moves +x" (asymmetric stretch).
    mol = _hcn_molecule()
    H = lindh_model_hessian(mol)
    u = np.zeros(9) # translation vector
    u[0] = -1.0   # H moves -x
    u[3] = +1.0   # C moves +x (atoms 0 and 1)
    u /= np.linalg.norm(u)
    assert float(u @ H @ u) > 1.0  # should be a few eV/Å²


# ============================================================
# Dimer rotation on a known quadratic potential
# ============================================================

class _QuadraticPotential:
    """E(x) = 0.5 (x - x0)^T H (x - x0)."""
    def __init__( self, H_true: np.ndarray, x0: np.ndarray ):
        self.H = H_true
        self.x0 = x0.flatten()

    def energy_forces( self, x: np.ndarray ):
        dx = x.flatten() - self.x0
        E = 0.5 * float(dx @ self.H @ dx)
        g = self.H @ dx
        return E, (-g).reshape(x.shape)


def test_dimer_rotation_finds_lowest_mode_of_quadratic():
    # 4 atoms in a tetrahedron-ish arrangement so trans/rot basis is 6.
    rng = np.random.default_rng(7)
    x = rng.standard_normal( (4, 3) )

    # Build a true Hessian: identity in physical subspace + a soft mode along
    # a known internal direction.
    V = _trans_rot_basis(x)  # (12, 6)

    # An "internal" direction: take a random vector, project trans/rot out.
    u_true = rng.standard_normal(12)
    u_true = _project_vec( u_true, V ) 
    u_true /= np.linalg.norm( u_true )
    H_phys = np.eye(12) - V @ V.T  # I on physical, 0 on trans/rot
    H_true = 1.5 * H_phys - 1.0 * np.outer( u_true, u_true )  # soft mode lam = 0.5

    mol = MoleculeGraph(
        Z=pt.tensor([1, 1, 1, 1]),
        x=pt.tensor(x, dtype=pt.float64),
        bonds=_directed([(0, 1), (1, 2), (2, 3)]),  # arbitrary, only used for Lindh seed
    )
    potential = _QuadraticPotential(H_true, x)

    u_est, lam_est = estimate_lowest_mode( potential, mol, max_iter=20, eps=1e-3, tol=1e-6 )

    # Same direction (or its negative).
    overlap = abs(float(u_est @ u_true))
    assert overlap > 0.999, f"overlap={overlap}, lam_est={lam_est}"
    assert abs(lam_est - 0.5) < 1e-3


# ============================================================
# PRFOOptimizer accepts MoleculeGraph
# ============================================================

def test_prfo_optimizer_accepts_moleculegraph():
    mol = _hcn_molecule()
    # Dummy evaluator that doesn't really matter — we only check construction
    # extracts positions correctly and stashes the molecule.
    class _Dummy:
        def energy_forces(self, x):
            return 0.0, np.zeros_like(x)
    # init_mode=None skips the Lanczos auto-seed (which would call _Dummy's
    # zero-force evaluator and produce a meaningless u).
    opt = PRFOOptimizer(_Dummy(), mol, init_mode=None)
    assert opt.molecule is mol
    assert opt._shape == (3, 3)
    np.testing.assert_allclose( opt.x.reshape(3, 3),  _to_numpy_helper(mol.x), atol=1e-12 )

def _to_numpy_helper(t):
    return t.detach().cpu().numpy()


