"""Tests for the L-BFGS geometry minimizer (chemdm.relaxMolecule.minimize_with_lbfgs).

These exercise the optimizer on analytic potentials that mimic real molecular
PES features -- an anharmonic Lennard-Jones bond with a stiff repulsive wall, a
Morse bond, a Lennard-Jones trimer with a non-trivial equilateral minimum, and a
deliberately ill-conditioned harmonic well (stiff + soft normal modes, the case
that made Adam jump uphill and then drift). Each stub returns energy in eV and
forces (= -dE/dx) in eV/Angstrom, matching chemdm.xtbSetup.XTBPotential, so the
minimizer's internal eV->kJ/mol conversion is exercised too.

A finite-difference check validates the stub gradients themselves, so a failing
optimizer assertion can't be blamed on a bad test potential.
"""
import numpy as np
import pytest

from chemdm.relaxMolecule import minimize_with_lbfgs, minimize_with_adam, relaxMolecule


# Analytic potentials with the XTBPotential.energy_forces contract.
class LennardJones:
    """Pairwise LJ over all atom pairs. Min pair distance r* = 2**(1/6) * sigma."""
    def __init__(self, eps=0.01, sigma=1.5):
        self.eps, self.sigma = eps, sigma

    @property
    def r_min(self):
        return 2.0 ** (1.0 / 6.0) * self.sigma

    def energy_forces( self, x : np.ndarray ) -> tuple[float, np.ndarray]:
        x = np.asarray(x, dtype=float)
        n = len(x)
        E = 0.0
        grad = np.zeros_like(x)
        for i in range(n):
            for j in range(i + 1, n):
                rij = x[i] - x[j]
                r = float( np.linalg.norm(rij) )
                sr6 = (self.sigma / r) ** 6
                sr12 = sr6 * sr6
                E += 4.0 * self.eps * (sr12 - sr6)
                dEdr = 4.0 * self.eps * (-12.0 * sr12 + 6.0 * sr6) / r
                gi = dEdr * rij / r
                grad[i] += gi
                grad[j] -= gi
        return E, -grad


class MorseBond:
    """Diatomic Morse bond E = D (1 - exp(-a(r-r0)))^2. Min at r0."""
    def __init__(self, D=0.1, a=1.5, r0=1.4):
        self.D, self.a, self.r0 = D, a, r0

    def energy_forces( self, x : np.ndarray ) -> tuple[float, np.ndarray]:
        x = np.asarray(x, dtype=float)
        rij = x[0] - x[1]
        r = np.linalg.norm(rij)
        e = np.exp(-self.a * (r - self.r0))
        E = self.D * (1.0 - e) ** 2
        dEdr = 2.0 * self.D * (1.0 - e) * self.a * e
        g0 = dEdr * rij / r
        return E, -np.stack([g0, -g0])


class AnisotropicHarmonic:
    """Per-atom harmonic well anchored at `target` with per-atom stiffness `k`.

    Convex with a unique minimum at `target`. Disparate k entries make it
    ill-conditioned (stiff vs soft modes), the regime where Adam overshoots."""
    def __init__(self, target, k):
        self.target = np.asarray(target, dtype=float)
        self.k = np.asarray(k, dtype=float)[:, None]   # (n,1) broadcast over xyz

    def energy_forces( self, x : np.ndarray ) -> tuple[float, np.ndarray]:
        x = np.asarray(x, dtype=float)
        d = x - self.target
        E = 0.5 * float(np.sum(self.k * d * d))
        grad = self.k * d
        return E, -grad


def pair_distances(x):
    x = np.asarray(x)
    return np.array([np.linalg.norm(x[i] - x[j])
                     for i in range(len(x)) for j in range(i + 1, len(x))])


# Cases reused across the generic property tests: (name, potential, x0).
def _cases():
    lj = LennardJones()
    morse = MorseBond()
    harm = AnisotropicHarmonic(
        target=np.array([[0., 0., 0.], [1.3, 0., 0.], [0., 1.1, 0.], [0.5, 0.5, 0.9]]),
        k=np.array([40.0, 0.4, 40.0, 0.4]),            # condition number 100
    )
    return [
        ("lj_dimer_repulsive", lj, np.array([[0., 0., 0.], [1.30, 0., 0.]])),   # inside the wall
        ("lj_dimer_attractive", lj, np.array([[0., 0., 0.], [2.40, 0., 0.]])),  # past the well
        ("morse_compressed", morse, np.array([[0., 0., 0.], [1.00, 0., 0.]])),
        ("morse_stretched", morse, np.array([[0., 0., 0.], [2.20, 0., 0.]])),
        ("lj_trimer", lj, np.array([[0., 0., 0.], [1.4, 0., 0.], [0.7, 1.1, 0.]])),
        ("aniso_harmonic", harm,
         np.array([[0., 0., 0.], [1.3, 0., 0.], [0., 1.1, 0.], [0.5, 0.5, 0.9]]) + 0.3),
    ]


CASE_IDS = [c[0] for c in _cases()]
TOL = 1e-2  # kJ/mol/A


# Stub-potential sanity: forces are -gradient of the energy.
@pytest.mark.parametrize("pot,x", [
    (LennardJones(), np.array([[0., 0., 0.], [1.7, 0.2, 0.1], [0.6, 1.5, -0.3]])),
    (MorseBond(), np.array([[0., 0., 0.], [1.55, 0.0, 0.0]])),
    (AnisotropicHarmonic(np.zeros((3, 3)), np.array([5.0, 1.0, 20.0])),
     np.array([[0.1, -0.2, 0.05], [0.3, 0.4, -0.1], [-0.2, 0.1, 0.25]])),
])
def test_stub_forces_match_finite_difference(pot, x):
    _, F = pot.energy_forces(x)
    h = 1e-6
    F_fd = np.zeros_like(x)
    for i in range(x.shape[0]):
        for d in range(3):
            xp = x.copy(); xp[i, d] += h
            xm = x.copy(); xm[i, d] -= h
            Ep, _ = pot.energy_forces(xp)
            Em, _ = pot.energy_forces(xm)
            F_fd[i, d] = -(Ep - Em) / (2 * h)
    np.testing.assert_allclose(F, F_fd, atol=1e-5, rtol=1e-4)


# Generic optimizer guarantees, across all cases.
@pytest.mark.parametrize("name,pot,x0", _cases(), ids=CASE_IDS)
def test_energy_is_monotone_non_increasing(name, pot, x0):
    """The line search must reject uphill steps -> no initial jump."""
    _, hist = minimize_with_lbfgs(pot, x0, force_tolerance_kJ_mol_A=TOL, max_steps=200)
    E = np.array([h["energy_kJ_mol"] for h in hist])
    assert np.all(np.diff(E) <= 1e-6), f"energy increased: {np.diff(E)}"


@pytest.mark.parametrize("name,pot,x0", _cases(), ids=CASE_IDS)
def test_converges_below_tolerance(name, pot, x0):
    _, hist = minimize_with_lbfgs(pot, x0, force_tolerance_kJ_mol_A=TOL, max_steps=200)
    assert hist[-1]["max_force_rms"] < TOL
    assert len(hist) < 201            # converged, did not run to the step cap


@pytest.mark.parametrize("name,pot,x0", _cases(), ids=CASE_IDS)
def test_settles_instead_of_drifting(name, pot, x0):
    """As forces vanish the step -> 0, so the final move is tiny (no drift)."""
    _, hist = minimize_with_lbfgs(pot, x0, force_tolerance_kJ_mol_A=TOL, max_steps=200)
    assert hist[-1]["max_step_A"] < 1e-2


# --------------------------------------------------------------------------- #
# Correct minima.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("x0", [
    np.array([[0., 0., 0.], [1.30, 0., 0.]]),
    np.array([[0., 0., 0.], [2.40, 0., 0.]]),
])
def test_lj_dimer_finds_equilibrium_bond_length(x0):
    lj = LennardJones()
    x_opt, _ = minimize_with_lbfgs(lj, x0, force_tolerance_kJ_mol_A=TOL, max_steps=200)
    r = np.linalg.norm(x_opt[0] - x_opt[1])
    assert r == pytest.approx(lj.r_min, abs=1e-3)


def test_morse_finds_equilibrium_bond_length():
    morse = MorseBond(r0=1.42)
    x_opt, _ = minimize_with_lbfgs(morse, np.array([[0., 0., 0.], [1.0, 0., 0.]]),
                                   force_tolerance_kJ_mol_A=TOL, max_steps=200)
    r = np.linalg.norm(x_opt[0] - x_opt[1])
    assert r == pytest.approx(morse.r0, abs=1e-3)


def test_lj_trimer_relaxes_to_equilateral():
    lj = LennardJones()
    x0 = np.array([[0., 0., 0.], [1.4, 0., 0.], [0.7, 1.1, 0.]])
    x_opt, _ = minimize_with_lbfgs(lj, x0, force_tolerance_kJ_mol_A=TOL, max_steps=300)
    d = pair_distances(x_opt)
    assert np.allclose(d, lj.r_min, atol=1e-3)       # all three sides equal r*


def test_aniso_harmonic_reaches_known_minimum_despite_conditioning():
    target = np.array([[0., 0., 0.], [1.3, 0., 0.], [0., 1.1, 0.], [0.5, 0.5, 0.9]])
    harm = AnisotropicHarmonic(target, k=np.array([40.0, 0.4, 40.0, 0.4]))
    x0 = target + 0.3
    x_opt, _ = minimize_with_lbfgs(harm, x0, force_tolerance_kJ_mol_A=TOL, max_steps=200)
    np.testing.assert_allclose(x_opt, target, atol=1e-2)


# --------------------------------------------------------------------------- #
# Bookkeeping / API behavior.
# --------------------------------------------------------------------------- #
def test_history_row_schema_matches_adam():
    lj = LennardJones()
    x0 = np.array([[0., 0., 0.], [1.30, 0., 0.]])
    _, h_lbfgs = minimize_with_lbfgs(lj, x0, force_tolerance_kJ_mol_A=TOL, max_steps=50)
    _, h_adam = minimize_with_adam(lj, x0, force_tolerance_kJ_mol_A=TOL, max_steps=50)
    assert set(h_lbfgs[0]) == set(h_adam[0])
    assert h_lbfgs[0]["step"] == 0


def test_already_minimized_exits_immediately():
    lj = LennardJones()
    x0 = np.array([[0., 0., 0.], [lj.r_min, 0., 0.]])   # start exactly at the minimum
    x_opt, hist = minimize_with_lbfgs(lj, x0, force_tolerance_kJ_mol_A=TOL, max_steps=200)
    assert len(hist) == 1                                # logged step 0, no optimization needed
    np.testing.assert_allclose(x_opt, x0)


def test_dispatcher_routes_to_lbfgs():
    lj = LennardJones()
    x0 = np.array([[0., 0., 0.], [1.30, 0., 0.]])
    x_opt = relaxMolecule(lj, x0, minimizer="lbfgs", force_tol=TOL, max_steps=200)
    assert np.linalg.norm(x_opt[0] - x_opt[1]) == pytest.approx(lj.r_min, abs=1e-3)
    with pytest.raises(ValueError):
        relaxMolecule(lj, x0, minimizer="nonexistent")


def test_lbfgs_needs_fewer_steps_than_adam_on_stiff_problem():
    """The motivation for the switch: on an ill-conditioned well, curvature +
    line search converge in far fewer steps than Adam's fixed-direction step."""
    target = np.array([[0., 0., 0.], [1.3, 0., 0.], [0., 1.1, 0.], [0.5, 0.5, 0.9]])
    harm = AnisotropicHarmonic(target, k=np.array([40.0, 0.4, 40.0, 0.4]))
    x0 = target + 0.3
    _, h_lbfgs = minimize_with_lbfgs(harm, x0, force_tolerance_kJ_mol_A=TOL, max_steps=2000)
    _, h_adam = minimize_with_adam(harm, x0, force_tolerance_kJ_mol_A=TOL, max_steps=2000)
    assert len(h_lbfgs) < len(h_adam)


# --------------------------------------------------------------------------- #
# Real-xTB integration (skipped unless the full xTB/OpenMM stack is installed).
# --------------------------------------------------------------------------- #
def test_real_xtb_h2_relaxes():
    for mod in ("xtb", "ase", "openmm", "openmmxtb"):
        pytest.importorskip(mod)
    from chemdm.xtbSetup import XTBPotential

    Z = np.array([1, 1])
    x0 = np.array([[0., 0., 0.], [0.90, 0., 0.]])       # stretched H2
    xtb = XTBPotential(Z)
    x_opt, hist = minimize_with_lbfgs(xtb, x0, force_tolerance_kJ_mol_A=1.0, max_steps=100)

    E = np.array([h["energy_kJ_mol"] for h in hist])
    assert np.all(np.diff(E) <= 1e-6)                   # monotone, no jump
    assert hist[-1]["max_force_rms"] < 1.0
    r = float(np.linalg.norm(x_opt[0] - x_opt[1]))
    assert 0.60 < r < 0.95                              # near the GFN2-xTB H2 bond length (~0.77 A)
