"""
Diagnostic + regression tests for the Psi4 backends in the force-field
registry (`chemdm.potentials._FORCE_FIELDS`).

Consolidates the two exploratory probes used while adding the Psi4 force
fields, kept here for continuity: re-run after any environment change (e.g.
installing simple-dftd3 / gcp-correction) to see which methods currently have
working analytic gradients.

    # human-readable table of what works in the current env:
    conda run -n chemdm python tests/test_psi4_methods.py

    # as pytest (needs the chemdm env, where psi4 lives):
    conda run -n chemdm python -m pytest tests/test_psi4_methods.py -q

The `-d3bj` and `3c` methods need `simple-dftd3` + `dftd3-python` + `gcp`;
until then they are reported as NO-DISP / skipped rather than failed.
"""
import numpy as np
import pytest

from chemdm.potentialInterface import make_potential, _FORCE_FIELDS

# The psi4-backed entries of the registry, keyed by canonical id.
_PSI4 = {ff["id"]: ff for ff in _FORCE_FIELDS if ff["backend"] == "psi4"}

# Small, distorted, off-axis water: nonzero gradient, not in a standard frame.
Z = np.array([8, 1, 1])
X0 = np.array([[0.10, 0.05, 0.00],
               [0.75, 0.60, 0.30],
               [-0.60, 0.55, -0.20]])

# Cheap bases so the tests stay fast; the registry defaults are larger.
_TEST_BASIS = {"ccsd(t)": "sto-3g"}
_DEFAULT_TEST_BASIS = "def2-svp"


def _build(ff):
    """Construct a registered Psi4 potential with a cheap basis for testing.

    Composite (`3c`) methods carry their own basis (build basis is None), so we
    never override those.
    """
    kw = {"num_threads": 2}
    if _PSI4[ff]["build"].get("basis") is not None:
        kw["basis"] = _TEST_BASIS.get(ff, _DEFAULT_TEST_BASIS)
    return make_potential(ff, Z, **kw)


def _is_missing_dispersion(err):
    msg = str(err).lower()
    return "dftd3" in msg or "gcp" in msg or "s-dftd3" in msg


@pytest.mark.parametrize("ff", sorted(_PSI4))
def test_method_energy_and_forces(ff):
    try:
        e, f = _build(ff).energy_forces(X0)
    except Exception as err:
        if _PSI4[ff].get("needs_dispersion") and _is_missing_dispersion(err):
            pytest.skip(f"{ff} needs simple-dftd3 + dftd3-python + gcp (not installed)")
        raise
    assert np.isfinite(e)
    assert f.shape == (3, 3)
    assert np.all(np.isfinite(f))


def test_numerical_gradient_mp2():
    """Sign/unit check through the registry path on a correlated method."""
    pot = _build("mp2")
    h = 1e-3
    _, f = pot.energy_forces(X0)
    fd = np.zeros_like(X0)
    for i in range(3):
        for k in range(3):
            xp = X0.copy(); xp[i, k] += h
            xm = X0.copy(); xm[i, k] -= h
            fd[i, k] = -(pot.energy_forces(xp)[0] - pot.energy_forces(xm)[0]) / (2 * h)
    assert np.allclose(f, fd, atol=2e-2), np.abs(f - fd).max()


def _diagnostic_table():
    print(f"{'force_field':13} {'status':9} {'energy[eV]':>14} {'|f|max':>10}")
    print("-" * 50)
    for ff in sorted(_PSI4):
        try:
            e, f = _build(ff).energy_forces(X0)
            print(f"{ff:13} {'OK':9} {e:14.4f} {np.abs(f).max():10.4f}")
        except Exception as err:
            no_disp = _PSI4[ff].get("needs_dispersion") and _is_missing_dispersion(err)
            tag = "NO-DISP" if no_disp else "FAIL"
            print(f"{ff:13} {tag:9} {'-':>14} {'-':>10}   "
                  f"({type(err).__name__}: {str(err)[:45]})")


if __name__ == "__main__":
    _diagnostic_table()
