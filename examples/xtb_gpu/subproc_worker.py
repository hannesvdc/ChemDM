"""
Standalone single-backend evaluator, invoked via ``subprocess.run`` by
``check_correspondence.py``.

Why a separate process per backend: xtb-python links conda's OpenMP while the
pip ``tblite`` wheel vendors its own ``libomp``. Loading both in one process
aborts (OMP #15), segfaults, or -- under multiprocessing.spawn -- deadlocks.
Running each backend in its own fresh top-level interpreter that imports EXACTLY
ONE backend sidesteps the clash entirely (and mirrors production, where xtb runs
in its own pool processes).

For the same reason the tblite path must NOT import ``TBLitePotential`` (it
subclasses ``XTBPotential`` and would pull in xtb+openmm). It builds the tblite
ASE calculator directly with the identical kwargs and the identical
energy/force convention, so the numbers match ``TBLitePotential`` exactly.

Protocol: read a pickled job {backend, kwargs, geometries:[(Z, x_A), ...]},
write a pickled list of (energy_eV, forces_eV_A).
"""

import pickle
import sys

import numpy as np


def _tblite_eval( kwargs : dict, Z : np.ndarray, x_A : np.ndarray ):
    from ase import Atoms
    from tblite.ase import TBLite

    # Mirror TBLitePotential: uhf (unpaired electrons) -> multiplicity (2S+1).
    tb_kwargs = dict( method=kwargs["method"], charge=kwargs.get("charge", 0),
                      multiplicity=kwargs.get("uhf", 0) + 1, accuracy=kwargs["accuracy"],
                      electronic_temperature=kwargs["electronic_temperature"],
                      max_iterations=kwargs["max_iterations"], verbosity=0 )
    atoms = Atoms( numbers=np.asarray(Z, dtype=int), positions=np.asarray(x_A, dtype=float) )
    atoms.calc = TBLite( **tb_kwargs )
    e = float( atoms.get_potential_energy() )
    f = np.asarray( atoms.get_forces(), dtype=float )
    return e, f


def _xtb_eval( kwargs : dict, Z : np.ndarray, x_A : np.ndarray ):
    from chemdm.xtbSetup import XTBPotential

    pot = XTBPotential( Z=np.asarray(Z, dtype=int), **kwargs )
    e, f = pot.energy_forces( np.asarray(x_A, dtype=float) )
    return float(e), np.asarray(f, dtype=float)


def _dxtb_eval( kwargs : dict, Z : np.ndarray, x_A : np.ndarray ):
    from dxtb_potential import DxtbPotential

    pot = DxtbPotential( Z=np.asarray(Z, dtype=int), **kwargs )
    e, f = pot.energy_forces( np.asarray(x_A, dtype=float) )
    return float(e), np.asarray(f, dtype=float)


_EVAL = { "xtb": _xtb_eval, "tblite": _tblite_eval, "dxtb": _dxtb_eval }


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]
    with open(in_path, "rb") as fh:
        job = pickle.load(fh)

    fn = _EVAL[ job["backend"] ]
    results = [ fn(job["kwargs"], Z, x_A) for (Z, x_A) in job["geometries"] ]

    with open(out_path, "wb") as fh:
        pickle.dump(results, fh)


if __name__ == "__main__":
    main()
