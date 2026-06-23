"""Does the optimizer cause NEB resolution-inconsistency, and does FIRE fix it?

Run CI-NEB (climbing image on) at 20 and 100 images with Adam and with FIRE, and
report, for each optimizer, the converged-path RMSD (per-image Kabsch-aligned, so
rigid drift between resolutions is removed) and the barrier at both resolutions.
Springs are uniform (dk=0); this is the production climbing-image setting minus
the variable springs.

Read-out:
  RMSD small AND barriers agree  -> resolution-consistent (the climbing image
                                    converges the saddle/barrier at both counts).
  large RMSD or barrier gap      -> resolution-dependent; chase the cause.

NOTE: the worker pool uses spawn, which re-imports this file in each child, so
the run loop MUST sit under `if __name__ == "__main__":` -- otherwise every child
re-runs it and recursively spawns pools.
"""
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from rdkit import Chem

from chemdm.Constants import KJ_MOL_PER_EV
from chemdm.geometry import kabsch_align_numpy
from chemdm.path_smoothing import smooth_path_penalized_least_squares
from chemdm.nebXtbDirect import run_neb_xtb, neb_adam, init_xtb_worker, evaluate_path_process_parallel
from chemdm.commands.transition_path import load_attention_model, _ml_initial_guess, cleanupPath

ALPHA = 0.02
K = 10.0 * KJ_MOL_PER_EV
FORCE_TOL = 0.1       # kJ/mol/A
N_STEPS = 2000
LR = 1e-2             # Adam only; run_neb_xtb (FIRE) takes no learning rate
MAX_WORKERS = 8
RESOLUTIONS = [20, 100]
OPTIMIZERS = ["adam", "fire"]
CONF_DIR = Path(__file__).resolve().parents[1] / "rdkit" / "conformers_ad"


def load_mol(path):
    mol = Chem.MolFromMolFile( str(path), removeHs=False ) # type: ignore
    Z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int64)
    x = mol.GetConformer().GetPositions().astype(float)
    bonds = [pair for b in mol.GetBonds()
             for pair in ([b.GetBeginAtomIdx(), b.GetEndAtomIdx()],
                          [b.GetEndAtomIdx(), b.GetBeginAtomIdx()])]
    return Z, x, np.array(bonds, dtype=np.int64)


def smoothed_guess(model, Z, xA, xB, GA, GB, n):
    raw, s0 = _ml_initial_guess(model, Z, xA, xB, GA, GB, n)
    if np.sum(Z != 1) > 6:
        raw = cleanupPath(Z, raw, s0, GA, GB)
    return smooth_path_penalized_least_squares(raw, ALPHA)


def resample( path, n=200 ):
    flat = path.reshape(len(path), -1)
    d = np.linalg.norm(np.diff(flat, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    s /= s[-1]
    grid = np.linspace(0.0, 1.0, n)
    return np.stack([np.interp(grid, s, flat[:, j]) for j in range(flat.shape[1])], axis=1)


def rmsd(a, b, n=200):
    A = resample(a, n).reshape(n, -1, 3)
    B = resample(b, n).reshape(n, -1, 3)
    # NEB only pins the endpoints, so interior images can relax into different
    # rigid frames between resolutions. Kabsch-align each image pair to strip
    # that translation/rotation gauge and compare internal geometry only.
    B = np.stack([ kabsch_align_numpy(B[i], A[i]) for i in range(n) ])
    return float(np.sqrt(np.mean(np.sum((A - B) ** 2, axis=2))))


def run_optimizer( opt, Z, path0 ):
    """Converge one band and return (path, barrier_kJ_mol). FIRE goes through
    run_neb_xtb; Adam is driven via neb_adam on its own spawn pool (run_neb_xtb
    is FIRE-only now). Both return path energies as their 2nd value."""
    if opt == "fire":
        path_opt, E_path, _ = run_neb_xtb( Z, path0, N_STEPS, K, 0.02, FORCE_TOL,
                                           max_workers=MAX_WORKERS, dk=0.0, climb=True )
    else:
        # Same spawn pool run_neb_xtb sets up internally, but driving neb_adam.
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor( max_workers=MAX_WORKERS, mp_context=ctx,
                                  initializer=init_xtb_worker, initargs=(Z,) ) as pool:
            neb_ef = lambda p: evaluate_path_process_parallel( p, pool )
            path_opt, E_path, _ = neb_adam( neb_ef, path0, N_STEPS, LR, K, 0.02, FORCE_TOL,
                                            dk=0.0, climb=True )
    return path_opt, float( np.max(E_path) - E_path[0] )


if __name__ == "__main__":
    Z, xA, GA = load_mol(CONF_DIR / "conformer_0.mol")
    _, xB, GB = load_mol(CONF_DIR / "conformer_2.mol")
    xB = kabsch_align_numpy(xB, xA, Z)

    model = load_attention_model()
    guesses = {n: smoothed_guess(model, Z, xA, xB, GA, GB, n) for n in RESOLUTIONS}

    print("\nConverged-path RMSD between resolutions (uniform springs, climbing image on):")
    for opt in OPTIMIZERS:
        results = { n: run_optimizer(opt, Z, guesses[n]) for n in RESOLUTIONS }   # n -> (path, barrier)
        r = rmsd( results[RESOLUTIONS[0]][0], results[RESOLUTIONS[-1]][0] )
        barriers = "  ".join( f"{n}img {results[n][1]:.2f}" for n in RESOLUTIONS )
        print(f"  {opt:5s}:  {RESOLUTIONS[0]}-vs-{RESOLUTIONS[-1]} RMSD = {r:.4f} A  |  barrier kJ/mol: {barriers}")
