"""Is the NN initial guess resolution-independent?

Build the raw NN path and the smoothed NN path at n_images=20 and n_images=100
for one reaction, resample both onto a common arclength grid, and report the
RMSD. Raw and smoothed should each be (nearly) the same path at both resolutions
-- if so, the NN and the smoothing are sound, and any post-NEB difference is the
optimizer's fault. This isolates the two stages.
"""
from pathlib import Path

import numpy as np
from rdkit import Chem

from chemdm.geometry import kabsch_align_numpy
from chemdm.path_smoothing import smooth_path_penalized_least_squares
from chemdm.commands.transition_path import load_attention_model, _ml_initial_guess, cleanupPath

ALPHA = 0.02
CONF_DIR = Path(__file__).resolve().parents[1] / "rdkit" / "conformers_ad"
REACTANT = CONF_DIR / "conformer_2.mol"
PRODUCT = CONF_DIR / "conformer_1.mol"


def load_mol( path ):
    """Atomic numbers (n,), coordinates (n,3) Angstrom, and bidirectional bond
    edge list (E,2) from a .mol file."""
    mol = Chem.MolFromMolFile( str(path), removeHs=False )
    Z = np.array( [a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int64 )
    x = mol.GetConformer().GetPositions().astype(float)
    bonds = [pair for b in mol.GetBonds()
             for pair in ([b.GetBeginAtomIdx(), b.GetEndAtomIdx()],
                          [b.GetEndAtomIdx(), b.GetBeginAtomIdx()])]
    return Z, x, np.array(bonds, dtype=np.int64)


def guesses( model, Z, xA, xB, GA, GB, n ):
    """Raw NN path and smoothed NN path for n images."""
    raw, s0 = _ml_initial_guess( model, Z, xA, xB, GA, GB, n )
    path = raw.copy()
    if np.sum(Z != 1) > 6:
        path = cleanupPath( Z, path, s0, GA, GB )
    return raw, smooth_path_penalized_least_squares( path, ALPHA )


def resample( path, n=200 ):
    """Resample a path onto n points evenly spaced in normalized arclength."""
    flat = path.reshape(len(path), -1)
    d = np.linalg.norm(np.diff(flat, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    s /= s[-1]
    grid = np.linspace(0.0, 1.0, n)
    return np.stack([np.interp(grid, s, flat[:, j]) for j in range(flat.shape[1])], axis=1)


def rmsd( a, b, n=200 ):
    """Per-atom RMSD between two paths, on a common arclength grid."""
    A = resample(a, n).reshape(n, -1, 3)
    B = resample(b, n).reshape(n, -1, 3)
    return float(np.sqrt( np.mean( np.sum((A - B)**2, axis=2) ) ))


Z, xA, GA = load_mol( REACTANT )
_, xB, GB = load_mol( PRODUCT )
xB = kabsch_align_numpy( xB, xA, Z )

model = load_attention_model( )
raw20, smooth20 = guesses( model, Z, xA, xB, GA, GB, 20 )
raw50, smooth50 = guesses( model, Z, xA, xB, GA, GB, 50 )
raw100, smooth100 = guesses( model, Z, xA, xB, GA, GB, 100 )

print(f"raw NN path RMSD (20 vs 100):      {rmsd(raw20, raw100):.4f} A")
print(f"raw NN path RMSD (50 vs 100):      {rmsd(raw50, raw100):.4f} A")
print(f"smoothed NN path RMSD (20 vs 100): {rmsd(smooth20, smooth100):.4f} A")
print(f"smoothed NN path RMSD (50 vs 100): {rmsd(smooth50, smooth100):.4f} A")