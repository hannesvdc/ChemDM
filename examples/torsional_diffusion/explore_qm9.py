"""
Walk the GEOM-QM9 directory and report dataset statistics:

    - number of molecules
    - distribution of n_atoms (with-H), n_heavy_atoms
    - distribution of n_rotatable_bonds (paper definition)
    - distribution of n_unique_conformers per molecule
    - element coverage (atomic numbers actually present)
    - count of molecules with ≥ 1 rotatable bond (i.e. trainable for our model;
      molecules with 0 rotatable bonds contribute nothing to torsional
      diffusion training and should be filtered out)

Run:
    /opt/homebrew/anaconda3/envs/py311/bin/python explore_qm9.py
    /opt/homebrew/anaconda3/envs/py311/bin/python explore_qm9.py --max-files 2000
    /opt/homebrew/anaconda3/envs/py311/bin/python explore_qm9.py --hist-bins 20

If you want to also see a few example molecules with various sizes:
    /opt/homebrew/anaconda3/envs/py311/bin/python explore_qm9.py --show-examples 5
"""

from __future__ import annotations

import argparse
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np

from qm9_parser import load_qm9_molecule

from chemdm.util import Z_TO_SYMBOL

from dotenv import load_dotenv

def _histogram( values: list[int] ) -> str:
    """One bin per integer value — these counts are always integer-valued."""
    if not values:
        return "(no data)"
    a = np.asarray(values)
    edges = np.arange(int(a.min()), int(a.max()) + 2)
    counts, _ = np.histogram( a, bins=edges )
    peak = max( counts.max(), 1 )
    width = 40
    lines = []
    for c, lo in zip(counts, edges[:-1]):
        bar = "#" * int(width * c / peak)
        lines.append(f"  {int(lo):>4d}  {c:>7d}  {bar}")
    return "\n".join(lines)


def explore( qm9_dir: Path, max_files: int | None = None, show_examples: int = 0 ) -> None:
    
    # Cheap step. Sorting SMILES strings does not make sense, 
    # but keeps the same ordering as the OSX file viewer.
    # We sample randomly afterwards to get a good grasp of the full dataset.
    files = sorted( f for f in os.listdir(qm9_dir) if f.endswith(".pickle") )
    max_files = min( max_files, len(files) ) if max_files is not None else len( files )
    rng = np.random.default_rng()
    files = rng.choice( files, size=max_files, replace=False )
    print(f"Found {len(files):,} pickle files in {qm9_dir}\n")

    n_atoms: list[int] = []
    n_heavy: list[int] = []
    n_rot:   list[int] = []
    n_conf:  list[int] = []
    element_counts: Counter[int] = Counter()
    n_zero_rot = 0
    n_skipped = 0
    examples: list[tuple[int, int, int, str]] = []

    t0 = time.time()
    for i, fn in enumerate(files):
        if i and i % 5000 == 0:
            dt = time.time() - t0
            rate = i / dt
            eta = (len(files) - i) / rate
            print(f"  ... processed {i:>7,d}/{len(files):,}  ({rate:.0f} mol/s, eta {eta:.0f}s)")
        path = qm9_dir / fn
        try:
            d = load_qm9_molecule( path )
        except Exception as e:
            n_skipped += 1
            if n_skipped <= 5:
                print(f"  skip {fn}: {e}")
            continue

        N = int( d.Z.shape[0] )
        nh = int( (d.Z != 1).sum().item() )
        nr = int( d.bonds.shape[0] )
        nc = len( d.conformers )

        n_atoms.append( N )
        n_heavy.append( nh )
        n_rot.append( nr )
        n_conf.append( nc )
        for z in d.Z.tolist():
            element_counts[z] += 1
        if nr == 0:
            n_zero_rot += 1

        examples.append( (N, nh, nr, d.smiles) )

    n = len(n_atoms)
    print(f"\n=== summary ({n:,} molecules, {n_skipped} skipped) ===")
    if n == 0:
        return

    def stats( name: str, vals: list[int] ):
        a = np.asarray(vals)
        print( f"  {name:24s} min={a.min():>5d}  max={a.max():>6d} mean={a.mean():7.2f}  median={int(np.median(a)):>4d}" )

    stats( "n_atoms (with H)", n_atoms)
    stats( "n_heavy_atoms", n_heavy)
    stats( "n_rotatable_bonds", n_rot)
    stats( "n_unique_conformers", n_conf)

    pct = 100.0 * n_zero_rot / n
    print( f"\n  molecules with 0 rotatable bonds: {n_zero_rot:,}/{n:,} ({pct:.1f}%)" )
    print( f"  -> trainable for torsional diffusion: {n - n_zero_rot:,} ({100 - pct:.1f}%)" )

    print( f"\n  elements present (atomic number -> count):" )
    for z, c in sorted(element_counts.items()):
        sym = Z_TO_SYMBOL.get(z, f"Z={z}")
        print(f"    {z:>3d} ({sym:>2s}): {c:>10,d}")

    print( f"\n  histogram of n_rotatable_bonds:" )
    print( _histogram(n_rot) )

    # Cap aggressively — the conformer distribution is long-tailed (max can be
    # in the thousands) and we want one short histogram, not hundreds of bins.
    # The last row aggregates every molecule with >= cap conformers.
    cap = 30
    n_conf_capped = [ min(c, cap) for c in n_conf ]
    print( f"\n  histogram of n_unique_conformers ({cap}+ aggregated into the last row):" )
    print( _histogram(n_conf_capped) )

    total_examples = sum( n_conf[i] for i in range(n) if n_rot[i] > 0 )
    print( f"\n  total trainable (mol, conformer) pairs: {total_examples:,}" )

    if show_examples > 0:
        print( f"\n  example molecules sorted by n_rotatable_bonds (descending):" )
        ex = sorted( examples, key=lambda t: -t[2] )
        for N, nh, nr, smi in ex[:show_examples]:
            print( f"    N={N:>2d} heavy={nh:>2d} rot={nr:>2d}  {smi}" )

    _plot_histograms( n_rot, n_conf_capped, cap_conf=cap, n_sample=n )


def _plot_histograms( n_rot: list[int], n_conf_capped: list[int], cap_conf: int, n_sample: int ) -> None:
    """Side-by-side bar charts for the two integer distributions."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots( 1, 2, figsize=(12, 4) )

    def bar_int( ax, values: list[int], title: str, xlabel: str ):
        a = np.asarray(values)
        edges = np.arange( int(a.min()), int(a.max()) + 2 )
        counts, _ = np.histogram( a, bins=edges )
        ax.bar( edges[:-1], counts, width=1.0, edgecolor="black", align="center" )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("# molecules")
        ax.set_title(title)
        if len(edges) <= 32:
            ax.set_xticks(edges[:-1])
        ax.grid(axis="y", alpha=0.3)

    bar_int( axes[0], n_rot,
             title="n_rotatable_bonds",
             xlabel="n_rotatable_bonds" )
    bar_int( axes[1], n_conf_capped,
             title=f"n_unique_conformers ({cap_conf}+ aggregated)",
             xlabel="n_unique_conformers" )

    fig.suptitle( f"GEOM-QM9 sample of {n_sample:,} molecules" )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    load_dotenv()
    qm9_folder = Path( os.environ["QM9_FOLDER"] )

    p = argparse.ArgumentParser()
    p.add_argument( "--max-files", type=int, default=None,
                   help="if set, only inspect the first N files (alphabetical)")
    p.add_argument( "--show-examples", type=int, default=10,
                   help="how many example molecules (sorted by n_rotatable) to list at the end")
    args = p.parse_args()

    explore( qm9_folder, max_files=args.max_files, show_examples=args.show_examples )