"""
Tests for `chemdm.TorsionalDiffusionData.find_rotatable_bonds` (+ `build_torsional_data`).

`find_rotatable_bonds` identifies the freely-rotatable bonds of a molecule by the
torsional-diffusion paper definition (Sec. 3): a bond is rotatable iff severing it
splits the molecular graph into exactly two connected components, each with >= 2
atoms. This includes double bonds and excludes ring bonds and terminal bonds
(C-H, terminal methyl-hydrogen). The library implementation is a dependency-free
BFS; the original implementation used networkx.

The tests:

    1.  Differential parity vs an independent networkx oracle (the original
        qm9_parser implementation) over a broad SMILES set -- rotatable bonds AND
        their c-side atom sets must match exactly (compared order-independently,
        since networkx and RDKit iterate edges in different orders).
    2.  Known rotatable-bond counts (ethane=1, propane=2, butane=3, benzene=0, ...).
    3.  Ring / fused-ring / aromatic bonds are excluded.
    4.  Per-bond invariants: canonical b<c ordering; the c-side contains c, excludes
        b, has >= 2 atoms, and is sorted/unique; the other side also has >= 2 atoms.
    5.  `build_torsional_data`: symmetric edge_index, aligned/valid COO side arrays,
        canonical rotatable bonds, and the rigid (0-rotatable) molecule case.
    6.  Documented divergence on disconnected inputs (salts) -- the BFS reports the
        real per-fragment bonds, networkx rejects all; single-molecule conformer
        generation is always connected, so this never bites.

Once this passes, the networkx `find_rotatable_bonds` in `examples/torsional_diffusion/
qm9_parser.py` is redundant and can import the library version instead.
"""

import networkx as nx
import pytest
from rdkit import Chem

from chemdm.TorsionalDiffusionData import find_rotatable_bonds, build_torsional_data


# networkx reference oracle (original qm9_parser implementation)
def nx_find_rotatable_bonds(mol):
    """Reference: sever each bond, keep it iff the graph splits into exactly two
    components each with >= 2 atoms. Returns (rotatable [(b, c), b < c], sides)."""
    G = nx.Graph()
    G.add_nodes_from(range(mol.GetNumAtoms()))
    for b in mol.GetBonds():
        G.add_edge(b.GetBeginAtomIdx(), b.GetEndAtomIdx())

    rotatable, sides = [], []
    for u, v in list(G.edges()):
        G.remove_edge(u, v)
        comps = list(nx.connected_components(G))
        if len(comps) == 2 and all(len(c) >= 2 for c in comps):
            b, c = (u, v) if u < v else (v, u)
            rotatable.append((b, c))
            c_side = comps[0] if c in comps[0] else comps[1]
            sides.append(sorted(c_side))
        G.add_edge(u, v)
    return rotatable, sides


def mol_from_smiles(smi):
    m = Chem.MolFromSmiles(smi)
    assert m is not None, f"bad SMILES {smi!r}"
    return Chem.AddHs(m)


def as_bond_side_map(rot, sides):
    """{(b, c): tuple(c-side atoms)} -- order-independent view for comparison."""
    return {tuple(bc): tuple(s) for bc, s in zip(rot, sides)}


# Broad, diverse set of CONNECTED molecules.
PARITY_SMILES = [
    "C", "O", "CC", "CCC", "CCCC", "CCCCCC", "CC(C)(C)C",        # alkanes / branched
    "CCO", "OCC(O)CO", "CC(C)CO",                               # alcohols
    "CN", "CCN", "CN(C)CCO",                                    # amines
    "c1ccccc1", "Cc1ccccc1", "CCc1ccccc1", "c1ccncc1",          # aromatics / heteroaromatic
    "C1CCCCC1", "C1CCCCCC1", "C1CCC(CC1)CC",                    # saturated rings + substituent
    "c1ccc2ccccc2c1",                                          # naphthalene (fused)
    "CC(=O)Oc1ccccc1C(=O)O",                                   # aspirin
    "Cn1cnc2c1c(=O)n(C)c(=O)n2C",                              # caffeine
    "FC(F)(F)c1ccccc1", "CCOC(=O)C", "O=C(O)CCC(=O)O", "CC(=O)Nc1ccccc1",
]


@pytest.mark.parametrize("smi", PARITY_SMILES)
def test_parity_with_networkx(smi):
    """BFS library impl == networkx oracle (rotatable bonds AND c-side sets)."""
    m = mol_from_smiles(smi)
    lib = as_bond_side_map(*find_rotatable_bonds(m))
    ref = as_bond_side_map(*nx_find_rotatable_bonds(m))
    assert lib == ref, f"{smi}: BFS vs networkx disagree"


KNOWN_COUNTS = [
    ("C", 0),                         # methane: only terminal C-H bonds
    ("O", 0),                         # water
    ("CC", 1),                        # ethane: C-C (each side a methyl = 4 atoms)
    ("CCC", 2),                       # propane
    ("CCCC", 3),                      # butane
    ("CCCCCC", 5),                    # hexane
    ("CCO", 2),                       # ethanol: C-C and C-O
    ("c1ccccc1", 0),                  # benzene
    ("C1CCCCC1", 0),                  # cyclohexane
    ("Cc1ccccc1", 1),                 # toluene: only the CH3-ring bond
    ("CC(=O)Oc1ccccc1C(=O)O", 5),     # aspirin
]


@pytest.mark.parametrize("smi,n_expected", KNOWN_COUNTS)
def test_known_counts(smi, n_expected):
    rot, _ = find_rotatable_bonds(mol_from_smiles(smi))
    assert len(rot) == n_expected, f"{smi}: expected {n_expected}, got {len(rot)}"


@pytest.mark.parametrize("smi", ["c1ccccc1", "C1CCCCC1", "c1ccncc1", "c1ccc2ccccc2c1"])
def test_ring_bonds_excluded(smi):
    """Pure-ring / fused-ring / aromatic molecules have no rotatable bonds."""
    assert find_rotatable_bonds(mol_from_smiles(smi))[0] == []


@pytest.mark.parametrize("smi", PARITY_SMILES)
def test_rotatable_bond_invariants(smi):
    """Structural invariants of every returned rotatable bond + its c-side."""
    m = mol_from_smiles(smi)
    n = m.GetNumAtoms()
    rot, sides = find_rotatable_bonds(m)
    assert len(rot) == len(sides)
    for (b, c), side in zip(rot, sides):
        assert b < c, "bonds must be canonical b < c"
        assert c in side and b not in side, "c-side must contain c, not b"
        assert len(side) >= 2, "c-side must have >= 2 atoms"
        assert n - len(side) >= 2, "the b-side must also have >= 2 atoms"
        assert side == sorted(side) and len(set(side)) == len(side), "sorted & unique"
        assert all(0 <= a < n for a in side)


def test_build_torsional_data_structure():
    """build_torsional_data: symmetric edges, aligned/valid COO side arrays."""
    m = mol_from_smiles("CC(=O)Oc1ccccc1C(=O)O")   # aspirin: rings + rotatable
    d = build_torsional_data(m)
    n = m.GetNumAtoms()
    m_rot = d.rotatable_bonds.shape[0]

    assert d.Z.shape == (n,)
    edges = {tuple(e) for e in d.edge_index.tolist()}
    assert d.edge_index.shape == (2 * m.GetNumBonds(), 2)          # both directions
    assert all((v, u) in edges for (u, v) in edges), "edge_index not symmetric"

    assert d.rotatable_bonds.shape == (m_rot, 2)
    assert (d.rotatable_bonds[:, 0] < d.rotatable_bonds[:, 1]).all(), "canonical b<c"
    assert d.side_atom_idx.shape == d.side_bond_idx.shape          # COO aligned
    assert int(d.side_bond_idx.min()) == 0
    assert int(d.side_bond_idx.max()) == m_rot - 1                 # every bond has side atoms
    assert (d.side_atom_idx >= 0).all() and (d.side_atom_idx < n).all()


def test_rigid_molecule_empty_tensors():
    """A molecule with no rotatable bonds yields empty (0,*) tensors, not errors."""
    d = build_torsional_data(mol_from_smiles("c1ccccc1"))          # benzene
    assert d.rotatable_bonds.shape == (0, 2)
    assert d.side_atom_idx.shape == (0,)
    assert d.side_bond_idx.shape == (0,)


def test_disconnected_divergence():
    """Documented edge case: on a DISCONNECTED input the BFS reports the real
    per-fragment rotatable bonds while networkx rejects all (>2 total components).
    Single-molecule conformer generation is always connected, so this never bites."""
    m = mol_from_smiles("CCCC.CCCC")                              # two butanes
    lib = find_rotatable_bonds(m)[0]
    ref = nx_find_rotatable_bonds(m)[0]
    assert len(lib) == 6                                          # 3 per butane
    assert len(ref) == 0                                         # networkx: never exactly 2 comps
