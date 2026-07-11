import torch as pt

from dataclasses import dataclass

from rdkit import Chem


@dataclass
class TorsionalDiffusionData:
    smiles: str
    Z: pt.Tensor                 # (N,)        long
    edge_index: pt.Tensor        # (E, 2)      long — undirected covalent bonds, both directions
    rotatable_bonds: pt.Tensor   # (m, 2)      long — rotatable bonds, canonical b<c ordering
    side_atom_idx: pt.Tensor     # (P,)        long — atoms on the c-side of some bond
    side_bond_idx: pt.Tensor     # (P,)        long — which bond (0..m-1) each side-atom belongs to
    conformers: list[dict]       # each: {x: (N, 3) float64, totalenergy, relativeenergy, boltzmannweight}


def find_rotatable_bonds( mol: Chem.Mol ) -> tuple[ list[tuple[int, int]], list[list[int]] ]:
    """Rotatable bonds by the torsional-diffusion paper definition: severing the
    bond splits the molecular graph into exactly two connected components, each
    with >= 2 atoms. Equivalently, the bond is a *bridge* whose two sides are both
    non-terminal. This includes double bonds and excludes ring bonds and terminal
    bonds (C-H, terminal methyl). Pure BFS -- no networkx dependency.

    Returns
    -------
    rotatable : list of (b, c) atom-index pairs, canonical b < c.
    sides     : sides[i] = sorted atom indices on the c-side of bond i (the
                component containing c after severance).
    """
    n = mol.GetNumAtoms()
    adj: list[list[int]] = [ [] for _ in range( n ) ]
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        adj[i].append( j )
        adj[j].append( i )

    def reachable( start: int, cut: frozenset ) -> set[int]:
        """Atoms reachable from `start` without traversing the severed bond `cut`."""
        seen = { start }
        stack = [ start ]
        while stack:
            a = stack.pop()
            for nb in adj[a]:
                if frozenset( (a, nb) ) == cut or nb in seen:
                    continue
                seen.add( nb )
                stack.append( nb )
        return seen

    rotatable: list[tuple[int, int]] = []
    sides:     list[list[int]]       = []
    for bd in mol.GetBonds():
        u, v = bd.GetBeginAtomIdx(), bd.GetEndAtomIdx()
        cut = frozenset( (u, v) )
        u_side = reachable( u, cut )
        if v in u_side:
            continue                                   # still connected -> ring bond, not a bridge
        v_side = reachable( v, cut )
        if len( u_side ) >= 2 and len( v_side ) >= 2:
            b, c = (u, v) if u < v else (v, u)
            rotatable.append( (b, c) )
            c_side = v_side if c == v else u_side       # the component containing c
            sides.append( sorted( c_side ) )
    return rotatable, sides


def build_torsional_data( mol: Chem.Mol ) -> TorsionalDiffusionData:
    """Build the topology a `TorsionalDiffusionData` needs -- Z, undirected edges
    (both directions), rotatable bonds and their c-side atom masks -- from an RDKit
    molecule. The molecule should already carry explicit hydrogens (Chem.AddHs) so
    the atom indexing matches the embedded backbones. `conformers` is left empty:
    sampling never reads it (only the QM9 eval path needs the ground-truth ensemble).
    """
    Z = pt.tensor( [ a.GetAtomicNum() for a in mol.GetAtoms() ], dtype=pt.long )

    # Covalent bonds as directed edges, both directions -- shape (E, 2), the same
    # layout as `rotatable_bonds`.
    edges = []
    for b in mol.GetBonds():
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        edges.append( (u, v) )
        edges.append( (v, u) )
    edge_index = pt.tensor( edges, dtype=pt.long ).reshape( -1, 2 )

    rot, sides = find_rotatable_bonds( mol )
    rotatable_bonds = pt.tensor( rot, dtype=pt.long ).reshape( -1, 2 )

    # COO flatten of the per-bond c-side atom sets.
    side_atom_flat: list[int] = []
    side_bond_flat: list[int] = []
    for i, atoms in enumerate( sides ):
        side_atom_flat.extend( atoms )
        side_bond_flat.extend( [ i ] * len( atoms ) )
    side_atom_idx = pt.tensor( side_atom_flat, dtype=pt.long )
    side_bond_idx = pt.tensor( side_bond_flat, dtype=pt.long )

    return TorsionalDiffusionData(
        smiles=Chem.MolToSmiles( mol ),
        Z=Z,
        edge_index=edge_index,
        rotatable_bonds=rotatable_bonds,
        side_atom_idx=side_atom_idx,
        side_bond_idx=side_bond_idx,
        conformers=[],
    )