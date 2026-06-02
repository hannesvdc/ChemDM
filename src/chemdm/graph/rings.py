from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import torch as pt

from chemdm.MoleculeGraph import Molecule


@dataclass(frozen=True)
class RingInfo:
    """
    Ring information for one Molecule bond graph.

    atom_in_ring:
        Bool tensor of shape (N,).

    atom_ring_count:
        Long tensor of shape (N,). Number of detected rings containing atom. Can be larger than one for fused rings.

    atom_ring_sizes:
        Python list of length N. atom_ring_sizes[i] is a set of ring sizes.

    rings:
        List of detected rings, each represented as a tuple of atom indices.
    """

    atom_in_ring: pt.Tensor
    atom_ring_count: pt.Tensor
    atom_ring_sizes: list[set[int]]
    rings: list[tuple[int, ...]]


def _unique_undirected_edges(edge_index: pt.Tensor) -> list[tuple[int, int]]:
    """
    Convert directed edge_index of shape (E, 2) to unique undirected edges.
    """
    edges: set[tuple[int, int]] = set()

    for i, j in edge_index.detach().cpu().long().tolist():
        if i == j:
            continue

        a, b = (i, j) if i < j else (j, i)
        edges.add( (a, b) )

    return sorted(edges)


def _neighbors_per_atom( n_atoms: int, edges: list[tuple[int, int]] ) -> list[set[int]]:
    adj = [set() for _ in range(n_atoms)]

    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)

    return adj


def _shortest_path_without_edge(
    adj: list[set[int]],
    start: int,
    target: int,
    removed_edge: tuple[int, int],
) -> list[int] | None:
    """
    Find shortest path from start to target while ignoring removed_edge.

    Returns the path as atom indices [start, ..., target], or None.
    """
    a, b = removed_edge

    queue = deque([start])
    parent: dict[int, int | None] = {start: None}

    while queue:
        u = queue.popleft()

        if u == target:
            break

        for v in adj[u]:
            if (u == a and v == b) or (u == b and v == a):
                continue

            if v in parent:
                continue

            parent[v] = u
            queue.append(v)

    if target not in parent:
        return None

    path = [target]
    while parent[path[-1]] is not None:
        path.append(parent[path[-1]])  # type: ignore[arg-type]

    path.reverse()
    return path


def _canonical_ring( ring: list[int] ) -> tuple[int, ...]:
    """
    Canonicalize a ring so duplicates collapse.

    The ring is treated as cyclic and orientation-free.

    Runtime complexity is O(k^2) with k the ring size.
    """
    if len(ring) < 3:
        return tuple()

    n = len(ring)
    candidates: list[tuple[int, ...]] = []

    for xs in (ring, list(reversed(ring))):
        for k in range(n):
            candidates.append(tuple(xs[k:] + xs[:k]))

    return min(candidates)


def detect_ring_info(molecule: Molecule) -> RingInfo:
    """
    Detect rings in a Molecule from its directed bond graph. This is the general-purpose
    entry point for ring detection and ring information calculations.

    This uses a simple shortest-cycle-per-bond strategy:
        for each bond i-j:
            remove i-j
            find shortest path i -> j
            path + i-j is a ring

    This is simple, robust for small molecular graphs, and fast enough to
    precompute once per molecule/endpoint.
    """
    n_atoms = int( molecule.Z.shape[0] )
    device = molecule.Z.device

    edges = _unique_undirected_edges(molecule.edge_index)
    adj = _neighbors_per_atom(n_atoms, edges)

    # Ring detection. If a path between i and j exists while the bond i <-> j is removed,
    # a ring must exist!
    rings_set: set[tuple[int, ...]] = set()
    for i, j in edges:
        path = _shortest_path_without_edge( adj, start=i, target=j, removed_edge=(i, j) )

        if path is None:
            continue

        # path already contains i ... j. Adding edge j-i closes the ring.
        ring = _canonical_ring( path )

        if len(ring) >= 3:
            rings_set.add(ring)

    rings = sorted(rings_set, key=lambda r: (len(r), r))

    # Calculate all the ring info
    atom_ring_sizes: list[set[int]] = [set() for _ in range(n_atoms)]
    atom_ring_count_py = [0 for _ in range(n_atoms)]
    for ring in rings:
        ring_size = len(ring)

        for atom_idx in ring:
            atom_ring_sizes[atom_idx].add(ring_size)
            atom_ring_count_py[atom_idx] += 1
    atom_in_ring = pt.tensor( [len(sizes) > 0 for sizes in atom_ring_sizes],  dtype=pt.bool, device=device )
    atom_ring_count = pt.tensor( atom_ring_count_py, dtype=pt.long, device=device )

    return RingInfo(
        atom_in_ring=atom_in_ring,
        atom_ring_count=atom_ring_count,
        atom_ring_sizes=atom_ring_sizes,
        rings=rings,
    )


def atom_ring_size_flags(
    ring_info: RingInfo,
    ring_sizes: tuple[int, ...] = (3, 4, 5, 6, 7),
) -> pt.Tensor:
    """
    Convert atom ring-size sets to multi-hot flags.

    Returns shape:
        (N, len(ring_sizes))
    """
    flags = [
        [size in sizes for size in ring_sizes]
        for sizes in ring_info.atom_ring_sizes
    ]

    return pt.tensor(
        flags,
        dtype=pt.bool,
        device=ring_info.atom_in_ring.device,
    )