import openmm as mm
import openmm.app as app
import openmm.unit as unit

from openmmtools import testsystems


def build_alanine_dipeptide_simulation(
    temperature=300.0,
    friction=1.0,
    timestep_fs=2.0,
    platform_name=None,
    platform_properties=None,
    phi_target=None,   # radians
    psi_target=None,   # radians
    minimize=True,
    ):
    """
    Build alanine dipeptide in implicit solvent using openmmtools.

    Optional:
        phi_target, psi_target : target backbone torsions in radians.
                                 If both are provided, the initial structure
                                 is rotated in Cartesian coordinates before
                                 creating the simulation context.

    Returns:
        simulation, topology, positions
    """
    # Ready-made alanine dipeptide test system
    ad = testsystems.AlanineDipeptideImplicit()

    system = ad.system
    positions = ad.positions
    topology = ad.topology

    # Standard alanine dipeptide quartets for the openmmtools / MDTraj ordering
    phi_quartet = (4, 6, 8, 14)
    psi_quartet = (6, 8, 14, 16)

    # Convert positions to numpy array in nm
    xyz = positions.value_in_unit(unit.nanometer)

    # Optional torsion seeding
    if phi_target is not None and psi_target is not None:
        xyz = set_phi_psi(
            topology=topology,
            xyz=xyz,
            phi_quartet=phi_quartet,
            psi_quartet=psi_quartet,
            phi_target=phi_target,
            psi_target=psi_target,
        )
        positions = xyz * unit.nanometer

    integrator = mm.LangevinMiddleIntegrator(
        temperature * unit.kelvin,
        friction / unit.picosecond,
        timestep_fs * unit.femtoseconds,
    )

    if platform_name is None:
        simulation = app.Simulation(topology, system, integrator)
    else:
        platform = mm.Platform.getPlatformByName(platform_name)
        if platform_properties is None:
            simulation = app.Simulation(topology, system, integrator, platform)
        else:
            simulation = app.Simulation(
                topology, system, integrator, platform, platform_properties
            )
    simulation.context.setPositions(positions)

    if minimize:
        simulation.minimizeEnergy( maxIterations=5 )

    # Return the minimized positions, not just the initial guess
    state = simulation.context.getState(getPositions=True)
    positions = state.getPositions(asNumpy=True)

    simulation.context.setVelocitiesToTemperature(temperature * unit.kelvin)

    return simulation, topology, positions

import numpy as np
from collections import deque


def wrap_to_pi(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def compute_dihedral(p0, p1, p2, p3):
    """
    Signed dihedral angle in radians.

    Parameters
    ----------
    p0, p1, p2, p3 : np.ndarray
        Arrays of shape (..., 3)

    Returns
    -------
    angle : np.ndarray
        Array of shape (...) with signed dihedral angles in radians.
    """
    p0 = np.asarray(p0)
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    p3 = np.asarray(p3)

    b0 = p1 - p0
    b1 = p2 - p1
    b2 = p3 - p2

    b1_norm = np.linalg.norm(b1, axis=-1, keepdims=True)
    b1_hat = b1 / (b1_norm + 1e-12)

    v = b0 - np.sum(b0 * b1_hat, axis=-1, keepdims=True) * b1_hat
    w = b2 - np.sum(b2 * b1_hat, axis=-1, keepdims=True) * b1_hat

    x = np.sum(v * w, axis=-1)
    y = np.sum(np.cross(b1_hat, v) * w, axis=-1)

    return np.arctan2(y, x)

def compute_torsion_from_xyz(xyz, atoms):
    """
    Compute one torsion angle from coordinates.

    Parameters
    ----------
    xyz : np.ndarray
        Array of shape (..., n_atoms, 3)
    atoms : tuple[int, int, int, int]
        Atom quartet defining the torsion.

    Returns
    -------
    angle : np.ndarray
        Array of shape (...) in radians.
    """
    xyz = np.asarray(xyz)
    if xyz.ndim < 2 or xyz.shape[-1] != 3:
        raise ValueError(f"xyz must have shape (..., n_atoms, 3), got {xyz.shape}")

    i, j, k, l = atoms
    return wrap_to_pi(
        compute_dihedral(
            xyz[..., i, :],
            xyz[..., j, :],
            xyz[..., k, :],
            xyz[..., l, :],
        )
    )

def compute_phi_psi_from_xyz(
    xyz,
    phi_atoms=(4, 6, 8, 14),
    psi_atoms=(6, 8, 14, 16),
):
    """
    Compute alanine-dipeptide phi/psi torsions from Cartesian coordinates.

    Parameters
    ----------
    xyz : np.ndarray
        Array of shape (..., n_atoms, 3)
    phi_atoms : tuple[int, int, int, int]
        Atom quartet defining phi.
    psi_atoms : tuple[int, int, int, int]
        Atom quartet defining psi.

    Returns
    -------
    phi, psi : np.ndarray
        Arrays of shape (...) in radians.
    """
    xyz = np.asarray(xyz)
    if xyz.ndim < 2 or xyz.shape[-1] != 3:
        raise ValueError(f"xyz must have shape (..., n_atoms, 3), got {xyz.shape}")

    phi = compute_torsion_from_xyz(xyz, phi_atoms)
    psi = compute_torsion_from_xyz(xyz, psi_atoms)
    return phi, psi


def rodrigues_rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotation matrix for rotation by `angle` around `axis`.
    """
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)

    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1.0 - c

    return np.array([
        [c + x*x*C,     x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s,   c + y*y*C,   y*z*C - x*s],
        [z*x*C - y*s,   z*y*C + x*s, c + z*z*C  ],
    ])


def build_bond_graph(topology) -> dict[int, set[int]]:
    """
    Build an adjacency graph from an OpenMM topology.
    """
    graph = {atom.index: set() for atom in topology.atoms()}
    for bond in topology.bonds():
        a = bond[0].index
        b = bond[1].index
        graph[a].add(b)
        graph[b].add(a)
    return graph


def atoms_on_side_of_bond(graph: dict[int, set[int]], fixed_atom: int, rotating_atom: int) -> list[int]:
    """
    If we cut the bond fixed_atom -- rotating_atom,
    return all atoms connected to rotating_atom.
    """
    visited = {fixed_atom}
    q = deque([rotating_atom])
    side = []

    while q:
        u = q.popleft()
        if u in visited:
            continue
        visited.add(u)
        side.append(u)
        for v in graph[u]:
            if v not in visited:
                q.append(v)

    return side


def rotate_group_about_axis(
    xyz: np.ndarray,
    atom_indices: list[int],
    axis_point1: np.ndarray,
    axis_point2: np.ndarray,
    angle: float,
) -> np.ndarray:
    """
    Rotate selected atoms about the axis through axis_point1 -> axis_point2.
    """
    xyz_new = xyz.copy()
    R = rodrigues_rotation_matrix(axis_point2 - axis_point1, angle)

    for idx in atom_indices:
        v = xyz[idx] - axis_point1
        xyz_new[idx] = axis_point1 + R @ v

    return xyz_new


def set_dihedral(
    topology,
    xyz: np.ndarray,
    quartet: tuple[int, int, int, int],
    target_angle: float,
) -> tuple[np.ndarray, float, float]:
    """
    Set the dihedral defined by quartet = (i, j, k, l) to target_angle.
    Rotates the side connected to atom k around bond j-k.

    Returns:
        xyz_new, old_angle, new_angle
    """
    i, j, k, l = quartet
    xyz = np.asarray(xyz, dtype=float)

    old_angle = compute_dihedral(xyz[i], xyz[j], xyz[k], xyz[l])
    delta = wrap_to_pi(target_angle - old_angle)

    graph = build_bond_graph(topology)
    rotating_atoms = atoms_on_side_of_bond(graph, fixed_atom=j, rotating_atom=k)

    xyz_new = rotate_group_about_axis(
        xyz,
        atom_indices=rotating_atoms,
        axis_point1=xyz[j],
        axis_point2=xyz[k],
        angle=delta,
    )

    new_angle = compute_dihedral(xyz_new[i], xyz_new[j], xyz_new[k], xyz_new[l])
    new_angle = wrap_to_pi(new_angle)

    return xyz_new, old_angle, new_angle

def set_phi_psi(
    topology,
    xyz: np.ndarray,
    phi_quartet: tuple[int, int, int, int],
    psi_quartet: tuple[int, int, int, int],
    phi_target: float,
    psi_target: float,
) -> np.ndarray:
    """
    Sequentially set phi then psi.
    """
    xyz1, _, _ = set_dihedral(topology, xyz, phi_quartet, phi_target)
    xyz2, _, _ = set_dihedral(topology, xyz1, psi_quartet, psi_target)
    return xyz2