import numpy as np
import torch as pt

from typing import Optional

# This file includes some very general geometry functions that are useful
# across the ChemDM library.

def center_xyz_numpy(xyz: np.ndarray) -> np.ndarray:
    """
    xyz: (..., n_atoms, 3)
    Returns coordinates centered by geometric centroid.
    """
    return xyz - xyz.mean(axis=-2, keepdims=True)

def kabsch_align_numpy(P: np.ndarray, 
                       Q: np.ndarray,
                       Z : Optional[np.ndarray] = None) -> np.ndarray:
    """
    Align P onto Q using the Kabsch algorithm.

    Parameters
    ----------
    P, Q: Arrays of shape (n_atoms, 3). P is aligned onto Q.
    Z: Optional atomic numbers of shape (n_atoms,). If provided,
            hydrogen atoms (Z == 1) are excluded from the alignment fit.
            The resulting rigid transform is still applied to all atoms.

    Returns
    -------
    P_aligned: Array of shape (n_atoms, 3), all atoms of P aligned into Q's frame.
    """
    n_atoms = P.shape[0]
    if Z is None:
        align_idx = np.arange(n_atoms)
    else:
        Z = np.asarray(Z, dtype=np.long)
        if Z.shape != (n_atoms,):
            raise ValueError(f"Z must have shape ({n_atoms},), got {Z.shape}.")
        align_idx = np.where(Z != 1)[0]  # non-hydrogen atoms
        if len(align_idx) < 3:
            raise ValueError(
                f"Need at least 3 non-hydrogen atoms for a robust Kabsch alignment; "
                f"got {len(align_idx)}." )

    # Centroids computed only over alignment atoms.
    P_centroid = P[align_idx].mean(axis=0, keepdims=True)
    Q_centroid = Q[align_idx].mean(axis=0, keepdims=True)
    P_centered_all = P - P_centroid # Center all atoms using the alignment centroids.

    # Center only alignment atoms for fitting rotation.
    P_fit = P[align_idx] - P_centroid
    Q_fit = Q[align_idx] - Q_centroid
    C = P_fit.T @ Q_fit
    V, S, Wt = np.linalg.svd(C)

    # Reflection correction.
    d = np.sign(np.linalg.det(V @ Wt))
    D = np.diag([1.0, 1.0, d])
    R = V @ D @ Wt

    # Apply transform to all atoms, not just heavy atoms.
    P_aligned = P_centered_all @ R + Q_centroid

    return P_aligned

def center_xyz_torch(xyz: pt.Tensor) -> pt.Tensor:
    """
    xyz: (..., n_atoms, 3)
    Returns coordinates centered by geometric centroid.
    """
    return xyz - xyz.mean(dim=-2, keepdim=True)


def kabsch_align_torch(P: pt.Tensor, 
                       Q: pt.Tensor,) -> pt.Tensor:
    """
    Align P onto Q with Kabsch.
    P, Q: (n_atoms, 3)
    Returns aligned P, centered around the origin.
    """
    Pc = P - P.mean(dim=0, keepdim=True)
    Qc = Q - Q.mean(dim=0, keepdim=True)

    C = Pc.T @ Qc
    U, S, Vh = pt.linalg.svd(C)

    det = pt.linalg.det(U @ Vh)
    D = pt.diag(
        pt.tensor(
            [1.0, 1.0, 1.0 if det >= 0 else -1.0],
            dtype=P.dtype,
            device=P.device,
        )
    )

    R = U @ D @ Vh
    return Pc @ R


def kabsch_rotation_torch(P: pt.Tensor,
                          Q: pt.Tensor) -> pt.Tensor:
    """
    Compute the Kabsch rotation matrix that best aligns P onto Q.
    Both are centered before computing R.

    P, Q: (n_atoms, 3)

    Returns
    -------
    R : (3, 3) rotation matrix. Apply as X_aligned = X_centered @ R.
    """
    Pc = P - P.mean(dim=0, keepdim=True)
    Qc = Q - Q.mean(dim=0, keepdim=True)

    C = Pc.T @ Qc
    U, S, Vh = pt.linalg.svd(C)

    det = pt.linalg.det(U @ Vh)
    D = pt.diag(
        pt.tensor(
            [1.0, 1.0, 1.0 if det >= 0 else -1.0],
            dtype=P.dtype,
            device=P.device,
        )
    )

    return U @ D @ Vh

def normalize(v: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, eps)

def perpendicular_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    axis: (..., 3)
    Returns u, v with shape (..., 3), perpendicular to axis.
    """
    axis = normalize(axis, axis=-1)

    tmp = np.zeros_like(axis)
    use_x = np.abs(axis[..., 0]) < 0.9
    tmp[use_x] = np.array([1.0, 0.0, 0.0])
    tmp[~use_x] = np.array([0.0, 1.0, 0.0])

    u = tmp - np.sum(tmp * axis, axis=-1, keepdims=True) * axis
    u = normalize(u, axis=-1)
    v = np.cross(axis, u)
    v = normalize(v, axis=-1)

    return u, v

def perpendicular_basis_continuous( axis: np.ndarray ) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a continuous perpendicular basis along a path.

    axis: (M, 3)

    Returns
    -------
    u, v: both (M, 3)

    This avoids abrupt frame flips from choosing a global x/y reference
    independently at each image.
    """
    axis = normalize(axis, axis=-1)
    M = axis.shape[0]

    u = np.zeros_like(axis)
    v = np.zeros_like(axis)

    # Initialize u[0] using a safe global reference.
    if abs(axis[0, 0]) < 0.9:
        tmp = np.array([1.0, 0.0, 0.0])
    else:
        tmp = np.array([0.0, 1.0, 0.0])

    u0 = tmp - np.dot(tmp, axis[0]) * axis[0]
    u0 = u0 / np.linalg.norm(u0)

    u[0] = u0
    v[0] = np.cross(axis[0], u[0])
    v[0] = v[0] / np.linalg.norm(v[0])

    for m in range(1, M):
        # Project previous u onto the plane perpendicular to current axis.
        um = u[m - 1] - np.dot(u[m - 1], axis[m]) * axis[m]

        # Rare fallback if previous u becomes nearly parallel to current axis.
        if np.linalg.norm(um) < 1e-8:
            if abs(axis[m, 0]) < 0.9:
                tmp = np.array([1.0, 0.0, 0.0])
            else:
                tmp = np.array([0.0, 1.0, 0.0])
            um = tmp - np.dot(tmp, axis[m]) * axis[m]

        um = um / np.linalg.norm(um)

        # Optional sign correction: prevent u from flipping by 180 degrees.
        if np.dot(um, u[m - 1]) < 0.0:
            um = -um

        u[m] = um
        v[m] = np.cross(axis[m], u[m])
        v[m] = v[m] / np.linalg.norm(v[m])
    return u, v

def apply_torsion_update( x:  pt.Tensor,
                          bonds: pt.Tensor,
                          side_atom_idx: pt.Tensor,
                          side_bond_idx: pt.Tensor,
                          delta_tau: pt.Tensor,
    ) -> pt.Tensor:
    """
    Apply per-bond torsion increments Δτ to a (possibly batched) atom point
    cloud, leaving bond lengths and bond angles invariant.

    For each rotatable bond i with endpoints (b_i, c_i) we rotate the atoms
    on its c-side by Δτ_i around the bond axis (x_{c_i} - x_{b_i}) passing
    through x_{b_i}. Proposition 1 of Jing et al. 2022 ("Torsional Diffusion
    for Molecular Conformer Generation") shows that this operation is
    well-defined regardless of any choice of reference neighbors, and that
    the final geometry is independent of the order in which bonds are
    processed when multiple Δτ_i are applied. We process bonds in their
    natural order.

    Important: this function assumes bonds are undirected, and b is always
    the left (fixed) end point, and c will be rotated. If a bond appears twice
    (b -> c) and (c -> b), two torsion updates will be applied (one in each direction).

    Parameters
    ----------
    x : (N, 3) float
        Atom positions; may be a flat batched cloud (multiple molecules
        concatenated along dim 0).
    bonds : (m, 2) long
        Rotatable-bond endpoints (b_i, c_i) as global atom indices into `x`.
    side_atom_idx : (P,) long
        COO-style: atom (global) index on the c-side of some bond.
    side_bond_idx : (P,) long
        COO-style: which row of `bonds` each side-atom belongs to.
    delta_tau : (m,) float
        Per-bond rotation angle in radians.

    Returns
    -------
    x_new : (N, 3) float
        New positions with the torsion updates applied. Input is not mutated.

    Vectorization
    -------------
    Sequential over `m` bonds (Python loop), vectorized over the c-side
    atoms within each bond using Rodrigues' rotation formula:

        p' = p cosθ + (k x p) sinθ + k (k · p) (1 - cosθ)

    where k is the unit bond axis and p = x_atom - x_{b_i}.
    """
    assert bonds.ndim == 2 and bonds.shape[1] == 2
    assert side_atom_idx.shape == side_bond_idx.shape
    assert delta_tau.shape == (bonds.shape[0],)

    x_new = x.clone()
    m = bonds.shape[0]
    for i in range(m):
        b = bonds[i, 0]
        c = bonds[i, 1]

        bond_vec = x_new[c] - x_new[b]  # (3,)
        bond_axis = bond_vec / pt.linalg.norm(bond_vec).clamp_min(1.0e-8)   # (3,)

        atoms = side_atom_idx[side_bond_idx == i]  # (P_i,)
        if atoms.numel() == 0:
            continue

        center = x_new[b]  # (3,)
        p = x_new[atoms] - center   # (P_i, 3)

        angle = delta_tau[i]
        cos_a = pt.cos( angle )
        sin_a = pt.sin( angle )

        # Broadcasting bond_axis (3,) over (P_i, 3):
        cross = pt.linalg.cross( bond_axis.expand_as(p), p, dim=-1 )            # (P_i, 3)
        dot = (bond_axis * p).sum(dim=-1, keepdim=True)                       # (P_i, 1)
        p_rot = p * cos_a + cross * sin_a + bond_axis * dot * (1.0 - cos_a)     # (P_i, 3)

        x_new[atoms] = p_rot + center

    return x_new