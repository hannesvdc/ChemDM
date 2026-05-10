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