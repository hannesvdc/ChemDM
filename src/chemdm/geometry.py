import numpy as np
import torch as pt

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
                       ) -> np.ndarray:
    """
    Align P onto Q using Kabsch.
    P, Q: (n_atoms, 3)
    Returns aligned P.
    """
    Pc = P - P.mean(axis=0, keepdims=True)
    Qc = Q - Q.mean(axis=0, keepdims=True)

    C = Pc.T @ Qc
    V, S, Wt = np.linalg.svd(C)

    # Reflection correction
    d = np.sign(np.linalg.det(V @ Wt))
    D = np.diag([1.0, 1.0, d])

    R = V @ D @ Wt
    P_aligned = Pc @ R

    return P_aligned

def center_xyz_torch(xyz: pt.Tensor) -> pt.Tensor:
    """
    xyz: (..., n_atoms, 3)
    Returns coordinates centered by geometric centroid.
    """
    return xyz - xyz.mean(dim=-2, keepdim=True)


def kabsch_align_torch(P: pt.Tensor, 
                       Q: pt.Tensor) -> pt.Tensor:
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