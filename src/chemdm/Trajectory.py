import torch as pt

from chemdm.geometry import kabsch_rotation_torch

from dataclasses import dataclass

@dataclass
class Trajectory:
    Z : pt.Tensor
    xA : pt.Tensor # (n_atoms, 3)
    xB : pt.Tensor # (n_atoms, 3)
    GA : pt.Tensor
    GB : pt.Tensor
    s : pt.Tensor
    x : pt.Tensor # (n_images, n_atoms, 3)

@pt.no_grad()
def enforceCOM( trajectory : Trajectory ) -> Trajectory:
    """
    Center each molecule on the trajectory to have zero center of mass. Returns
    a copy of the trajectory.
    """
    x_centered = trajectory.x - pt.mean(trajectory.x, dim=1, keepdim=True)
    xA_centered = trajectory.xA - pt.mean(trajectory.xA, dim=0, keepdim=True)
    xB_centered = trajectory.xB - pt.mean(trajectory.xB, dim=0, keepdim=True)

    return Trajectory(
        Z=trajectory.Z,
        xA=xA_centered,
        xB=xB_centered,
        GA=trajectory.GA,
        GB=trajectory.GB,
        s=trajectory.s,
        x=x_centered,
    )

@pt.no_grad()
def alignToReactant( trajectory : Trajectory ) -> Trajectory:
    """
    Kabsch-align xB and each path image individually onto xA (rotation only).
    Each image gets its own optimal rotation since intermediate geometries
    differ from both xA and xB.
    Assumes zero center of mass (call enforceCOM first).
    Returns a copy of the trajectory.
    """
    R_B = kabsch_rotation_torch( trajectory.xB, trajectory.xA )
    xB_aligned = trajectory.xB @ R_B

    n_images = trajectory.x.shape[0]
    x_aligned = pt.empty_like( trajectory.x )
    for k in range( n_images ):
        R_k = kabsch_rotation_torch( trajectory.x[k], trajectory.xA )
        x_aligned[k] = trajectory.x[k] @ R_k

    return Trajectory(
        Z=trajectory.Z,
        xA=trajectory.xA,
        xB=xB_aligned,
        GA=trajectory.GA,
        GB=trajectory.GB,
        s=trajectory.s,
        x=x_aligned,
    )