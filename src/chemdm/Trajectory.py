import torch as pt

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