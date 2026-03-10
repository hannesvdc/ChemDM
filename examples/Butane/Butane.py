import math
import torch as pt

class Butane:
    theta0 = 112 * math.pi / 180.0
    k_theta = 62500
    c0 = 1031.36
    c1 = 2037.82
    c2 = 158.52
    c3 = -3227.7

def potential( x : pt.Tensor,
               eps : float = 1e-7 ) -> pt.Tensor:
    """
    Evaluate the potential energy of `x` with shape (B, 4, 3). The output
    is a vector of scalar potentials, combined size (B,).
    """
    x1 = x[:,0,:]
    x2 = x[:,1,:]
    x3 = x[:,2,:]
    x4 = x[:,3,:]

    # Compute the angles \theta_1 and \theta_2
    v12 = x1 - x2
    v32 = x3 - x2
    v43 = x4 - x3
    v23 = x2 - x3
    cos_theta1 = pt.sum(v12 * v32, dim=1) / ( pt.norm(v12, dim=1) * pt.norm(v32, dim=1) )
    cos_theta2 = pt.sum(v43 * v23, dim=1) / ( pt.norm(v43, dim=1) * pt.norm(v23, dim=1) )
    cos_theta1 = pt.clamp(cos_theta1, -1.0 + eps, 1.0 - eps)
    cos_theta2 = pt.clamp(cos_theta2, -1.0 + eps, 1.0 - eps)
    theta_1 = pt.arccos( cos_theta1 )
    theta_2 = pt.arccos( cos_theta2 )

    # Compute the cosine of the dihedral (torsion) angle
    b1 = x2 - x1
    b2 = x3 - x2
    b3 = x4 - x3

    # Normals to the planes (x1,x2,x3) and (x2,x3,x4)
    n1 = pt.cross(b1, b2, dim=1)
    n2 = pt.cross(b2, b3, dim=1)
    n1_norm = pt.norm(n1, dim=1, keepdim=True )
    n2_norm = pt.norm(n2, dim=1, keepdim=True )
    n1_hat = n1 / n1_norm
    n2_hat = n2 / n2_norm
    cos_phi = pt.sum(n1_hat * n2_hat, dim=1 )

    return 0.5*Butane.k_theta * ( (theta_1 - Butane.theta0)**2 + (theta_2 - Butane.theta0)**2 ) \
           + Butane.c0 + Butane.c1 * cos_phi + Butane.c2 * cos_phi**2 + Butane.c3 * cos_phi**3