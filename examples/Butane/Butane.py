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
    The potential should be maximal whenever phi = +- pi, or cos(phi) = -1.
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
           + Butane.c0 - Butane.c1 * cos_phi + Butane.c2 * cos_phi**2 - Butane.c3 * cos_phi**3

def potential_internal( q : pt.Tensor,
                        eps : float = 1e-7 ) -> pt.Tensor:
    """
    Evaluate the potential energy in internal coordintes. 
    The internal coordinates are (cos(theta1), cos(theta2), cos(phi), sin(phi)) - 
    `q` has shape (B, 4). The output is a vector of scalar potentials, combined size (B,).
    """
    theta_1 = pt.arccos( q[:,0] )
    theta_2 = pt.arccos( q[:,1] )
    cos_phi = q[:,2]
    sin_phi = q[:,3]
    
    scale = cos_phi**2 + sin_phi**2
    cos_phi = cos_phi / pt.sqrt( scale )

    return 0.5*Butane.k_theta * ( (theta_1 - Butane.theta0)**2 + (theta_2 - Butane.theta0)**2 ) \
           + Butane.c0 - Butane.c1 * cos_phi + Butane.c2 * cos_phi**2 - Butane.c3 * cos_phi**3

def internalToCartesian( q: pt.Tensor,
                         bond_length : float = 1.0,
                        ) -> pt.Tensor:
    """
    Convert the (flattened) internal representation of Butane to a canonical Cartesian
    representation.
    """
    theta1 = pt.arccos( q[:,0] )
    theta2 = pt.arccos( q[:,1] )
    cos_phi = q[:,2]
    sin_phi = q[:,3]
    
    x = pt.zeros( (len(cos_phi), 4, 3) )
    x[:, 1, :] = 0.0 # x2 = origin
    x[:, 2, 0] = bond_length # x3 = on the x-axis
    x[:, 0, 0] = -bond_length * pt.cos( theta1 ) # x1 in xy-plane
    x[:, 0, 1] =  bond_length * pt.sin( theta1 )
    x[:, 3, 0] =  bond_length - bond_length * pt.cos( theta2 ) # x4 from theta2 and dihedral phi
    x[:, 3, 1] =  bond_length * pt.sin( theta2 ) * cos_phi
    x[:, 3, 2] =  bond_length * pt.sin( theta2 ) * sin_phi

    # Subtract the center of mass to obtain translation invariance.
    x = x - pt.mean( x, dim=1, keepdim=True )
    return x

def cartesianToInternal( x : pt.Tensor ) -> pt.Tensor:
    """ 
    Computes the cosine of the two angles $\\theta_1$ and $\\theta_2$, and
    the sin and cos of the torsion angle $\\phi$ of butane.
    Input `x` must have shape (...,4,3), output has shape (...,4).
    """
    x1 = x[...,0,:]
    x2 = x[...,1,:]
    x3 = x[...,2,:]
    x4 = x[...,3,:]

    # Bond-angle cosines
    v12 = x1 - x2 # (...,3)
    v32 = x3 - x2
    v43 = x4 - x3
    v23 = x2 - x3
    cos_theta1 = pt.sum(v12 * v32, dim=-1, keepdim=True) / (
        pt.norm(v12, dim=-1, keepdim=True) * pt.norm(v32, dim=-1, keepdim=True) )
    cos_theta2 = pt.sum(v43 * v23, dim=-1, keepdim=True) / (
        pt.norm(v43, dim=-1, keepdim=True) * pt.norm(v23, dim=-1, keepdim=True) )
    
    # Torsion angle around bond x2-x3
    b1 = x2 - x1 # (...,3)
    b2 = x3 - x2
    b3 = x4 - x3

    # Normals to the planes (x1,x2,x3) and (x2,x3,x4)
    n1 = pt.cross(b1, b2, dim=-1)
    n2 = pt.cross(b2, b3, dim=-1)

    n1_norm = pt.norm(n1, dim=-1, keepdim=True)
    n2_norm = pt.norm(n2, dim=-1, keepdim=True)
    b2_norm = pt.norm(b2, dim=-1, keepdim=True)

    n1_hat = n1 / n1_norm # (...,3)
    n2_hat = n2 / n2_norm
    b2_hat = b2 / b2_norm

    # cos(phi)
    cos_phi = pt.sum(n1_hat * n2_hat, dim=-1, keepdim=True)
    sin_phi = pt.sum(pt.cross(n1_hat, n2_hat, dim=-1) * b2_hat, dim=-1, keepdim=True)

    return pt.cat( (cos_theta1, cos_theta2, cos_phi, sin_phi), dim=-1 )