import torch as pt

def potential( xy : pt.Tensor ) -> pt.Tensor:
    if xy.ndim == 1:
        xy = xy[None,:]
    assert xy.ndim == 2 and xy.shape[1]==2, f"`xy` must have shape (B, 2) with B the batch size, but got {xy.shape}."
    x = xy[:,0]
    y = xy[:,1]

    return -200.0 * pt.exp( -(x-1.0)**2 - 10.0*y**2 )\
           -100.0 * pt.exp( -x**2 - 10.0*(y-1/2)**2 ) \
           -170.0 * pt.exp( -(13/2)*(x+1/2)**2 + 11.0*(x+1/2)*(y-3/2) - (13/2)*(y-3/2)**2 )\
            +15.0 * pt.exp(  (7/10)*(x+1)**2 + (3/5)*(x+1.0)*(y-1.0) + (7/10)*(y-1.0)**2 )

def get_fixed_points() -> pt.Tensor:
    return pt.tensor([[0.623499404930877, 0.0280377585286857], # min
                      [0.212486582000662, 0.292988325107368], # saddle
                      [-0.0500108229982061, 0.466694104871972], # min
                      [-0.822001558732732, 0.624312802814871], # saddle
                      [-0.558223634633024, 1.44172584180467]]) # min