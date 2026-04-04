import torch as pt

def getGradientNorm( model : pt.nn.Module ):
    grads = []
    for param in model.parameters():
        if param.grad is not None:
            grads.append(param.grad.view(-1))
    grads = pt.cat(grads)
    return pt.norm(grads).item()

@pt.no_grad()
def perCoordinateRMSE( x : pt.Tensor,
                       x_pred : pt.Tensor ) -> float:
    """
    Root-mean-square error per Cartesian coordinate between two position tensors.

    Arguments
    ---------
    x : (N, 3) reference positions.
    x_pred : (N, 3) predicted positions.

    Returns
    -------
    rmse : float
        sqrt( mean( (x - x_pred)^2 ) ), averaged over all N*3 entries.
    """
    assert x.shape == x_pred.shape, \
        f"`x` and `x_pred` must have the same shape, got {x.shape} and {x_pred.shape}."
    return pt.sqrt( pt.mean( (x - x_pred) ** 2 ) ).item()

@pt.no_grad()
def isInteger( x : pt.Tensor,
              float_tol : float = 1e-7 ) -> pt.Tensor:
    """
    Check if the entries of the input tensor are integers. Returns a bool tensor
    of the same size indicating whether the corresponding entry in x is an integer.

    Arguments
    ---------
    x: pt.Tensor (any size)
        The array to check.
    float_tol : float
        Tolerance used to check floating point numbers. Default 1e-7 for single precision.

    Returns
    -------
    is_integer : pt.Tensor of type bool
        The boolean output tensor.
    """
    if x.dtype in [pt.uint8, pt.int8, pt.int16, pt.int32, pt.int64]:
        return pt.ones_like( x, dtype=pt.bool )
    return pt.abs( x - x.int() ) <= float_tol