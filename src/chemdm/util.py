import torch as pt

def getGradientNorm( model : pt.nn.Module ):
    grads = []
    for param in model.parameters():
        if param.grad is not None:
            grads.append(param.grad.view(-1))
    grads = pt.cat(grads)
    return pt.norm(grads).item()