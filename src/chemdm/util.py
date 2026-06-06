import random
import torch as pt

import itertools
from collections import Counter
from typing import List, Tuple

from chemdm.MoleculeGraph import MoleculeGraph, BatchedMoleculeGraph, batchMolecules
from chemdm.Trajectory import Trajectory, enforceCOM, alignToReactant


# Atomic-number → element symbol, covering everything the project's datasets
# (Transition1x, GEOM-QM9, GEOM-DRUGS) actually use. Extend as needed.
Z_TO_SYMBOL: dict[int, str] = {
    1: "H",   6: "C",   7: "N",   8: "O",   9: "F",
    15: "P", 16: "S",  17: "Cl", 35: "Br", 53: "I",
}


def formula_from_Z( Z : pt.Tensor ) -> str:
    """
    Empirical formula string from a tensor of atomic numbers, in Hill order
    (C first, H next, then everything else alphabetical). Single-atom counts
    are written without a trailing "1", e.g.

        formula_from_Z(tensor([6, 6, 1, 1, 1, 1, 1, 1]))  ->  "C2H6"
        formula_from_Z(tensor([8, 1, 1]))                 ->  "H2O"

    Unknown elements are spelled "Z<atomic_number>" (e.g. "Z76").
    """
    counts = Counter( Z_TO_SYMBOL.get(int(z), f"Z{int(z)}") for z in Z.tolist() )
    parts: list[str] = []
    for s in ("C", "H"):
        if s in counts:
            n = counts.pop(s)
            parts.append( s if n == 1 else f"{s}{n}" )
    for s in sorted(counts):
        n = counts[s]
        parts.append( s if n == 1 else f"{s}{n}" )
    return "".join(parts)

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

def collate_molecules(batch : List[List[Trajectory]]
                     ) -> Tuple[BatchedMoleculeGraph, BatchedMoleculeGraph, pt.Tensor, pt.Tensor]:
    """
    Batching molecules and random points on the path.
    """
    trajectories = list(itertools.chain.from_iterable( batch )) # squash the nested list of trajectories

    # Sample random points on the trajectory for each molecule
    xA_molecules = []
    xB_molecules = []
    s_list = []
    x_list = []
    for trajectory in trajectories:
        trajectory = enforceCOM( trajectory )
        trajectory = alignToReactant( trajectory )

        xA = MoleculeGraph( trajectory.Z, trajectory.xA, trajectory.GA )
        xA_molecules.append( xA )
        xB = MoleculeGraph( trajectory.Z, trajectory.xB, trajectory.GB )
        xB_molecules.append( xB )

        s_idx = random.randint( 0, len(trajectory.s)-1 )
        s_list.append( trajectory.s[s_idx] * pt.ones_like(trajectory.Z) )
        x_list.append( trajectory.x[s_idx,:,:] )
    s = pt.cat( s_list ) # (N_atoms,)
    x_ref = pt.cat( x_list, dim=0 ) # (N_atoms,3)

    xA = batchMolecules( xA_molecules )
    xB = batchMolecules( xB_molecules )
    return xA, xB, s, x_ref