import torch as pt

from dataclasses import dataclass
from typing import List, Set

@dataclass
class Trajectory:
    Z : pt.Tensor
    xA : pt.Tensor
    xB : pt.Tensor
    GA : List[Set[int]]
    GB : List[Set[int]]
    s : pt.Tensor
    x : pt.Tensor