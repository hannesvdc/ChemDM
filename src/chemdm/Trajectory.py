import torch as pt

from dataclasses import dataclass
from typing import List, Set

@dataclass
class Trajectory:
    Z : pt.Tensor
    xA : pt.Tensor
    xB : pt.Tensor
    GA : pt.Tensor
    GB : pt.Tensor
    s : pt.Tensor
    x : pt.Tensor