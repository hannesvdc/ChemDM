import torch as pt
import torch.nn as nn

from chemdm.NewtonE3NNLayer import E3State
from chemdm.MoleculeGraph import BatchedMoleculeGraph


class NewtonLoss(nn.Module):
    """
    Loss over all refinement states with exponentially larger weight on later
    refinement steps.

    Assumes:
        states[0] = initial state, usually linear interpolation
        states[1] = after first refinement step
        ...
        states[K] = after K refinement steps

    The loss ignores states[0] by default.
    """

    def __init__( self, gamma: float = 0.7 ) -> None:
        super().__init__()

        assert 0.0 < gamma <= 1.0
        self.gamma = gamma

    def _single_state_loss( self, x_pred: pt.Tensor,  
                                  x_target: pt.Tensor,
                                  molecule_id : pt.Tensor ) -> pt.Tensor:
        """
        RMSD-like coordinate loss.

        x_pred, x_target: shape (N, 3)
        """
        assert x_pred.shape == x_target.shape, f"`x_pred` and `x_target` must have the same shape,  but got {x_pred.shape} and {x_target.shape}."
        assert molecule_id.shape[0] == x_pred.shape[0],  "`molecule_id` must have one entry per atom."

        squared_dist = ((x_pred - x_target) ** 2).sum(dim=-1)  # (N,)

        molecule_losses = []
        for mol_id in pt.unique(molecule_id):
            mask = (molecule_id == mol_id)
            molecule_losses.append( squared_dist[mask].mean() )

        return pt.stack( molecule_losses ).mean()

    def forward( self, states: list[E3State], 
                       x_final : BatchedMoleculeGraph,
                       x_target: pt.Tensor, ) -> pt.Tensor:
        loss_states = states[1:]
        assert len(loss_states) > 0, "No refinement states available for loss."

        K = len(loss_states)
        device = x_target.device
        dtype = x_target.dtype

        # Later states receive larger weight.
        # For K states:
        #   [gamma^(K-1), gamma^(K-2), ..., gamma^0]
        weights = pt.tensor( [self.gamma ** (K - 1 - k) for k in range(K)], device=device, dtype=dtype )
        weights = weights / weights.sum()

        loss = pt.zeros((), device=device, dtype=dtype)
        for weight, state in zip(weights, loss_states):
            loss = loss + weight * self._single_state_loss(state.x, x_target, x_final.molecule_id)

        return loss