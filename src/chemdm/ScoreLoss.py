import torch as pt
import torch.nn as nn
import chemdm.diffusion as df

class ScoreLoss( nn.Module ):
    def __init__( self, device : pt.device, dtype : pt.dtype ):
        super().__init__()
        self.device = device
        self.dtype = dtype

    def forward(self,
                score_network : nn.Module,
                u0 : pt.Tensor, # (B,2)
                xA : pt.Tensor, # (B, 2)
                xB : pt.Tensor, #(B,2)
                s : pt.Tensor, # (B,)
                ) -> pt.Tensor:
        """
        Evaluate the score-based loss function based on random samples.
        """
        u0 = u0.to(device=self.device)
        xA = xA.to(device=self.device)
        xB = xB.to(device=self.device)
        s = s.to(device=self.device)

        # Sample t quadratically
        B_ = u0.shape[0]
        tmin = 1e-4
        t = tmin + (df.T - tmin) * pt.rand((B_,), device=self.device, dtype=self.dtype)**2

        # Forward diffusion
        mt = df.mean_factor_tensor(t)[:,None]
        vt = df.var_tensor(t)[:,None]
        stds = pt.sqrt(vt)
        noise = pt.randn_like( u0 )
        ut = u0 * mt + noise * stds

        # Propage the noise through the network
        output = score_network(ut, t, xA, xB, s)

        # Compute the loss
        ref_output = -(ut - u0 * mt) / vt
        loss = pt.mean(vt * (output - ref_output)**2)
    
        return loss