import math
import torch as pt
import torch.nn as nn

from typing import Tuple

class DDPMSchedule( nn.Module ):
    """
    Denoising Diffusion Probabilistic Model noise schedule.

    Timestep convention: t in {0, 1, ..., T-1}.
      t = 0     nearly clean   (alpha_bar close to 1)
      t = T-1   nearly pure noise (alpha_bar close to 0)
    """

    def __init__( self,
                  T : int = 1000,
                  schedule : str = "linear"
                ) -> None:
        super().__init__()
        self.T = T

        if schedule == "linear":
            beta = pt.linspace( 1e-4, 0.02, T )
            alpha = 1.0 - beta
            alpha_bar = pt.cumprod( alpha, dim=0 )
            alpha_bar_prev = pt.cat([ pt.tensor([1.0]), alpha_bar[:-1] ])
        elif schedule == "cosine":
            # Cosine schedule (Nichol & Dhariwal, 2021)
            s = 0.008
            steps = pt.linspace( 0, T, T + 1, dtype=pt.float64 )
            f = pt.cos( (steps / T + s) / (1.0 + s) * (math.pi / 2.0) ) ** 2
            alpha_bar_full = (f / f[0]).float()
            # alpha_bar_full[0] = 1 (no noise), alpha_bar_full[T] ~ 0 (full noise)
            alpha_bar = alpha_bar_full[1:]        # T entries for t = 0 .. T-1
            alpha_bar_prev = alpha_bar_full[:-1]  # alpha_bar_prev[0] = 1.0
            beta = pt.clamp( 1.0 - alpha_bar / alpha_bar_prev, min=1e-5, max=0.999 )
            alpha = 1.0 - beta
        else:
            raise ValueError( f"Unknown schedule '{schedule}'. Use 'linear' or 'cosine'." )

        self.register_buffer( "beta", beta )
        self.register_buffer( "alpha", alpha )
        self.register_buffer( "alpha_bar", alpha_bar )
        self.register_buffer( "alpha_bar_prev", alpha_bar_prev )
        self.register_buffer( "sqrt_alpha_bar", pt.sqrt(alpha_bar) )
        self.register_buffer( "sqrt_one_minus_alpha_bar", pt.sqrt(1.0 - alpha_bar) )

    def q_sample( self,
                  x_0 : pt.Tensor,
                  t : pt.Tensor,
                  noise : pt.Tensor | None = None,
                ) -> Tuple[pt.Tensor, pt.Tensor]:
        """
        Forward process: sample x_t given x_0.

        q(x_t | x_0) = N( sqrt(alpha_bar_t) x_0,  (1 - alpha_bar_t) I )

        Arguments
        ---------
        x_0 : (N, 3) clean positions.
        t   : (N,)   integer timesteps, one per atom (atoms in the same
              molecule share the same value).
        noise : optional (N, 3) pre-sampled noise.

        Returns
        -------
        x_t   : (N, 3) noisy positions.
        noise  : (N, 3) the noise that was used.
        """
        if noise is None:
            noise = pt.randn_like( x_0 )

        sqrt_ab  = self.sqrt_alpha_bar[t][:, None]            # (N, 1)
        sqrt_1mab = self.sqrt_one_minus_alpha_bar[t][:, None]  # (N, 1)

        x_t = sqrt_ab * x_0 + sqrt_1mab * noise
        return x_t, noise

    @pt.no_grad()
    def p_sample_step( self,
                       x_0_pred : pt.Tensor,
                       x_t : pt.Tensor,
                       t : pt.Tensor
                     ) -> pt.Tensor:
        """
        Single reverse DDPM step: sample x_{t-1} given x_t and predicted x_0.

        Uses the closed-form posterior q(x_{t-1} | x_t, x_0) with x_0
        replaced by the network prediction x_0_pred.

        Arguments
        ---------
        x_0_pred : (N, 3) predicted clean positions.
        x_t      : (N, 3) current noisy positions.
        t        : (N,)   integer timesteps.

        Returns
        -------
        x_prev : (N, 3) sample at timestep t-1.
        """
        beta_t          = self.beta[t][:, None]
        alpha_bar_t     = self.alpha_bar[t][:, None]
        alpha_bar_prev_t = self.alpha_bar_prev[t][:, None]

        # Posterior mean
        coeff_x0 = pt.sqrt( alpha_bar_prev_t ) * beta_t / (1.0 - alpha_bar_t)
        coeff_xt = pt.sqrt( self.alpha[t][:, None] ) * (1.0 - alpha_bar_prev_t) / (1.0 - alpha_bar_t)
        mu = coeff_x0 * x_0_pred + coeff_xt * x_t

        # Posterior variance
        var = beta_t * (1.0 - alpha_bar_prev_t) / (1.0 - alpha_bar_t)

        # At t = 0 the posterior collapses to a point (var = 0); the mask
        # ensures we do not add noise on the last step.
        noise = pt.randn_like( x_t )
        mask  = (t > 0).float()[:, None]
        x_prev = mu + mask * pt.sqrt( var ) * noise

        return x_prev

    @pt.no_grad()
    def ddim_sample_step( self,
                          x_0_pred : pt.Tensor,
                          x_t : pt.Tensor,
                          t : pt.Tensor,
                          t_prev : pt.Tensor,
                          eta : float = 0.0
                        ) -> pt.Tensor:
        """
        Single DDIM reverse step: x_{t_prev} given x_t and predicted x_0.

        Supports arbitrary step sizes (t and t_prev need not be consecutive)
        and an interpolation parameter eta between deterministic (eta=0) and
        stochastic DDPM (eta=1).

        Arguments
        ---------
        x_0_pred : (N, 3) predicted clean positions.
        x_t      : (N, 3) current noisy positions.
        t        : (N,)   current integer timesteps.
        t_prev   : (N,)   target integer timesteps (t_prev < t).
                   Use -1 to indicate the final step (alpha_bar_prev = 1).
        eta      : float  interpolation between deterministic (0) and DDPM (1).

        Returns
        -------
        x_prev : (N, 3) sample at timestep t_prev.
        """
        alpha_bar_t = self.alpha_bar[t][:, None]

        # For t_prev = -1, use alpha_bar_prev = 1.0 (clean)
        is_final = (t_prev < 0)
        t_prev_clamped = pt.clamp( t_prev, min=0 )
        alpha_bar_prev_t = pt.where(
            is_final[:, None],
            pt.ones( 1, device=x_t.device, dtype=x_t.dtype ),
            self.alpha_bar[t_prev_clamped][:, None]
        )

        # Predicted noise from x_0_pred
        eps_pred = (x_t - pt.sqrt(alpha_bar_t) * x_0_pred) / pt.sqrt(1.0 - alpha_bar_t)

        # DDIM variance
        sigma = eta * pt.sqrt(
            (1.0 - alpha_bar_prev_t) / (1.0 - alpha_bar_t) *
            (1.0 - alpha_bar_t / alpha_bar_prev_t)
        )

        # Direction pointing to x_t
        dir_xt = pt.sqrt( 1.0 - alpha_bar_prev_t - sigma**2 ) * eps_pred

        # DDIM update
        x_prev = pt.sqrt( alpha_bar_prev_t ) * x_0_pred + dir_xt

        # Add noise (only when eta > 0 and not the final step)
        if eta > 0:
            noise = pt.randn_like( x_t )
            mask = (~is_final).float()[:, None]
            x_prev = x_prev + mask * sigma * noise

        return x_prev
