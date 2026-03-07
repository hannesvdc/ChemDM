import torch as pt

from chemdm.ScoreNetwork import ScoreNetwork
import chemdm.diffusion as df

# Backward simulation of the ODE using Euler in normalized space.
def sample_sgm_euler(score_model : ScoreNetwork,
                  xA: pt.Tensor,
                  xB: pt.Tensor,
                  s_grid : pt.Tensor,
                  dt: float = 5e-3 
                ) -> pt.Tensor:
    assert xA.ndim == 2 and xA.shape[1] == 2, f"xA must be (B,2), got {xA.shape}"
    assert xB.ndim == 2 and xB.shape[1] == 2, f"xB must be (B,2), got {xB.shape}"
    assert xA.shape == xB.shape, f"xA and xB must have same shape, got {xA.shape} and {xB.shape}"
    if s_grid.ndim != 1:
        s_grid = s_grid.flatten()
    B = xA.shape[0]
    S = s_grid.shape[0]
    device = xA.device
    dtype = xA.dtype

    # Expand on a Cartesian 'grid' but store in 3d 
    # xA_exp, xB_exp: (B, S, 2)
    # s_exp: (B, S)
    xA_exp = xA[:, None, :].expand(B, S, 2)
    xB_exp = xB[:, None, :].expand(B, S, 2)
    s_exp = s_grid[None, :].expand(B, S)

    # Flatten for score-model input
    xA_flat = xA_exp.reshape(B * S, 2)   # (B*S, 2)
    xB_flat = xB_exp.reshape(B * S, 2)   # (B*S, 2)
    s_flat  = s_exp.reshape(B * S)       # (B*S,)

    # Random noise
    y = pt.randn( (B, S, 2), device=device, dtype=dtype)

    n_steps = int(df.T / dt)
    dt = df.T / n_steps
    for n in range(n_steps):
        print('t =', n*dt)
        t_diff = n * dt
        t = (df.T - t_diff) * pt.ones( (B*S,), device=device, dtype=dtype )  # network expects "forward time" t
        t = pt.clamp(t, min=1e-3)

        y_flat = y.reshape(B * S, 2)

        # Compute the score
        score = score_model( y_flat, t, xA_flat, xB_flat, s_flat )

        # Do one backward Euler step
        # NOTE:
        # The exact probability-flow ODE would put a factor 0.5 in front of
        # beta_t * score. In practice, with this approximate learned score and
        # this nearly deterministic dataset, that update was too weak to
        # recover meaningful paths. We therefore use the reverse-SDE drift
        # coefficient here as a heuristic deterministic sampler.
        beta_t = df.beta(t)[:, None]
        y_flat = y_flat + dt*(0.5*beta_t*y_flat + beta_t*score)

        # Reshape back to (B, S, 2)
        y = y_flat.reshape(B, S, 2)
    return y

# Backward simulation of the SDE using Euler-Maruyama in normalized space.
@pt.no_grad()
def sample_sgm_em(score_model : ScoreNetwork,
                  xA: pt.Tensor,
                  xB: pt.Tensor,
                  s_grid : pt.Tensor,
                  dt: float = 5e-3 
                ) -> pt.Tensor:
    assert xA.ndim == 2 and xA.shape[1] == 2, f"xA must be (B,2), got {xA.shape}"
    assert xB.ndim == 2 and xB.shape[1] == 2, f"xB must be (B,2), got {xB.shape}"
    assert xA.shape == xB.shape, f"xA and xB must have same shape, got {xA.shape} and {xB.shape}"
    if s_grid.ndim != 1:
        s_grid = s_grid.flatten()
    B = xA.shape[0]
    S = s_grid.shape[0]
    device = xA.device
    dtype = xA.dtype

    # Expand on a Cartesian 'grid' but store in 3d 
    # xA_exp, xB_exp: (B, S, 2)
    # s_exp: (B, S)
    xA_exp = xA[:, None, :].expand(B, S, 2)
    xB_exp = xB[:, None, :].expand(B, S, 2)
    s_exp = s_grid[None, :].expand(B, S)

    # Flatten for score-model input
    xA_flat = xA_exp.reshape(B * S, 2)   # (B*S, 2)
    xB_flat = xB_exp.reshape(B * S, 2)   # (B*S, 2)
    s_flat  = s_exp.reshape(B * S)       # (B*S,)

    # Random noise
    y = pt.randn( (B, S, 2), device=device, dtype=dtype)

    n_steps = int(df.T / dt)
    dt = df.T / n_steps
    for n in range(n_steps):
        print('t =', n*dt)
        t_diff = n * dt
        t = (df.T - t_diff) * pt.ones( (B*S,), device=device, dtype=dtype )  # network expects "forward time" t
        t = pt.clamp(t, min=1e-3)

        y_flat = y.reshape(B * S, 2)

        # Compute the score
        score = score_model( y_flat, t, xA_flat, xB_flat, s_flat )

        # Do one backward EM step
        beta_t = df.beta(t)[:, None]
        y_flat = y_flat + dt*(0.5*beta_t*y_flat + beta_t*score) #+ pt.sqrt(beta_t*dt)*pt.randn_like(y_flat)

        # Reshape back to (B, S, 2)
        y = y_flat.reshape(B, S, 2)
    return y

# @pt.no_grad()
# def sample_sgm_heun( score_model : ScoreNetwork,
#                      cond_norm: pt.Tensor,
#                      dt: float = 5e-3,
#                      tmin: float = 1e-4,
#                      power: float = 2.0, 
#                      n_grid : int = 100) -> pt.Tensor:
#     B = cond_norm.shape[0]
#     device = cond_norm.device
#     dtype = cond_norm.dtype
#     y = pt.randn((B, 2 * n_grid), device=device, dtype=dtype)

#     # number of steps (keep your original interface)
#     n_steps = int(1.0 / dt)

#     # Non-uniform time grid: t goes from 1 -> tmin with more resolution near tmin
#     # u in [0,1], then t(u) = tmin + (1 - tmin) * (1 - u)^power
#     u = pt.linspace(0.0, 1.0, n_steps + 1, device=device, dtype=dtype)
#     t_grid = tmin + (1.0 - tmin) * (1.0 - u).pow(power)

#     for n in range(n_steps):
#         t0 = t_grid[n].expand(B)       # (B,)
#         t1 = t_grid[n + 1].expand(B)   # (B,)
#         dt_step = (t0 - t1)            # positive scalar (as tensor): step size in reverse-time

#         print(f"step {n:5d}/{n_steps}  t={t0[0].item():.6f}  dt={dt_step[0].item():.6f}")

#         # Drift at (y, t0)
#         beta0 = beta(t0)[:, None]  # (B,1)
#         score0 = score_model(y, t0, cond_norm)  # (B, 2*n_grid)
#         f0 = 0.5 * beta0 * y + beta0 * score0   # (B, 2*n_grid)

#         # Noise for predictor/corrector (Heun for SDE)
#         z = pt.randn_like(y)
#         g0 = pt.sqrt(beta0 * dt_step[:, None])
#         y_pred = y + f0 * dt_step[:, None] + g0 * z

#         # Drift at (y_pred, t1)
#         beta1 = beta(t1)[:, None]
#         score1 = score_model(y_pred, t1, cond_norm)
#         f1 = 0.5 * beta1 * y_pred + beta1 * score1

#         # Heun Update
#         y = y + 0.5 * (f0 + f1) * dt_step[:, None] + g0 * z

#     return y