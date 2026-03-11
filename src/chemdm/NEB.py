import torch as pt
from typing import Callable, Tuple, Optional

def force_factory(V: Callable[[pt.Tensor], pt.Tensor], k: float = 1.0):
    """
    Returns an neb_force(x) suitable for optimizing an NEB band.
    The force F_i is the NEB force using energy-weighted tangents.
    """

    def _normalize(v: pt.Tensor, eps: float = 1e-12) -> pt.Tensor:
        return v / (pt.linalg.norm(v, dim=-1, keepdim=True) + eps)

    def neb_force(x: pt.Tensor) -> pt.Tensor:
        """
        x: (N, d) band of images including endpoints.
        returns: scalar objective
        """
        assert x.ndim == 2, f"x must be (N, d), got {x.shape}"
        N, d = x.shape
        assert N >= 3, "Need at least 3 images (2 endpoints + >=1 interior)."

        # Make sure we can differentiate V wrt x
        x = x.requires_grad_(True)

        # Energies E_i = V(x_i)
        E = V(x)

        # Gradients dV/dx for all images (N, d)
        g = pt.autograd.grad( E.sum(), x, create_graph=True )[0]

        # Neighbor differences
        dx_fwd = x[2:] - x[1:-1]    # (N-2, d)  x_{i+1}-x_i
        dx_bwd = x[1:-1] - x[:-2]   # (N-2, d)  x_i-x_{i-1}

        # Energy differences
        dE_fwd = E[2:] - E[1:-1]    # (N-2,)
        dE_bwd = E[1:-1] - E[:-2]   # (N-2,)

        # Masks for monotonic segments
        # Increasing: E_{i+1} > E_i > E_{i-1}  => use forward tangent
        t_int = pt.zeros((N - 2, d), dtype=x.dtype, device=x.device)
        inc = (dE_fwd > 0) & (dE_bwd > 0)
        dec = (dE_fwd < 0) & (dE_bwd < 0)
        mixed = ~(inc | dec)
        t_int[inc] = dx_fwd[inc]
        t_int[dec] = dx_bwd[dec]

        # Energy-weighted rule on mixed cases:
        # If E_{i+1} >= E_{i-1}, weight forward by |E_{i+1}-E_i| and backward by |E_i-E_{i-1}|
        # Else swap weights (equivalently, still "point toward higher-energy neighbor")
        if mixed.any():
            w_f = dE_fwd.abs()  # |E_{i+1}-E_i|
            w_b = dE_bwd.abs()  # |E_i-E_{i-1}|

            cond = (E[2:] >= E[:-2])  # (N-2,)
            # Two possible weighted combos
            t_a = dx_fwd * w_f[:, None] + dx_bwd * w_b[:, None]
            t_b = dx_fwd * w_b[:, None] + dx_bwd * w_f[:, None]
            t_mix = pt.where(cond[:, None], t_a, t_b)

            t_int[mixed] = t_mix[mixed]

        # Normalize interior tangents
        tau_int = _normalize(t_int)  # (N-2, d)
        g_int = g[1:-1]  # (N-2, d)
        g_par = (g_int * tau_int).sum(dim=-1, keepdim=True)  # (N-2, 1)
        F_perp = -g_int + g_par * tau_int                    # (N-2, d)

        # Spring force along tangent: k( |x_{i+1}-x_i| - |x_i-x_{i-1}| ) tau
        dist_f = pt.linalg.norm(dx_fwd, dim=-1)  # (N-2,)
        dist_b = pt.linalg.norm(dx_bwd, dim=-1)  # (N-2,)
        F_spring_par = k * (dist_f - dist_b)[:, None] * tau_int  # (N-2, d)

        # Objective: drive NEB force to zero (endpoints excluded)
        F = F_perp + F_spring_par  # (N-2, d)
        return F

    return neb_force

def computeMEP( V : Callable[[pt.Tensor], pt.Tensor], 
                xA : pt.Tensor, # (d,)
                xB : pt.Tensor, # (d,)
                N : int,
                k : float,
                n_steps : int,
                *,
                lr : float = 1e-3,
                verbose : bool = False,
                generate_initial_path : Optional[Callable] = None,
              ) -> Tuple[pt.Tensor, pt.Tensor, float]:
    """
    Connect xA and xB using the Nudged Elastic Band method. Implemented in internal coordinates
    with a unique mapping q <-> V(q).

    Returns the inital guess for the path as well as the full trajectory.
    """
    device = xA.device
    dtype = xA.dtype

    # Initialize the trajectory linearly.
    xA = xA.flatten()
    xB = xB.flatten()
    t_grid = pt.linspace(0.0, 1.0, N+1, device=device, dtype=dtype )
    if generate_initial_path is not None:
        x0 = generate_initial_path( xA, xB, t_grid )
    else:
        x0 = xA + (xB - xA) * t_grid[:,None]

    # Optimize only interior images
    x_inner = pt.nn.Parameter(x0[1:-1].clone())  # (N-1,d)
    optimizer = pt.optim.Adam([x_inner], lr=lr)
    scheduler = pt.optim.lr_scheduler.CosineAnnealingLR( optimizer, T_max=n_steps, eta_min=lr/100)

    neb_force = force_factory( V, k )
    F_optimal = pt.inf
    for step in range(n_steps):
        optimizer.zero_grad( set_to_none=True )

        # Rebuild full band with fixed endpoints
        with pt.no_grad():
            x = pt.cat([xA[None, :], x_inner, xB[None, :]], dim=0)  # (N+1,d)

        # Compute NEB force on interior images
        F = neb_force( x )  # (N-1,d)
        x_inner.grad = (-F)  # gradient descent: x <- x - lr * grad = x + lr * F
        optimizer.step()
        scheduler.step( )

        F_mean = pt.linalg.norm(F, dim=-1).mean().item()
        if float(F_mean) < F_optimal:
            F_optimal = F_mean
            x_optimal = pt.clone(x).detach()

        # Print optimization information
        if verbose and (step % 200 == 0 or step == n_steps - 1):
            maxF = pt.linalg.norm(F, dim=-1).max().item()
            meanF = pt.linalg.norm(F, dim=-1).mean().item()
            print(f"step {step:5d}  max|F| {maxF:.3e}  mean|F| {meanF:.3e} ")

    # Return final band
    if F_optimal == pt.inf:
        with pt.no_grad():
            x_optimal = pt.cat([xA[None, :], x_inner, xB[None, :]], dim=0).detach()
    return x0, x_optimal, F_optimal