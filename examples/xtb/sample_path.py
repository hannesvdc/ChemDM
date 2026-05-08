import math
import torch as pt

from chemdm.MoleculeGraph import Molecule

@pt.no_grad()
def cosine_alpha_bar(T, device, dtype):
    s = 0.008
    steps = pt.linspace(0, T, T + 1, dtype=dtype, device=device)
    f = pt.cos((steps / T + s) / (1.0 + s) * (math.pi / 2.0)) ** 2
    f = f / f[0]
    return f[1:].to(dtype=dtype)

@pt.no_grad()
def ddim_step(c_t : pt.Tensor, eps_pred : pt.Tensor, t, t_prev, alpha_bar, clip_c0 : float = 5.0 ):
    alpha_t = alpha_bar[t].clamp_min(1e-4)
    sqrt_one_minus_alpha_t = pt.sqrt(1.0 - alpha_t)

    c0_pred = (c_t - sqrt_one_minus_alpha_t * eps_pred) / pt.sqrt(alpha_t)
    c0_pred = c0_pred.clamp(-clip_c0, clip_c0)

    if t_prev < 0:
        alpha_prev = pt.tensor(1.0, device=c_t.device, dtype=c_t.dtype)
    else:
        alpha_prev = alpha_bar[t_prev]

    c_prev = pt.sqrt(alpha_prev) * c0_pred + pt.sqrt(1.0 - alpha_prev) * eps_pred
    return c_prev


@pt.no_grad()
def sample_path(diffusion_model : pt.nn.Module, 
                xA : Molecule, 
                xB : Molecule, 
                s : pt.Tensor, 
                x_newton : Molecule, 
                residual_scale : float, 
                T : int, 
                n_steps : int = 20):
    alpha_bar = cosine_alpha_bar(T, s.device, s.dtype)

    x_base = x_newton.x
    c_t = pt.randn_like(x_base)

    times = pt.linspace(T - 2, 0, n_steps, device=x_base.device).round().long()
    times = pt.unique_consecutive(times)

    for i, t in enumerate(times):
        t_prev = times[i + 1] if i + 1 < len(times) else pt.tensor(-1, device=x_base.device)
        t_atom = pt.full((len(xA.Z),), int(t.item()), device=x_base.device, dtype=pt.long)
        t_norm = t_atom.to(dtype=x_base.dtype) / float(T - 1)

        eps_pred = diffusion_model(xA, xB, s, x_newton, c_t, t_norm)
        c_t = ddim_step(c_t, eps_pred, int(t.item()), int(t_prev.item()), alpha_bar)

    gate = 4.0 * s[:, None] * (1.0 - s[:, None])
    return x_base + gate * residual_scale * c_t