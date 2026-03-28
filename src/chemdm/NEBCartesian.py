import torch as pt
import openmm as mm
import openmm.unit as unit
from typing import Callable, Optional, Tuple

from chemdm.geometry import center_xyz_torch, kabsch_align_torch

def align_endpoints_cartesian(
    xA_xyz: pt.Tensor,
    xB_xyz: pt.Tensor,
) -> Tuple[pt.Tensor, pt.Tensor]:
    """
    Center xA and xB, then rotate xB onto xA.
    """
    xA_xyz = center_xyz_torch(xA_xyz)
    xB_xyz = center_xyz_torch(xB_xyz)
    xB_xyz = kabsch_align_torch(xB_xyz, xA_xyz)
    return xA_xyz, xB_xyz


def image_dot(a: pt.Tensor, b: pt.Tensor) -> pt.Tensor:
    """
    a, b: (n_images, n_atoms, 3)
    Returns per-image dot product: (n_images, 1, 1)
    """
    return (a * b).sum(dim=(1, 2), keepdim=True)


def image_norm(v: pt.Tensor, eps: float = 1e-12) -> pt.Tensor:
    """
    v: (n_images, n_atoms, 3)
    Returns per-image norm: (n_images, 1, 1)
    """
    return pt.sqrt((v * v).sum(dim=(1, 2), keepdim=True) + eps)


def normalize_images(v: pt.Tensor, eps: float = 1e-12) -> pt.Tensor:
    """
    Normalize one vector per image.
    v: (n_images, n_atoms, 3)
    """
    return v / image_norm(v, eps=eps)


def project_centered_band(x: pt.Tensor) -> pt.Tensor:
    """
    Remove translation image-by-image.
    x: (N_images, n_atoms, 3)
    """
    return x - x.mean(dim=1, keepdim=True)


# ============================================================
# OpenMM energy/force evaluator
# ============================================================

class OpenMMEnergyForceEvaluator:
    """
    Evaluate energies and forces for a band of Cartesian images using one OpenMM context.

    Input coordinates must be in nm.
    Returns:
      E: (N_images,) in kJ/mol
      F: (N_images, n_atoms, 3) in kJ/mol/nm
    """

    def __init__(self, simulation):
        self.simulation = simulation
        self.context = simulation.context
        self.n_atoms = simulation.topology.getNumAtoms()

    def __call__(self, x: pt.Tensor) -> Tuple[pt.Tensor, pt.Tensor]:
        """
        x: (N_images, n_atoms, 3), torch tensor in nm
        """
        assert x.ndim == 3, f"x must be (N_images, n_atoms, 3), got {x.shape}"
        N_images, n_atoms, _ = x.shape
        assert n_atoms == self.n_atoms

        device = x.device
        dtype = x.dtype

        E = pt.empty(N_images, dtype=dtype, device=device)
        F = pt.empty_like(x)

        x_cpu = x.detach().cpu()

        # Unfortunately, OpenMM doesn't do vectorization well yet.
        for i in range(N_images):
            xyz = x_cpu[i].numpy()
            self.context.setPositions(xyz * unit.nanometer)

            state = self.context.getState(getEnergy=True, getForces=True)

            energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            forces = state.getForces(asNumpy=True).value_in_unit(
                unit.kilojoule_per_mole / unit.nanometer
            )

            E[i] = pt.tensor(energy, dtype=dtype, device=device)
            F[i] = pt.tensor(forces, dtype=dtype, device=device)

        return E, F


# ============================================================
# Tangent / NEB force construction
# ============================================================

def force_factory_openmm(
    energy_force: Callable[[pt.Tensor], Tuple[pt.Tensor, pt.Tensor]],
    k: float = 1.0,
):
    """
    Returns neb_force(x), where x is (N_images, n_atoms, 3).

    energy_force(x) returns:
      E      : (N_images,)
      F_true : (N_images, n_atoms, 3)
    where F_true is the physical force = -grad V.
    """

    def neb_force(x: pt.Tensor) -> Tuple[pt.Tensor, pt.Tensor]:
        assert x.ndim == 3, f"x must be (N_images, n_atoms, 3), got {x.shape}"
        N_images = x.shape[0]
        assert N_images >= 3, "Need at least 3 images."

        E, F_true = energy_force(x)

        # Neighbor differences
        dx_fwd = x[2:] - x[1:-1]     # (N-2, n_atoms, 3)
        dx_bwd = x[1:-1] - x[:-2]    # (N-2, n_atoms, 3)

        # Energy differences
        dE_fwd = E[2:] - E[1:-1]     # (N-2,)
        dE_bwd = E[1:-1] - E[:-2]    # (N-2,)

        # Tangent selection
        t_int = pt.zeros_like(dx_fwd)

        inc = (dE_fwd > 0) & (dE_bwd > 0)
        dec = (dE_fwd < 0) & (dE_bwd < 0)
        mixed = ~(inc | dec)

        t_int[inc] = dx_fwd[inc]
        t_int[dec] = dx_bwd[dec]

        if mixed.any():
            w_f = dE_fwd.abs()
            w_b = dE_bwd.abs()

            cond = (E[2:] >= E[:-2])

            t_a = dx_fwd * w_f[:, None, None] + dx_bwd * w_b[:, None, None]
            t_b = dx_fwd * w_b[:, None, None] + dx_bwd * w_f[:, None, None]
            t_mix = pt.where(cond[:, None, None], t_a, t_b)

            t_int[mixed] = t_mix[mixed]

        tau_int = normalize_images(t_int)

        # Interior physical forces
        F_int = F_true[1:-1]

        # Perpendicular component
        F_par = image_dot(F_int, tau_int)
        F_perp = F_int - F_par * tau_int

        # Spring force along tangent
        dist_f = image_norm(dx_fwd).reshape(-1)   # (N-2,)
        dist_b = image_norm(dx_bwd).reshape(-1)   # (N-2,)
        F_spring = k * (dist_f - dist_b)[:, None, None] * tau_int

        F_neb = F_perp + F_spring
        return F_neb, E

    return neb_force


# ============================================================
# Initial path generator
# ============================================================

def generate_linear_cartesian_path(
    xA_xyz: pt.Tensor,
    xB_xyz: pt.Tensor,
    t_grid: pt.Tensor,
) -> pt.Tensor:
    """
    xA_xyz, xB_xyz: (n_atoms, 3)
    t_grid: (N_images,)
    returns: (N_images, n_atoms, 3)

    Linear interpolation between reactants and products typically produces a 
    very high energy path because bonds get squashed, angles skewed, ...
    but it is the only thing we have.
    """
    return xA_xyz[None, :, :] + (xB_xyz - xA_xyz)[None, :, :] * t_grid[:, None, None]


# ============================================================
# Main Cartesian NEB driver
# ============================================================

def computeMEP_openmm_cartesian(
    evaluator: OpenMMEnergyForceEvaluator,
    xA_xyz: pt.Tensor,   # (n_atoms, 3), nm
    xB_xyz: pt.Tensor,   # (n_atoms, 3), nm
    N: int,
    k: float,
    n_steps: int,
    *,
    lr: float = 1e-3,
    verbose: bool = False,
    generate_initial_path: Optional[Callable] = None,
    project_band: bool = True,
) -> Tuple[pt.Tensor, pt.Tensor, float]:
    """
    Cartesian NEB with OpenMM energies/forces.

    Parameters
    ----------
    evaluator : OpenMMEnergyForceEvaluator
    xA_xyz, xB_xyz : (n_atoms, 3)
        Endpoint coordinates in nm
    N : int
        Number of intervals, so total images = N+1
    k : float
        Spring constant
    n_steps : int
        Number of optimizer steps
    lr : float
        Adam learning rate
    verbose : bool
    generate_initial_path : callable or None
        If None, use simple linear interpolation
    project_band : bool
        Remove translation image-by-image after each step

    Returns
    -------
    x0        : initial band, shape (N+1, n_atoms, 3)
    x_optimal : best band found, shape (N+1, n_atoms, 3)
    F_optimal : best mean NEB force norm
    """
    device = xA_xyz.device
    dtype = xA_xyz.dtype

    # Align endpoints to remove rigid motion mismatch
    xA_xyz, xB_xyz = align_endpoints_cartesian(xA_xyz, xB_xyz)

    t_grid = pt.linspace(0.0, 1.0, N + 1, device=device, dtype=dtype)

    if generate_initial_path is not None:
        x0 = generate_initial_path(xA_xyz, xB_xyz, t_grid)
    else:
        x0 = generate_linear_cartesian_path(xA_xyz, xB_xyz, t_grid)

    # Optimize only interior images
    x_inner = pt.nn.Parameter(x0[1:-1].clone())   # (N-1, n_atoms, 3)
    optimizer = pt.optim.Adam([x_inner], lr=lr)
    scheduler = pt.optim.lr_scheduler.StepLR( optimizer, step_size=2000, gamma=0.1 )

    neb_force = force_factory_openmm(evaluator, k)

    F_optimal = pt.inf
    x_optimal = None

    for step in range(n_steps):
        optimizer.zero_grad(set_to_none=True)

        # Full band
        x = pt.cat([xA_xyz[None, :, :], x_inner, xB_xyz[None, :, :]], dim=0)

        # Compute NEB force
        F, E = neb_force(x)

        # Gradient descent with manual gradient
        x_inner.grad = -F
        optimizer.step()
        scheduler.step()

        if project_band:
            with pt.no_grad():
                x_proj = pt.cat([xA_xyz[None, :, :], x_inner, xB_xyz[None, :, :]], dim=0)
                x_proj = project_centered_band(x_proj)
                x_inner.data.copy_(x_proj[1:-1])

        F_norms = image_norm(F).reshape(-1)
        F_mean = F_norms.mean().item()

        if F_mean < float(F_optimal):
            F_optimal = F_mean
            x_optimal = pt.cat([xA_xyz[None, :, :], x_inner, xB_xyz[None, :, :]], dim=0).detach().clone()

        if verbose and (step % 100 == 0 or step == n_steps - 1):
            maxF = F_norms.max().item()
            meanF = F_norms.mean().item()
            lr = float(scheduler.get_last_lr()[-1])
            print( f"step {step:5d} | ", f"mean|F| {meanF:.3e} | ", f"max|F| {maxF:.3e} | ", 
                   f"Emin {E.min().item():.4f} | ", f"Emax {E.max().item():.4f} | ", f"lr {lr:.3e}")

    if x_optimal is None:
        x_optimal = pt.cat([xA_xyz[None, :, :], x_inner, xB_xyz[None, :, :]], dim=0).detach()

    return x0.detach(), x_optimal, F_optimal