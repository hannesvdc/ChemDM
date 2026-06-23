import sys
import os
import traceback
import numpy as np
import torch as pt

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

from chemdm.Constants import *
from chemdm.diagnostics import has_plateaued, has_started_increasing
from chemdm.opt import EnergyForceEvaluator
from typing import Callable, Optional

_WORKER_POTENTIAL: EnergyForceEvaluator | None = None

def init_xtb_worker( Z: np.ndarray ):
    """
    Called once inside each worker process.
    Creates one persistent XTBPotential per process.
    """
    global _WORKER_POTENTIAL

    print(f"[xtb-worker {os.getpid()}] initializer start", file=sys.stderr, flush=True)

    try:
        from chemdm.xtbSetup import XTBPotential
        print(f"[xtb-worker {os.getpid()}] imported XTBPotential", file=sys.stderr, flush=True)
        _WORKER_POTENTIAL = XTBPotential( Z=np.asarray(Z, dtype=int) )
        print(f"[xtb-worker {os.getpid()}] XTBPotential created", file=sys.stderr, flush=True)
    except BaseException:
        print(f"[xtb-worker {os.getpid()}] initializer failed", file=sys.stderr, flush=True)
        traceback.print_exc()
        raise

def evaluate_potential( potential: EnergyForceEvaluator, 
                        R_A: np.ndarray) -> tuple[float,np.ndarray]:
    """
    potential : EnergyForceEvaluator
    R_A: (n_atoms, 3), Angstrom

    returns:
        energy_kJ: float
        forces_kJ_A: (n_atoms, 3), kJ / mol / Angstrom
    """
    E_eV, F_eV_A = potential.energy_forces( R_A )
    
    E_kj_mol = E_eV / KJ_MOL_TO_EV
    F_kj_mol_A = F_eV_A / KJ_MOL_TO_EV

    return float(E_kj_mol), F_kj_mol_A

def evaluate_potential_worker( R_A: np.ndarray ) -> tuple[float, np.ndarray]:
    """
    Called inside worker process.

    R_A: (n_atoms, 3), Angstrom

    returns:
        energy: kJ/mol
        forces: kJ/mol/Angstrom
    """
    global _WORKER_POTENTIAL
    if _WORKER_POTENTIAL is None:
        raise RuntimeError("xTB worker was not initialized.")

    return evaluate_potential( _WORKER_POTENTIAL, R_A )


def evaluate_path_process_parallel( path_A: np.ndarray,
                                    pool: ProcessPoolExecutor,
                                   ) -> tuple[np.ndarray, np.ndarray]:
    """
    path_A: (M, n_atoms, 3), Angstrom

    returns:
        energies: (M,), kJ/mol
        forces: (M, n_atoms, 3), kJ/mol/Angstrom
    """
    path_A = np.asarray(path_A, dtype=float)

    results = list( pool.map(evaluate_potential_worker, path_A) )
    energies, forces = zip(*results)

    return np.asarray(energies), np.asarray(forces)

def evaluate_path( potential : EnergyForceEvaluator,
                   path_A: np.ndarray ):
    """
    path_A: (n_images, n_atoms, 3), Angstrom

    returns:
        energies_kJ_mol: (n_images,)
        forces_kJ_mol_A: (n_images, n_atoms, 3)
    """
    energies = []
    forces = []

    for R_A in path_A:
        E, F = evaluate_potential( potential, R_A )
        energies.append(E)
        forces.append(F)

    return np.asarray(energies), np.asarray(forces)


def image_dot(a: np.ndarray, b: np.ndarray):
    """
    Dot product over molecular coordinates.

    a, b: (..., n_atoms, 3)

    returns:
        (..., 1, 1)
    """
    return np.sum (a * b, axis=(-2, -1), keepdims=True )


def image_norm(a: np.ndarray, eps: float = 1e-12):
    """
    Norm over molecular coordinates.

    a: (..., n_atoms, 3)

    returns:
        (..., 1, 1)
    """
    return np.sqrt( image_dot(a, a) + eps)


def normalize_image_vector(a: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return a / image_norm(a, eps=eps)


def improved_tangents( x: np.ndarray,   # (M, n_atoms, 3)
                       E: np.ndarray,    # (M,)
                     ) -> np.ndarray:
    """
    Henkelman & Jonsson (2000) improved tangents for the interior images of a
    path. Each interior image points toward its higher-energy neighbor (uphill
    or downhill stretches), with an energy-weighted blend where it is a local
    extremum.

    returns:
        tau: (M-2, n_atoms, 3), unit tangent per interior image.
    """
    dx_fwd = x[2:] - x[1:-1]       # x_{i+1} - x_i
    dx_bwd = x[1:-1] - x[:-2]      # x_i - x_{i-1}

    dE_fwd = E[2:] - E[1:-1]
    dE_bwd = E[1:-1] - E[:-2]

    t = np.zeros_like(dx_fwd)

    inc = (dE_fwd > 0) & (dE_bwd > 0)
    dec = (dE_fwd < 0) & (dE_bwd < 0)
    mixed = ~(inc | dec)

    t[inc] = dx_fwd[inc]
    t[dec] = dx_bwd[dec]

    if mixed.any():
        wf = np.abs( dE_fwd )
        wb = np.abs( dE_bwd )

        ta = dx_fwd * wf[:, None, None] + dx_bwd * wb[:, None, None]
        tb = dx_fwd * wb[:, None, None] + dx_bwd * wf[:, None, None]

        cond = E[2:] >= E[:-2]
        tmix = np.where(cond[:, None, None], ta, tb)

        t[mixed] = tmix[mixed]

    return normalize_image_vector(t)


def _variable_spring_constants( E: np.ndarray, 
                                k_max: float, 
                                dk: float ) -> np.ndarray:
    """Per-spring constants, Eq. (6) of Henkelman, Uberuaga & Jonsson (2000).

    Spring j connects images j and j+1; its constant scales linearly with the
    higher of the two image energies, from k_max for the spring at the barrier
    top (energy E_max) down to k_max - dk for springs in low-energy regions (at
    or below the higher-energy endpoint, E_ref). Stiffer springs near the saddle
    draw more images into that region, improving the reaction-coordinate
    resolution where it matters. dk is the spring-constant range k_max - k_min.

    With dk = 0 (or no barrier above the endpoints) every spring is k_max,
    recovering constant-spring NEB.

    returns:
        k_spring: (M-1,), kJ/mol/Angstrom^2
    """
    E = np.asarray( E, dtype=float )
    E_spring = np.maximum( E[:-1], E[1:] )       # higher energy of each spring's two images
    E_max = E.max()
    E_ref = max( E[0], E[-1] )                   # higher-energy endpoint
    denom = E_max - E_ref

    if dk <= 0.0:
        return np.full( E_spring.shape, k_max )
    if denom <= 0.0:
        return np.full( E_spring.shape, k_max - dk )

    scaled = k_max - dk * (E_max - E_spring) / denom
    return np.where( E_spring > E_ref, scaled, k_max - dk )


def neb_force( x: np.ndarray,        # (M, n_atoms, 3)
               E: np.ndarray,        # (M,)
               F_true: np.ndarray,   # (M, n_atoms, 3), physical force = -grad E
               k: float,             # max spring constant, kJ/mol/Angstrom^2
               dk: float = 0.0,      # spring-constant range for variable springs (Eq. 6)
               climb: bool = False,  # turn the top image into a climbing image (Eq. 5)
             ) -> tuple[np.ndarray, np.ndarray]:
    """NEB force on the interior images.

    The true force is projected perpendicular to the improved tangent and added
    to a spring force along it (variable spring constants when dk > 0). With
    climb=True the highest-energy interior image becomes a climbing image: it
    feels no spring force and the component of the true force along the band is
    inverted, so it moves uphill to the saddle (Henkelman, Uberuaga & Jonsson
    2000, Eq. 5).

    returns:
        F_neb:  (M-2, n_atoms, 3), the force the optimizer follows.
        F_conv: (M-2, n_atoms, 3), the convergence force -- the perpendicular
                true force per image, replaced by the full true force for the
                climbing image (which must vanish in every direction at the
                saddle, not just perpendicular to the band).
    """
    tau = improved_tangents( x, E )                            # (M-2, n_atoms, 3)
    F_int = F_true[1:-1]
    F_perp = F_int - image_dot( F_int, tau ) * tau

    dist_f = image_norm( x[2:] - x[1:-1] ).squeeze((-2, -1))   # (M-2,)
    dist_b = image_norm( x[1:-1] - x[:-2] ).squeeze((-2, -1))
    k_spring = _variable_spring_constants( E, k, dk )          # (M-1,)
    F_spring = (k_spring[1:] * dist_f - k_spring[:-1] * dist_b)[:, None, None] * tau

    F_neb = F_perp + F_spring
    if climb:
        i = int( np.argmax( E[1:-1] ) )                        # interior index of the saddle
        F_neb[i] = F_int[i] - 2.0 * image_dot( F_int[i], tau[i] ) * tau[i]
        F_conv = F_perp.copy()
        F_conv[i] = F_int[i]
    else:
        F_conv = F_perp

    return F_neb, F_conv

def neb_force_metrics( path_A: np.ndarray,
                       E_np: np.ndarray,
                       F_np: np.ndarray,
                       k: float,
                       dk: float = 0.0,
                       climb: bool = False ) -> dict:
    """
    Compute standard NEB diagnostics for a full path.
    """
    F_neb, F_conv = neb_force(path_A, E_np, F_np, k, dk, climb)

    # Per-interior-image RMS force, shape (M-2,)
    F_neb_rms_i = np.sqrt(np.mean(F_neb**2, axis=(-2, -1)))
    F_conv_rms_i = np.sqrt(np.mean(F_conv**2, axis=(-2, -1)))
    rel_E = E_np - E_np[0]

    return {
        "F_neb": F_neb,
        "F_neb_rms_i": F_neb_rms_i,
        "F_perp_rms_i": F_conv_rms_i,
        "max_force_rms": float(F_conv_rms_i.max()),
        "mean_force_rms": float(F_conv_rms_i.mean()),
        "barrier_kJ_mol": float(rel_E.max()),
        "final_kJ_mol": float(rel_E[-1]),
        "worst_image": int(np.argmax(F_conv_rms_i) + 1),  # +1 because endpoints excluded
    }

def neb_adam( neb_energy_and_force: Callable,
              path0_A: np.ndarray,      # (M, n_atoms, 3), includes endpoints
              n_steps: int,
              lr: float,
              k: float,
              max_step_A: float,
              force_tol: float,
              callback : Optional[Callable] = None,
              dk: float = 0.0,
              climb: bool = False,
            ):
    assert path0_A.ndim == 3
    M, _, _ = path0_A.shape
    assert M >= 3, f"There must be at least 3 images along the transition path but got {M}"

    x0 = pt.tensor( path0_A, dtype=pt.float64 )
    xA = x0[0].clone()
    xB = x0[-1].clone()

    x_inner = pt.nn.Parameter( x0[1:-1].clone() )
    opt = pt.optim.Adam( [x_inner], lr=lr )
    def set_optimizer_lr( lr: float):
        for group in opt.param_groups:
            group["lr"] = lr

    lr_min = 1e-7
    step_count = 0

    # Climbing starts only once the regular band is partly converged, so the
    # tangent at the top image is meaningful before it begins to climb.
    climbing = False
    climb_start_tol = max( 5.0 * force_tol, 20.0 )   # kJ/mol/A; climb once the band is roughly shaped. Large + non-critical: the tangent self-refines as the band co-relaxes. 5*tol guards loose tolerances.

    best_x = None
    best_force = float("inf")
    history = []
    lr_history = []
    while lr > lr_min:
        opt.zero_grad( set_to_none=True )

        path_A = np.concatenate( [ xA[None, :, :], x_inner.detach().cpu().numpy(), xB[None, :, :] ], axis=0 )
        E_np, F_np = neb_energy_and_force( path_A )
        F_neb, F_conv = neb_force( path_A, E_np, F_np, k, dk, climb=climbing )

        # Per-image RMS NEB force, shape (M-2,)
        F_rms_i = np.sqrt( np.mean(F_conv**2, axis=(-2,-1)) )
        maxF = float( F_rms_i.max().item() )

        # Switch the climbing image on, reset best-path tracking (the metric
        # jumps as the uphill force is now counted), and recompute this step's
        # force with the climbing image already active.
        if climb and not climbing and maxF < climb_start_tol:
            print( "Switching to Climbing Image NEB", file=sys.stderr )
            climbing = True
            best_x = None
            best_force = float("inf")
            print( '[neb-xtb] climbing image on', file=sys.stderr )
            F_neb, F_conv = neb_force( path_A, E_np, F_np, k, dk, climb=climbing)
            F_rms_i = np.sqrt( np.mean(F_conv**2, axis=(-2,-1)) )
            maxF = float( F_rms_i.max().item() )

        meanF = float( F_rms_i.mean().item() )
        rel_E = E_np - E_np[0]
        barrier = float(rel_E.max())

        # Track best before stepping.
        if maxF < best_force:
            best_force = maxF
            best_x = np.copy( path_A )

        # Adam minimizes. To move along F_neb, use grad = -F_neb.
        grad = pt.tensor( -F_neb )
        x_inner.grad = grad
        old = x_inner.detach().clone()
        opt.step()

        # Cap max per-atom displacement.
        with pt.no_grad():
            disp = x_inner - old
            max_disp = float( pt.linalg.norm(disp, dim=-1).max().item() )

            if max_disp > max_step_A:
                disp *= max_step_A / max_disp
                x_inner.copy_(old + disp)
                max_disp = max_step_A

        row = { "step": step_count,
                "max_force_rms": maxF,
                "mean_force_rms": meanF,
                "barrier_kJ_mol": barrier,
                "max_step_A": max_disp,
                "best_force_rms": best_force,
        }
        history.append(row)
        lr_history.append(row)
        print( f"Iter {step_count:5d}: maxF {maxF:.6e},  meanF {meanF:.6e},  barrier {barrier:.6f} kJ/mol,  step {max_disp:.4e} A", file=sys.stderr )

        if step_count % 50 == 0 and callback is not None:
            callback( step_count, row["best_force_rms"] )
        # When CI-NEB is on, do not declare convergence until the climbing image
        # is active and its full uphill force has fallen below the tolerance.
        if maxF < force_tol and (not climb or climbing):
            status = "converged"
            print('Adam Comverged', file=sys.stderr )
            break

        if has_started_increasing( lr_history, window=6, rel_increase=0.02, ):
            status = "increasing"
            print( 'Adam started to increase. Reducing lr. ', file=sys.stderr )
            lr = 0.5*lr
            set_optimizer_lr( lr )
            lr_history.clear()
        elif has_plateaued( lr_history, window=6, rel_tol=0.02 ):
            status = "plateau"
            print( 'Adam Plateau Reached. Reducing lr. ', file=sys.stderr )
            lr = 0.5*lr
            set_optimizer_lr( lr )
            lr_history.clear()

        step_count += 1
        if step_count > n_steps:
            status = "max_steps"
            print( 'Adam Has Reached the Maximum Number of Steps. ', file=sys.stderr )
            break

    if best_x is None:
        best_x = np.concatenate( [ xA[None, :, :], x_inner.detach().cpu().numpy(), xB[None, :, :] ], axis=0 ) 
    E_best, F_best = neb_energy_and_force( best_x )
    if callback is not None:
        callback( step_count, neb_force_metrics( best_x, E_best, F_best, k, dk, climbing)["max_force_rms"] )

    info = { "status": status,
             "best_force_rms": best_force,
             "n_steps": len(history),
             "history": history, }
    return best_x, E_best, info


def neb_fire( neb_energy_and_force: Callable,
              path0_A: np.ndarray,      # (M, n_atoms, 3), includes endpoints
              max_steps: int,
              k: float,
              max_step_A: float,
              force_tol: float,
              callback : Optional[Callable] = None,
              dk: float = 0.0,
              climb: bool = False,
            ):
    """FIRE relaxation of the NEB band (Bitzek, Koumoutsakos, Gumbsch & Moser,
    PRL 97, 170201, 2006).

    Integrates the inertial dynamics dv/dt = F_neb with an adaptive timestep:
    while the step moves downhill in the force sense (F.v > 0) the timestep grows
    and the velocity is steered toward the force; the instant a step would go
    against the force (F.v <= 0) the velocity is frozen and the timestep shrinks.
    That freeze keeps the band in the basin of the fixed point nearest the
    initial guess, so the converged path does not jump to a different MEP.

    There is no fixed learning rate. max_step_A is a per-atom safety cap on 
    the displacement, not a tuned step size.
     
    FIRE has improved convergence properties over Adam and is typically much faster
    than quasi-Newton methods. FIRE also has the advantage of being first-principled. 

    Parameters
    ----------
    neb_energy_and_force: Callable
        Returns the potential energy and NEB force per image.
    path0_A: ndarray
        Initial guess for the transition path in Angstrom.
    max_steps: int
        Maximum number of FIRE steps.
    k : float
        The NEB spring constant. Serves as maximum sprint constant when `climb = True`.
    max_step_A : float
        Maximal allowed physical step size in Angstrom.
    force_tol: float
        Used to test convergence: `max_i ||F_i|| < force_tol`.
    callback: Optional[Callable]
    dk : float
        Delta k parameter for climbing image NEB. Default 0.0. Not used when `climb = False`.
    climb: bool
        If true, the function uses the climbing-image variant of NEB. Typically improves
        transition state computations.

    Returns
    -------
    best_x: ndarray
        The transition path as computed by this method.
    E_best: ndarray
        Potential energy of every image.
    info: dict    
    """
    xA, xB = path0_A[0], path0_A[-1]
    def full( x_inner ):
        return np.concatenate( [ xA[None], x_inner, xB[None] ], axis=0 )
    x = path0_A[1:-1].astype(np.float64).copy()   # interior images evolve
    v = np.zeros_like( x )

    # FIRE constants. Dimensionless and molecule-independent.
    N_MIN, F_INC, F_DEC, A_START, F_A = 5, 1.1, 0.5, 0.1, 0.99
    dt_init = 0.1 * max_step_A     # conservative seed; FIRE grows it toward dt_max
    dt = dt_init
    dt_max = 10.0 * dt_init
    alpha = A_START
    n_pos = 0

    # Stall safeguard: the global F.v reset can miss local overshoot when the band
    # as a whole keeps descending, leaving stiff images in an undamped limit cycle.
    # If the best force has not improved for STALL steps, hard-reset (zero velocity,
    # restore timestep + steering) to re-inject damping. After MAX_STALLS reset
    # attempts that still produce no new best, the band is at its force floor (it
    # cannot be driven lower with this dt/mass) -- stop and return the best band
    # rather than burn the rest of the step budget orbiting. Counters reset on any
    # genuine improvement, so MAX_STALLS counts *consecutive* fruitless resets.
    STALL = 4 * N_MIN
    MAX_STALLS = 3
    steps_since_improve = 0
    n_stalls = 0

    # Climbing starts only once the band is partly converged (see neb_adam).
    climbing = False
    climb_start_tol = max( 5.0 * force_tol, 20.0 )   # kJ/mol/A; climb once the band is roughly shaped. Large + non-critical: the tangent self-refines as the band co-relaxes. 5*tol guards loose tolerances.

    best_x, best_force = None, float("inf")
    history = []
    status = "max_steps"
    for step in range( max_steps + 1 ):
        path = full( x )
        E, F_true = neb_energy_and_force( path )
        F_neb, F_conv = neb_force( path, E, F_true, k, dk, climb=climbing )
        maxF = float( np.sqrt( np.mean(F_conv**2, axis=(-2, -1)) ).max() )

        # Switch climbing on once the band is partly converged; reset the
        # best-path tracking and the velocity, since the force field changes.
        if climb and not climbing and maxF < climb_start_tol:
            climbing = True
            best_x, best_force = None, float("inf")
            v[:] = 0.0
            print( '[neb-xtb] climbing image on', file=sys.stderr )
            F_neb, F_conv = neb_force( path, E, F_true, k, dk, climb=climbing )
            maxF = float( np.sqrt( np.mean(F_conv**2, axis=(-2, -1)) ).max() )

        if maxF < best_force:
            best_force, best_x = maxF, np.copy(path)
            steps_since_improve = 0
            n_stalls = 0
        else:
            steps_since_improve += 1

        barrier = float( (E - E[0]).max() )
        history.append( { "step": step, "max_force_rms": maxF, "barrier_kJ_mol": barrier,
                          "best_force_rms": best_force, "dt": dt } )
        print( f"FIRE {step:5d}: maxF {maxF:.6e},  barrier {barrier:.6f} kJ/mol,  dt {dt:.4e}", file=sys.stderr )
        if step % 50 == 0 and callback is not None:
            callback( step, best_force )

        if maxF < force_tol and (not climb or climbing):
            status = "converged"
            print( 'FIRE Converged', file=sys.stderr )
            break

        # FIRE update (semi-implicit Euler), following ASE's ordering.
        # FIRE uses unit mass for every atom. We don't want H to accelerate
        # 12x faster than C. The whole molecule must move at once.
        if steps_since_improve >= STALL:
            # Limit cycle the global F.v reset missed: restart damped from rest.
            n_stalls += 1
            if n_stalls >= MAX_STALLS:
                status = "stalled"
                print( f'[neb] stalled at force floor {best_force:.4g} kJ/mol/A after {n_stalls} resets', file=sys.stderr )
                break
            v[:] = 0.0
            dt, alpha, n_pos = dt_init, A_START, 0
            steps_since_improve = 0
            print( f'[neb] stall reset ({n_stalls}/{MAX_STALLS})', file=sys.stderr )
        else:
            P = float( np.sum( F_neb * v ) )
            if P > 0.0:
                v = (1.0 - alpha) * v + alpha * np.linalg.norm(v) * F_neb / (np.linalg.norm(F_neb) + 1e-12)
                n_pos += 1
                if n_pos > N_MIN:
                    dt = min( dt * F_INC, dt_max )
                    alpha *= F_A
            else:
                v[:] = 0.0
                alpha = A_START
                dt *= F_DEC
                n_pos = 0

        # Update the velocity field and update positions.
        v = v + dt * F_neb
        dx = dt * v
        max_disp = float( np.linalg.norm(dx, axis=-1).max() )
        if max_disp > max_step_A:        # safety cap, rarely binds once dt settles
            dx *= max_step_A / max_disp
        x = x + dx

    if best_x is None:
        best_x = full( x )
    E_best, F_best = neb_energy_and_force( best_x )
    if callback is not None:
        callback( step, neb_force_metrics( best_x, E_best, F_best, k, dk, climbing )["max_force_rms"] )

    info = { "status": status, "best_force_rms": best_force, "n_steps": len(history), "history": history }
    return best_x, E_best, info


def normalized_arclengths( path : np.ndarray ) -> np.ndarray:
    image_dist = np.linalg.norm( path[1:,:,:] - path[0:-1,:,:], axis=(1,2) )
    image_dist = np.concatenate( ([0], image_dist), axis=0 )
    arclenghts = np.cumsum( image_dist )
    normalized_arclengths = arclenghts / arclenghts[-1]
    return normalized_arclengths

def run_neb_xtb( Z : np.ndarray,
                 path0_A: np.ndarray,      # (M, n_atoms, 3), includes endpoints
                 n_steps: int = 250,
                 k: float = 1.0/KJ_MOL_TO_EV,  # kJ/mol/A^2
                 max_step_A: float = 0.02,
                 force_tol: float = 2.8945599636993004, # kJ/mol/A
                 max_workers: int = 4,
                 callback : Optional[Callable] = None,
                 dk: Optional[float] = None,
                 climb: bool = True,
                ):
    """
    Climbing-image Nudged-Elastic Band implementation using direct xTB forces
    (Henkelman, Uberuaga & Jonsson 2000): the highest-energy image climbs to the
    saddle while the band relaxes around it, with variable spring constants
    concentrating images near the barrier.

    Unit convention:
        positions: Angstrom
        energies: kJ/mol
        forces: kJ/mol/Angstrom
        k: kJ/mol/Angstrom^2 (maximum spring constant)
        force_tol: kJ/mol/Angstrom

    Parameters
    ----------
    climb: bool
        Enable the climbing image (CI-NEB). The converged band's highest image
        sits at the saddle, so it doubles as the transition state.
    dk: float
        Spring-constant range k_max - k_min for variable springs (Eq. 6).
        Defaults to k/2 (k_min = k/2); pass 0.0 for constant springs.

    Returns
    -------
    path_opt_A:
        Optimized path, shape (M, n_atoms, 3), Angstrom.
    E_opt_kJ_mol:
        Energies of optimized path, shape (M,), kJ/mol.
    best_force:
        Best max NEB force encountered, kJ / mol / Angstrom.
    """
    if dk is None:
        dk = 0.5 * k

    def run_with_evaluator( neb_energy_and_force: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]] ):

        # Measure how well we can do at all with fixed end points
        E0, F0 = neb_energy_and_force( path0_A )
        print("xA force RMS:", np.sqrt((F0[0] ** 2).mean()), "kJ/mol/A")
        print("xB force RMS:", np.sqrt((F0[-1] ** 2).mean()), "kJ/mol/A")
        if callback is not None:
            callback( 0, neb_force_metrics( path0_A, E0, F0, k, dk )["max_force_rms"] )

        path_opt_A, E_best, info = neb_fire( neb_energy_and_force, path0_A, n_steps, k,
                                             max_step_A, force_tol, callback, dk=dk, climb=climb )
        return path_opt_A, E_best, info["best_force_rms"]

    if max_workers <= 1:
        print("[neb-xtb] using serial evaluator", file=sys.stderr, flush=True)
        
        from chemdm.xtbSetup import XTBPotential
        xtb = XTBPotential( Z=Z )

        neb_energy_and_force = lambda path_A: evaluate_path(xtb, path_A)
        return run_with_evaluator( neb_energy_and_force )
    
    # Process-parallel mode.
    print( f"[neb-xtb] using spawn ProcessPoolExecutor with {max_workers} workers", file=sys.stderr, flush=True )
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor( max_workers=max_workers,
                              mp_context=ctx,
                              initializer=init_xtb_worker,
                              initargs=( Z, ),  ) as pool:
        neb_energy_and_force = lambda path_A: evaluate_path_process_parallel( path_A, pool )
        return run_with_evaluator( neb_energy_and_force )