import numpy as np
import scipy.optimize as opt
import scipy.sparse as sp
import torch as pt
import openmm as mm
import openmm.unit as unit

from chemdm.diagnostics import has_plateaued, has_started_increasing
from chemdm.Logger import LSQLogger

KJ_MOL_TO_EV = 0.01036427230133138 # eV / kJ
KJ_MOL_NM_TO_EV_A = KJ_MOL_TO_EV / 10.0


def evaluate_openmm_xtb(context: mm.Context, R_A: np.ndarray):
    """
    R_A: (n_atoms, 3), Angstrom

    returns:
        energy_eV: float
        forces_eV_A: (n_atoms, 3), eV / Angstrom
    """
    context.setPositions(R_A * unit.angstrom)

    state = context.getState(getEnergy=True, getForces=True)

    E_kj_mol = state.getPotentialEnergy().value_in_unit( unit.kilojoule_per_mole )

    F_kj_mol_nm = state.getForces(asNumpy=True).value_in_unit(  unit.kilojoule_per_mole / unit.nanometer ) # type: ignore

    E_eV = E_kj_mol * KJ_MOL_TO_EV
    F_eV_A = np.asarray(F_kj_mol_nm) * KJ_MOL_NM_TO_EV_A

    return float(E_eV), F_eV_A


def evaluate_path(context: mm.Context, path_A: np.ndarray):
    """
    path_A: (n_images, n_atoms, 3), Angstrom

    returns:
        energies_eV: (n_images,)
        forces_eV_A: (n_images, n_atoms, 3)
    """
    energies = []
    forces = []

    for R_A in path_A:
        E, F = evaluate_openmm_xtb(context, R_A)
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


def normalize_image_vector(a: np.ndarray, eps: float = 1e-12):
    return a / image_norm(a, eps=eps)


def neb_force( x: np.ndarray,       # (M, n_atoms, 3)
               E: np.ndarray,       # (M,)
               F_true: np.ndarray,  # (M, n_atoms, 3), physical force = -grad E
               k: float, ):
    """
    Returns NEB force on interior images only.

    returns:
        F_neb: (M-2, n_atoms, 3)
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

    tau = normalize_image_vector(t)

    F_int = F_true[1:-1]

    F_parallel = image_dot(F_int, tau) * tau
    F_perp = F_int - F_parallel

    dist_f = image_norm(dx_fwd).squeeze((-2, -1))
    dist_b = image_norm(dx_bwd).squeeze((-2, -1))

    F_spring = k * (dist_f - dist_b)[:, None, None] * tau

    return F_perp + F_spring

def neb_jac_sparsity(n_inner: int, n_atoms: int):
    block_size = n_atoms * 3
    n = n_inner * block_size

    rows = []
    cols = []
    for i in range(n_inner):
        row_start = i * block_size
        dependent_images = [i]
        if i - 1 >= 0:
            dependent_images.append(i - 1)
        if i + 1 < n_inner:
            dependent_images.append(i + 1)

        for j in dependent_images:
            col_start = j * block_size

            # Dense block: every residual coordinate of image i
            # may depend on every coordinate of image j.
            for r in range(block_size):
                for c in range(block_size):
                    rows.append(row_start + r)
                    cols.append(col_start + c)

    data = np.ones(len(rows), dtype=bool)

    return sp.coo_matrix(  (data, (rows, cols)), shape=(n, n) ).tocsr()

def neb_adam( context: mm.Context,
              path0_A: np.ndarray,      # (M, n_atoms, 3), includes endpoints
              n_steps: int = 1000,
              lr: float = 1e-3,
              k: float = 1.0,           # eV / A^2
              max_step_A: float = 0.02,
              force_tol: float = 0.03,  # eV / A
            ):
    assert path0_A.ndim == 3
    M, n_atoms, _ = path0_A.shape
    assert M >= 3

    x0 = pt.tensor( path0_A, dtype=pt.float64 )
    xA = x0[0].clone()
    xB = x0[-1].clone()

    x_inner = pt.nn.Parameter( x0[1:-1].clone() )
    opt = pt.optim.Adam([x_inner], lr=lr)

    best_x = None
    best_force = float("inf")
    history = []
    status = "max_steps"

    for step in range(n_steps):
        opt.zero_grad(set_to_none=True)

        path_A = np.concatenate( [ xA[None, :, :], x_inner.detach().cpu().numpy(), xB[None, :, :] ], axis=0 )        
        E_np, F_np = evaluate_path(context, path_A)
        F_neb = neb_force( path_A, E_np, F_np, k)

        # Per-image RMS NEB force, shape (M-2,)
        F_rms_i = np.sqrt( np.mean(F_neb**2, axis=(-2,-1)) )
        maxF = float(F_rms_i.max().item())
        meanF = float(F_rms_i.mean().item())

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

        row = {
            "step": step,
            "max_force_rms": maxF,
            "mean_force_rms": meanF,
            "barrier_eV": barrier,
            "max_step_A": max_disp,
            "best_force_rms": best_force,
        }
        history.append(row)
        print( f"Iter {step:5d}: maxF {maxF:.6e},  meanF {meanF:.6e},  barrier {barrier:.6f} eV,  step {max_disp:.4e} A" )

        if maxF < force_tol:
            status = "converged"
            print('Adam Comverged')
            break

        if has_started_increasing( history, window=50, rel_increase=0.02, ):
            status = "increasing"
            print( 'Adam started to increase')
            break

        if has_plateaued( history, window=50, rel_tol=0.02 ):
            status = "plateau"
            print( 'Adam Plateau Reached')
            break

    if best_x is None:
        best_x = np.concatenate( [ xA[None, :, :], x_inner.detach().cpu().numpy(), xB[None, :, :] ], axis=0 ) 
    E_best, _ = evaluate_path(context, best_x)

    info = { "status": status,
             "best_force_rms": best_force,
             "n_steps": len(history),
             "history": history, }
    return best_x, E_best, info


def neb_least_squares( context: mm.Context,
                       path0 : np.ndarray,
                       k: float = 1.0,           # eV / A^2
                       force_tol: float = 0.03,  # eV / A
                    ):
    M, n_atoms, _ = path0.shape
    n_inner = M - 2
    sparsity = neb_jac_sparsity(n_inner=n_inner, n_atoms=n_atoms)

    xA = np.copy( path0[0] )
    xB = np.copy( path0[-1] )

    # (n_inner, n_atoms, 3) <-> (n_inner * n_atoms * 3, )
    def pack(x_inner_A: np.ndarray) -> np.ndarray:
        return x_inner_A.reshape(-1)
    def unpack(y: np.ndarray ) -> np.ndarray:
        return y.reshape(n_inner, n_atoms, 3)
    def build_path(y: np.ndarray) -> np.ndarray:
        x_inner = unpack(y)
        return np.concatenate( [ xA[None, :, :], x_inner, xB[None, :, :], ], axis=0, )

    logger = LSQLogger( )
    def neb_force_residual( x_inner : np.ndarray ) -> np.ndarray:
        path_A = build_path(x_inner)
        E_np, F_np = evaluate_path(context, path_A)
        F_neb = neb_force(path_A, E_np, F_np, k)

        logger.observe( E_np, F_neb )
        return pack( F_neb )
    def callback( _ ):
        logger.commit()
        if logger.history[-1]["max_force_rms"] < force_tol:
            logger.converged = True
            raise StopIteration

    # Run the least-squares optimizer
    x0 = pack( path0[1:-1, :, :] )
    neb_force_residual( x0 );  callback( [] ) # Log the initial state
    result = opt.least_squares( neb_force_residual, x0, ftol=1e-6, callback=callback, jac_sparsity=sparsity )
    success = bool( result.success ) or logger.converged
    print( 'Least-Squares Converged: ', success )
    
    # Compute relevant return metrics
    path_opt = build_path( result.x )
    E_opt, F_opt = evaluate_path(context, path_opt)
    F_neb_opt = neb_force(path_opt, E_opt, F_opt, k)
    F_rms_i = np.sqrt((F_neb_opt**2).mean(axis=(1, 2)))
    final_max_force_rms = float(F_rms_i.max())

    print(F_rms_i)
    print("worst image:", np.argmax(F_rms_i) + 1)  # +1 because endpoints excluded

    info = { "success": success,
             "message": result.message,
             "cost": float(result.cost),
             "optimality": float(result.optimality),
             "best_force_rms": final_max_force_rms,
             "history": logger.history, 
            }
    return path_opt, E_opt, info

def normalized_arclengths( path : np.ndarray ) -> np.ndarray:
    image_dist = np.linalg.norm( path[1:,:,:] - path[0:-1,:,:], axis=(1,2) )
    image_dist = np.concatenate( ([0], image_dist), axis=0 )
    arclenghts = np.cumsum( image_dist )
    normalized_arclengths = arclenghts / arclenghts[-1]
    return normalized_arclengths

def run_neb_xtb( context: mm.Context,
                 path0_A: np.ndarray,      # (M, n_atoms, 3), includes endpoints
                 n_steps: int = 1000,
                 lr: float = 1e-3,
                 k: float = 1.0,           # eV / A^2
                 max_step_A: float = 0.02,
                 force_tol: float = 0.03,  # eV / A
                ):
    """
    Nudged-Elastic Band implementation OpenMM-xTB forces and Torch/Adam.

    Returns
    -------
    path_opt_A:
        Optimized path, shape (M, n_atoms, 3), Angstrom.

    E_opt_eV:
        Energies of optimized path, shape (M,), eV.

    best_force:
        Best max NEB force encountered, eV / Angstrom.
    """
    
    # Measure how well we can do at all with fixed end points
    _, F0 = evaluate_path(context, path0_A)
    xA_rms = np.sqrt((F0[0] ** 2).mean())
    xB_rms = np.sqrt((F0[-1] ** 2).mean())
    print("xA force RMS:", xA_rms, "eV/A")
    print("xB force RMS:", xB_rms, "eV/A")

    # Do Adam optimization first to get close to the MEP. If it converged: great!
    path_opt_A, E_best, info = neb_adam( context, path0_A, n_steps, lr, k, max_step_A, force_tol )
    if info["status"] == "converged":
        return path_opt_A, E_best, info["best_force_rms"]
    
    # Else run a fine-tuning step using a quasi-Newton method
    path_opt, E_best, info = neb_least_squares( context, path_opt_A, k, force_tol )
    return path_opt, E_best, info["best_force_rms"]