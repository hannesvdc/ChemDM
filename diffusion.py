import math
import torch as pt
import matplotlib.pyplot as plt

beta_min = 0.001
beta_max = 10.0 # used to be 3.0 but higher is necessary to avoid bias.
T = 1.0 # final diffusion time
beta = lambda t: beta_min + t * (beta_max - beta_min)
alpha = lambda t: beta_min * t + 0.5 * (beta_max - beta_min) * t**2
mean_factor_tensor = lambda t: pt.exp(-0.5 * alpha(t))
var_tensor = lambda t: 1 - pt.exp(-alpha(t))

def forwardSDE(X : pt.Tensor, dt : float) -> pt.Tensor:
    n_steps = int(T / dt)
    assert(abs(n_steps * dt - T) < 1e-10)

    for n in range(n_steps):
        t = n * dt
        X = X - 0.5 * beta(t) * X * dt + math.sqrt(beta(t) * dt) * pt.randn_like(X)

    return X

def p_t(initial_samples : pt.Tensor, # shape (J, d)
        eval_samples : pt.Tensor, # shape (N, d)
        t : float) -> pt.Tensor:
    d = initial_samples.shape[1]
    mt = math.exp(-0.5 * alpha(t)) * initial_samples # (J, d)
    vt = 1.0 - math.exp(-alpha(t))

    # permute the eval samples
    diff = eval_samples[None, :, :] - mt[:, None, :]    # (J, N, d)
    numerator = pt.sum(diff**2, dim=2) # (J, N)
    dist = pt.exp(-numerator / (2.0 * vt)) / (2.0 * math.pi * vt) ** (0.5 * d)

    return pt.mean(dist, dim=0) # Shape (N,)

def nabla_logp_t(initial_samples : pt.Tensor,
                 eval_samples : pt.Tensor,
                 t : float) -> pt.Tensor:
    mt = math.exp(-0.5 * alpha(t)) * initial_samples # (J, d)
    vt = 1.0 - math.exp(-alpha(t))

    # permute the eval samples
    diff = eval_samples[None, :, :] - mt[:, None, :]    # (J, N, d)
    logw = -pt.sum(diff * diff, dim=2) / (2.0 * vt)     # (J, N)
    w = pt.softmax(logw, dim=0)                         # (J, N)

    score = -pt.sum(w[:, :, None] * diff, dim=0) / vt   # (N, d)
    return score

def backwardExactSDE(initial_samples : pt.Tensor,
                Y : pt.Tensor,
                dt : float) -> pt.Tensor:
    n_steps = int(T / dt)
    assert(abs(n_steps * dt - T) < 1e-10)

    for n in range(n_steps):
        print('t =', n*dt)
        t = n * dt
        T_t = T - t
        drift = nabla_logp_t(initial_samples, Y, T_t)

        beta_1_t = beta(T_t)
        Y = Y + dt * (0.5 * beta_1_t * Y + beta_1_t * drift) + math.sqrt(beta_1_t * dt) * pt.randn_like(Y)

    return Y

def sampleInitial(N : int) -> pt.Tensor:
    # Generate circular samples as a test case
    n_points = 8
    angles = 2.0 * math.pi * pt.arange(0, n_points) / n_points
    points = pt.stack((pt.cos(angles), pt.sin(angles)), dim=1)

    # Sample X0 uniformly from these 8 points
    indices = pt.randint(0, n_points, (N,))
    X_0 = points[indices,:]
    
    return X_0

if __name__ == '__main__':
    # Generate circular samples as a test case
    N = 10_000
    X_0 = sampleInitial(N)
    
    # Forward simulation
    print('Simulating the Forward SDE')
    dt = 1e-3
    X_1 = forwardSDE(X_0, dt)

    # Backward simulation with independent random numbers
    print('Simulating the Backward SDE')
    Y_1 = backwardExactSDE(X_0, X_1, dt)

    # Plot the final particles in the 2D plane
    plt.scatter(Y_1[:,0].numpy(), Y_1[:,1].numpy(), label=r"$Y(t=1)$")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.legend()
    plt.show()