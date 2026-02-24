from dataclasses import dataclass
import numpy as np
from typing import List

from .integration import rk4_integrate_ode

"""
NOTE:
If your demos have different lengths, resample each to the same number of steps (and same tau, dt) before calling fit, so t and x are shared.
"""


@dataclass
class DMPModel:
    """Everything needed to rollout a DMP and compute coupling features"""
    weights: np.ndarray # shape: (n_joints, n_basis_functions)
    centers: np.ndarray # shape: (n_basis_functions,) in phase
    widths: np.ndarray # shape: (n_basis_functions,)
    alpha_canonical: float
    alpha_transformation: float
    beta_transformation: float
    tau: float
    n_joints: int

def _rbf_normalized(x: np.ndarray, centers: np.ndarray, widths: np.ndarray) -> np.ndarray:
    """Compute RBFs normalized to sum to 1"""
    psi = np.exp(-widths * (x[:, None] - centers[None, :]) ** 2)
    return psi / (psi.sum(axis=1, keepdims=True) + 1e-10)

def fit(demos: List[np.ndarray],
        tau: float,
        dt: float,
        n_basis_functions: int,
        alpha_canonical: float,
        alpha_transformation: float,
        beta_transformation: float) -> DMPModel:
    """
    fit a DMP from a list of joints trajectories. Each demo is (T, 5)
    Uses same tau for all; if demos have different lengths, they are resampled to the same T
    """

    # Stack and use one nominal duration.
    
    T = demos[0].shape[0]
    t = np.arange(T) * dt
    x = np.exp(-alpha_canonical * t / tau) # Canonical system, phase 1 -> 0

    # RBF centers in phase (exponentially spaced so more near 1)
    centers = np.exp(-np.linspace(0, 1, n_basis_functions)*alpha_canonical) # 1 to ~0
    widths = np.ones(n_basis_functions) * (n_basis_functions ** 1.5) / (centers.max() - centers.min() + 1e-10) # Wide at edges, narrow in middle

    # For each joint, collect target f from all demos and solves for weights.
    n_joints = demos[0].shape[1]
    weights = np.zeros((n_joints, n_basis_functions))

    for joint in range(n_joints):
        phi_list = []
        f_list = []

        for q in demos:
            dq = np.gradient(q[:, joint], dt) # angular velocity
            ddq = np.gradient(dq, dt) # angular acceleration

            q0_joint = q[0, joint]
            g_joint = q[-1, joint]
            diff_g_q0 = g_joint - q0_joint
            if np.abs(diff_g_q0) < 1e-9:
                diff_g_q0 = 1.0 # Avoid division by zero

            # Target forcing term
            f_target = (tau**2 * ddq - alpha_transformation * beta_transformation * (g_joint - q[:, joint]) + alpha_transformation * dq) / diff_g_q0
            phi = _rbf_normalized(x, centers, widths) # Shape: (T, n_basis_functions)
            phi_list.append(phi)
            f_list.append(f_target)

        phi_all = np.vstack(phi_list) 
        f_all = np.concatenate(f_list) 

        # Least squares: phi_all @ w = f_all
        w, *_ = np.linalg.lstsq(phi_all, f_all, rcond=None)
        weights[joint, :] = w

    #q0_fit = np.array([demos[0][0, joint] for joint in range(n_joints)])
    #g_fit = np.array([demos[0][-1, joint] for joint in range(n_joints)])

    return DMPModel(weights=weights, 
                    centers=centers, 
                    widths=widths, 
                    alpha_canonical=alpha_canonical, 
                    alpha_transformation=alpha_transformation, 
                    beta_transformation=beta_transformation, 
                    tau=tau, 
                    n_joints=n_joints)

def rollout_simple(model: DMPModel,
            q0: np.ndarray,
            g: np.ndarray,
            tau: float,
            dt: float) -> np.ndarray:
    """
    Rollout a DMP from a given initial and goal position.

    Uses Euler integration for simplicity.
    """
    n_steps = int(tau / dt) + 1
    q = q0.copy().astype(float) # initial position
    dq = np.zeros_like(q) # initial velocity
    q_gen = np.zeros((n_steps, model.n_joints)) # generated trajectory
    q_gen[0] = q # initial position

    t = 0.0
    for k in range(1, n_steps):
        x = np.exp(-model.alpha_canonical * t / tau)

        for j in range(model.n_joints):
            # Forcing term
            psi = np.exp(-model.widths * (x - model.centers)**2)
            psi_norm = psi / (psi.sum() + 1e-10)
            f = np.dot(psi_norm, model.weights[j])
            
            # Tranformation systems
            ddq = (model.alpha_transformation * model.beta_transformation * (g[j] - q[j]) - model.alpha_transformation * dq[j] + (g[j] - q0[j]) * f) / (tau**2)
            dq[j] += ddq * dt
            q[j] += dq[j] * dt



        q_gen[k] = q # store current position
        t += dt # increment time

    return q_gen

def rollout_rk4(model,
                q0: np.ndarray,
                g: np.ndarray,
                tau: float,
                dt: float) -> np.ndarray:
    """
    Rollout a DMP using RK4 integration.

    State y = [q (n,), dq (n,), x (1,)]
    where x is the canonical phase variable.
    """
    q0 = np.asarray(q0, dtype=float)
    g  = np.asarray(g, dtype=float)

    n_steps = int(tau / dt) + 1
    n = model.n_joints

    # Initial state
    q  = q0.copy()
    dq = np.zeros_like(q)
    x  = 1.0  # canonical phase starts at 1

    y = np.concatenate([q, dq, np.array([x])])

    q_gen = np.zeros((n_steps, n), dtype=float)
    q_gen[0] = q

    # Pre-grab these for speed/clarity
    alpha_c = model.alpha_canonical
    alpha_z = model.alpha_transformation
    beta_z  = model.beta_transformation
    centers = model.centers
    widths  = model.widths
    weights = model.weights  # shape (n_joints, n_basis)

    def forcing(x_scalar: float) -> np.ndarray:
        """Compute forcing f for all joints at phase x."""
        psi = np.exp(-widths * (x_scalar - centers) ** 2)          # (n_basis,)
        psi_norm = psi / (psi.sum() + 1e-10)                      # (n_basis,)
        # f_j = psi_norm dot weights[j]
        return weights @ psi_norm                                  # (n_joints,)

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        """
        y' = [dq, ddq, dx]
        
        RHS = Right Hand Side of the ODE
        """
        q  = y[:n]
        dq = y[n:2*n]
        x  = y[2*n]

        # Canonical system
        dx = -alpha_c * x / tau

        # Forcing (vector over joints)
        f = forcing(x)

        # Your transformation dynamics (vectorized)
        # ddq = (alpha*beta*(g-q) - alpha*dq + (g-q0)*f) / tau^2
        ddq = (alpha_z * beta_z * (g - q)
               - alpha_z * dq
               + (g - q0) * f) / (tau**2)

        dq_dot = ddq
        q_dot  = dq

        return np.concatenate([q_dot, dq_dot, np.array([dx])])

    t = 0.0
    for k in range(1, n_steps):
        # RK4 step
        k1 = rhs(t, y)
        k2 = rhs(t + 0.5*dt, y + 0.5*dt*k1)
        k3 = rhs(t + 0.5*dt, y + 0.5*dt*k2)
        k4 = rhs(t + dt,     y + dt*k3)
        y = y + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # Clamp x to avoid tiny negative due to numeric error
        y[2*n] = max(0.0, y[2*n])

        q_gen[k] = y[:n]
        t += dt

    return q_gen