from dataclasses import dataclass
import numpy as np
from typing import List

from scipy.signal import savgol_filter

"""
NOTE:
If your demos have different lengths, resample each to the same number of steps (and same tau, dt) before calling fit, so t and x are shared.
"""


@dataclass
class DMPModel:
    """Parameters required for DMP rollouts and forcing reconstruction."""
    weights: np.ndarray  # shape: (n_joints, n_basis_functions)
    centers: np.ndarray  # shape: (n_basis_functions,) in phase
    widths: np.ndarray  # shape: (n_basis_functions,)
    alpha_canonical: float
    alpha_transformation: float
    beta_transformation: float
    tau: float
    n_joints: int
    curvature_weights: np.ndarray # shape: (n_joints, n_basis_functions)

def _rbf_normalized(x: np.ndarray, centers: np.ndarray, widths: np.ndarray) -> np.ndarray:
    """Compute Gaussian RBF activations normalized row-wise."""
    psi = np.exp(-widths * (x[:, None] - centers[None, :]) ** 2)
    return psi / (psi.sum(axis=1, keepdims=True) + 1e-10)

def _validate_and_get_demo_shape(demos: List[np.ndarray]) -> tuple[int, int]:
    """Validate demos list and return (T_demo, n_joints)."""
    if not demos:
        raise ValueError("demos must be a non-empty list")
    if demos[0].ndim != 2:
        raise ValueError(f"demos[0] must be 2D (T, n_joints), got shape {demos[0].shape}")

    T_demo = int(demos[0].shape[0])
    n_joints = int(demos[0].shape[1])
    for i, q in enumerate(demos):
        if q.ndim != 2:
            raise ValueError(f"demos[{i}] must be 2D (T, n_joints), got shape {q.shape}")
        if q.shape[1] != n_joints:
            raise ValueError(f"demos[{i}] has n_joints={q.shape[1]}, expected {n_joints}")
        if q.shape[0] != T_demo:
            raise ValueError(
                "All demos must have the same length before calling fit "
                f"(got demos[0].shape[0]={T_demo} but demos[{i}].shape[0]={q.shape[0]})."
            )
    return T_demo, n_joints

def _centers_and_widths(alpha_canonical: float, n_basis_functions: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Build phase-space RBF centers and widths used by fit.
    """
    centers = np.exp(-np.linspace(0, 1, n_basis_functions) * alpha_canonical)  # 1 -> ~0
    if n_basis_functions <= 1:
        widths = np.array([1.0], dtype=np.float64)
    else:
        d = np.diff(centers)
        d = np.hstack((d, [d[-1]]))
        widths = 1.0 / (d ** 2 + 1e-12)
    return centers, widths

def savgol_estimation(
    q: np.ndarray,
    *,
    dt: float,
    savgol_window_length: int = 11,
    savgol_polyorder: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate dq and ddq using a Savitzky-Golay smoothing + derivative pass.

    Falls back to gradient-based derivatives when there are too few points
    for a valid Savitzky-Golay setup.
    """
    y = np.asarray(q, dtype=float)
    T = y.size

    wl = min(int(savgol_window_length), T if T % 2 == 1 else T - 1)
    if wl < 3:
        dq = np.gradient(y, dt)
        ddq = np.gradient(dq, dt)
        return dq, ddq
    if wl % 2 == 0:
        wl -= 1
    if wl < 3:
        dq = np.gradient(y, dt)
        ddq = np.gradient(dq, dt)
        return dq, ddq

    po = int(savgol_polyorder)
    if po >= wl:
        po = wl - 1
    if po < 1:
        po = 1

    y_smooth = savgol_filter(y, window_length=wl, polyorder=po, mode="interp")
    dq = savgol_filter(y_smooth, window_length=wl, polyorder=po, deriv=1, delta=dt, mode="interp")
    ddq = savgol_filter(y_smooth, window_length=wl, polyorder=po, deriv=2, delta=dt, mode="interp")
    return dq, ddq

def estimate_derivatives(
    q: np.ndarray,
    *,
    dt: float,
    derivative_method: str = "savgol",
    savgol_window_length: int = 11,
    savgol_polyorder: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate dq and ddq from a 1D trajectory q(t). 
    This is used to compute the forcing term for each joint in the transformation system.
    
    Args:
        q: np.ndarray: the trajectory
        dt: float: the time step
        derivative_method: str: the method to use for derivative estimation
        savgol_window_length: int: the window length for Savitzky-Golay filter
        savgol_polyorder: int: the polynomial order for Savitzky-Golay filter

    Returns:
        tuple[np.ndarray, np.ndarray]: the estimated dq and ddq
    """
    # 1. Convert the trajectory to a numpy array
    q = np.asarray(q, dtype=float)
    if q.ndim != 1:
        raise ValueError(f"Expected 1D q, got shape {q.shape}")
    if q.size < 3: # Not enough samples to do a proper Savitzky-Golay estimate.
        dq = np.gradient(q, dt)
        ddq = np.gradient(dq, dt)
        return dq, ddq

    # 2. Get the method to use for derivative estimation
    method = derivative_method.strip().lower()

    # 3. Estimate the derivatives using gradient method if specified
    if method == "gradient":
        dq = np.gradient(q, dt)
        ddq = np.gradient(dq, dt)
        return dq, ddq

    if method != "savgol":
        raise ValueError(f"Unknown derivative_method '{derivative_method}'. Use 'gradient' or 'savgol'.")

    # 4. Estimate derivatives using Savitzky-Golay helper.
    return savgol_estimation(
        q,
        dt=dt,
        savgol_window_length=savgol_window_length,
        savgol_polyorder=savgol_polyorder,
    )

def curvature_coupling(q, g, x, centers, widths, curvature_weights):
    """Curvature coupling function
    Args:
        q: current joint position, shape (n_joints,)
        g: goal joint position, shape (n_joints,)
        x: canonical phase, scalar (works as a phase gate)
        centers: centers, shape (n_basis,)
        widths: widths, shape (n_basis,)
        curvature_weights: shape (n_joints, n_basis)
    Returns:
        curvature_coupling: shape (n_joints,)
    """
    psi = np.exp(-widths * (x - centers) ** 2)
    psi_norm = psi / (psi.sum() + 1e-10)

    curvature_direction = curvature_weights @ psi_norm

    diff = g - q
    norm = np.linalg.norm(diff)

    if norm < 1e-5:
        return np.zeros_like(q)

    e = diff / norm
    P_perp = np.eye(q.shape[0]) - np.outer(e, e) # TODO: Check that q.shape[0] is the number of joints
    
    C_curv = x * (P_perp @ curvature_direction)
    return C_curv

def learn_curvature_weights_from_demo(
    demo: np.ndarray,
    model: DMPModel,
    dt: float,
    ridge_lambda: float,
) -> np.ndarray:
    """Learn curvature weights from a demo."""
    T, n_joints = demo.shape
    n_basis = model.centers.shape[0]

    q0 = demo[0]
    g = demo[-1]
    tau = model.tau

    dq = np.zeros_like(demo)
    ddq = np.zeros_like(demo)

    for joint in range(n_joints):
        dq[:, joint], ddq[:, joint] = estimate_derivatives(
            demo[:, joint],
            dt=dt,
            derivative_method="savgol",
            savgol_window_length=11,
            savgol_polyorder=3,
        )

        Phi_rows, Y_rows = [], []

        for k in range(T):
            t = k * dt
            x = canonical_phase(np.array([t]), tau=tau, alpha_canonical=model.alpha_canonical)
            x = float(x)

            q = demo[k]
            dq_k = dq[k]
            ddq_k = ddq[k]

            psi = np.exp(-model.widths * (x - model.centers) ** 2)
            psi_norm = psi / (psi.sum() + 1e-10)
            
            f = x * (model.weights @ psi_norm)

            baseline_numerator = (
                model.alpha_transformation * model.beta_transformation * (g - q) - model.alpha_transformation * dq_k + (g - q0) * f
            )

            observed_numerator = tau**2 * ddq_k

            C_target = observed_numerator - baseline_numerator

            diff = g-q
            norm = np.linalg.norm(diff)
            if norm < 1e-5:
                continue
            e = diff / norm
            P_perp = np.eye(q.shape[0]) - np.outer(e, e) # TODO: Check that q.shape[0] is the number of joints
            
            row_blocks = []
            for i in range(n_basis):
                row_blocks.append(x * psi_norm[i] * P_perp)

            Phi_k = np.concatenate(row_blocks, axis=1)
            Phi_rows.append(Phi_k)
            Y_rows.append(C_target)

            
        Phi = np.vstack(Phi_rows)
        Y = np.concatenate(Y_rows)

        A = Phi.T @ Phi + ridge_lambda * np.eye(Phi.shape[1])
        b = Phi.T @ Y
        c_flat = np.linalg.solve(A, b)
        curvature_weights = c_flat.reshape(n_basis, n_joints).T
        return curvature_weights

def _compute_f_target(
    q_joint: np.ndarray,
    *,
    tau: float,
    dt: float,
    alpha_transformation: float,
    beta_transformation: float,
    diff_g_q0_eps: float,
    savgol_window_length: int,
    savgol_polyorder: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]: 
    """
    Compute DMP forcing target for one joint plus diagnostics.

    Returns:
        f_target, dq, ddq, q0_joint, g_joint, g_minus_q0
    """
    dq, ddq = estimate_derivatives(
        q_joint,
        dt=dt,
        derivative_method="savgol",
        savgol_window_length=savgol_window_length,
        savgol_polyorder=savgol_polyorder,
    )

    q0_joint = float(q_joint[0])
    g_joint = float(q_joint[-1])
    g_minus_q0 = g_joint - q0_joint
    scale = 1.0 if abs(g_minus_q0) < diff_g_q0_eps else g_minus_q0

    f_target = (
        tau**2 * ddq
        - alpha_transformation * beta_transformation * (g_joint - q_joint)
        + alpha_transformation * dq
    ) / scale
    return f_target, dq, ddq, q0_joint, g_joint, g_minus_q0

def _solve_lwr_like_weights(
    *,
    demos: List[np.ndarray],
    phi: np.ndarray,
    n_joints: int,
    n_basis_functions: int,
    tau: float,
    dt: float,
    alpha_transformation: float,
    beta_transformation: float,
    ridge_lambda: float,
    diff_g_q0_eps: float,
    savgol_window_length: int,
    savgol_polyorder: int,
) -> np.ndarray:
    """
    Solve DMP weights with an LWR-style normal-equation solve.

    For each joint, it solves:
        (Phi^T Phi + lambda I) w = Phi^T f_target
    aggregated across all demos.

    return: weights, shape (n_joints, n_basis_functions)
    """
    n_demos = len(demos)
    # Phi is shared across demos, so A is the same per demo and can be scaled once.
    A = n_demos * (phi.T @ phi)
    A_reg = A + ridge_lambda * np.eye(n_basis_functions, dtype=np.float64)

    weights = np.zeros((n_joints, n_basis_functions), dtype=np.float64)

    for joint in range(n_joints):
        # b = sum_d Phi^T f_target_d
        b = np.zeros((n_basis_functions,), dtype=np.float64)
        for demo_idx, q in enumerate(demos):
            q_joint = q[:, joint]
            (
                f_target,
                dq,
                ddq,
                q0_joint,
                g_joint,
                g_minus_q0,
            ) = _compute_f_target(
                q_joint,
                tau=tau,
                dt=dt,
                alpha_transformation=alpha_transformation,
                beta_transformation=beta_transformation,
                diff_g_q0_eps=diff_g_q0_eps,
                savgol_window_length=savgol_window_length,
                savgol_polyorder=savgol_polyorder,
            )

            b += phi.T @ f_target

        weights[joint, :] = np.linalg.solve(A_reg, b)

    return weights

def _solve_lwr_weights(
    phase: np.ndarray,
    f_target: np.ndarray,
    centers: np.ndarray,
    widths: np.ndarray,
    regularization: float = 1e-8,
) -> np.ndarray:
    """
    Solve DMP forcing-term weights using Locally Weighted Regression (LWR).

    Parameters
    ----------
    phase : np.ndarray
        Canonical phase variable, shape (T,).
    f_target : np.ndarray
        Target forcing term, shape (T,) for one joint.
    centers : np.ndarray
        RBF centers, shape (N,).
    widths : np.ndarray
        RBF widths, shape (N,).
    regularization : float
        Small value added to denominator for numerical stability.

    Returns
    -------
    weights : np.ndarray
        Learned LWR weights, shape (N,).
    """

    phase = np.asarray(phase, dtype=float)
    f_target = np.asarray(f_target, dtype=float)
    centers = np.asarray(centers, dtype=float)
    widths = np.asarray(widths, dtype=float)

    if phase.ndim != 1:
        raise ValueError("phase must have shape (T,)")

    if f_target.ndim != 1:
        raise ValueError("f_target must have shape (T,)")

    if phase.shape[0] != f_target.shape[0]:
        raise ValueError("phase and f_target must have the same length")

    n_basis = centers.shape[0]
    weights = np.zeros(n_basis, dtype=float)

    for i in range(n_basis):
        psi_i = np.exp(-widths[i] * (phase - centers[i]) ** 2)

        numerator = np.sum(psi_i * phase * f_target)
        denominator = np.sum(psi_i * phase ** 2) + regularization

        weights[i] = numerator / denominator

    return weights

def _solve_lwr_weights_multi(
    phase: np.ndarray,
    f_target: np.ndarray,
    centers: np.ndarray,
    widths: np.ndarray,
    regularization: float = 1e-8,
) -> np.ndarray:
    """
    Solve DMP forcing-term weights using LWR for multiple joints.

    Parameters
    ----------
    phase : np.ndarray
        Shape (T,).
    f_target : np.ndarray
        Shape (T, D), where D is number of joints.
    centers : np.ndarray
        Shape (N,).
    widths : np.ndarray
        Shape (N,).

    Returns
    -------
    weights : np.ndarray
        Shape (D, N).
    """

    phase = np.asarray(phase, dtype=float)
    f_target = np.asarray(f_target, dtype=float)

    if f_target.ndim == 1:
        return _solve_lwr_weights(
            phase, f_target, centers, widths, regularization
        )[None, :]

    if f_target.shape[0] != phase.shape[0]:
        raise ValueError("f_target must have shape (T, D)")

    n_joints = f_target.shape[1]
    n_basis = len(centers)

    weights = np.zeros((n_joints, n_basis), dtype=float)

    for joint_idx in range(n_joints):
        weights[joint_idx, :] = _solve_lwr_weights(
            phase=phase,
            f_target=f_target[:, joint_idx],
            centers=centers,
            widths=widths,
            regularization=regularization,
        )

    return weights

def canonical_phase(t: np.ndarray, *, tau: float, alpha_canonical: float) -> np.ndarray:
    """Canonical DMP phase variable x(t) = exp(-alpha * t / tau)."""
    t = np.asarray(t, dtype=float)
    return np.exp(-alpha_canonical * t / tau)

def fit(
    demos: List[np.ndarray],
    tau: float,
    dt: float,
    n_basis_functions: int,
    alpha_canonical: float,
    alpha_transformation: float,
    beta_transformation: float,
) -> DMPModel:
    """
    Fit a DMP from a list of joint trajectories.
    This is the main function to fit a DMP model to a list of joint trajectories.

    Args:
        demos: list of trajectories, each of shape (T, n_joints), radians.
    
    Returns:
        DMPModel: the fitted DMP model
    """
    # 1. Validate and get demo shape
    T_demo, n_joints = _validate_and_get_demo_shape(demos)

    # 2. Get canonical phase variable for each trajectory time step.
    t_demo = np.arange(T_demo, dtype=np.float64) * dt
    x = np.exp(-alpha_canonical * t_demo / tau) # canonical phase variable (0 to 1)

    # 3. Get center and width for the basis functions (in phase-space)
    centers, widths = _centers_and_widths(alpha_canonical, n_basis_functions)
    phi = _rbf_normalized(x, centers, widths) * x[:, None]

    # 4. Solve LWR weights
    ridge_lambda = 1e-6
    diff_g_q0_eps = 0.02 # previously 1e-6 (0.02 is about 1.15 deg)
    savgol_window_length = 11 #11
    savgol_polyorder = 3 #3

    weights = _solve_lwr_like_weights(
        demos=demos,
        phi=phi,
        n_joints=n_joints,
        n_basis_functions=n_basis_functions,
        tau=tau,
        dt=dt,
        alpha_transformation=alpha_transformation,
        beta_transformation=beta_transformation,
        ridge_lambda=ridge_lambda,
        diff_g_q0_eps=diff_g_q0_eps,
        savgol_window_length=savgol_window_length,
        savgol_polyorder=savgol_polyorder,
    )

    base_model = DMPModel(
        weights=weights,
        centers=centers,
        widths=widths,
        alpha_canonical=alpha_canonical,
        alpha_transformation=alpha_transformation,
        beta_transformation=beta_transformation,
        tau=tau,
        n_joints=n_joints,
        curvature_weights=np.zeros((n_joints, n_basis_functions)),
    )

    curvature_weights_all = [
        learn_curvature_weights_from_demo(
            demo=demo,
            model=base_model,
            dt=dt,
            ridge_lambda=ridge_lambda,
        )
        for demo in demos
    ]
    curvature_weights = np.mean(curvature_weights_all, axis=0)

    # 5. Return the model
    return DMPModel(
        weights=weights,
        centers=centers,
        widths=widths,
        alpha_canonical=alpha_canonical,
        alpha_transformation=alpha_transformation,
        beta_transformation=beta_transformation,
        tau=tau,
        n_joints=n_joints,
        curvature_weights=curvature_weights,
    )

def rollout_simple(model: DMPModel,
            q0: np.ndarray,
            g: np.ndarray,
            tau: float,
            dt: float) -> np.ndarray:
    """
    Rollout a DMP from a given initial and goal position.
    This is a simple rollout function that uses Euler integration for simplicity.
    Rollout means to generate a trajectory from the initial position to the goal position.

    Args:
        model: DMPModel: the fitted DMP model
        q0: np.ndarray: the initial position
        g: np.ndarray: the goal position
        tau: float: the time constant
        dt: float: the time step

    Returns:
        np.ndarray: the generated trajectory
    
    """
    # 0. Initialize the trajectory
    n_steps = int(round(tau / dt)) + 1 # Number of time steps
    q_gen = np.zeros((n_steps, model.n_joints))  # Generated trajectory
    q = q0.copy().astype(float)  # Current position
    q_gen[0] = q # Initial position
    dq = np.zeros_like(q)  # Current velocity

    t = 0.0 # Time
    for k in range(1, n_steps):
        # 1. Get canonical phase variable
        x = canonical_phase(t, tau=tau, alpha_canonical=model.alpha_canonical)

        # 2. Compute the forcing term for each joint
        for joint in range(model.n_joints):
            # 2.1. Compute scalar-phase RBF activations for this timestep
            psi = np.exp(-model.widths * (x - model.centers) ** 2)
            psi_norm = psi / (psi.sum() + 1e-10)
            f = x * np.dot(psi_norm, model.weights[joint]) # forcing term for each joint

            # 2.2. Compute the transformation system
            ddq = (model.alpha_transformation * model.beta_transformation * (g[joint] - q[joint]) - model.alpha_transformation * dq[joint] + (g[joint] - q0[joint]) * f) / (tau**2)
            # 2.3. Update the velocity and position (Euler integration)
            dq[joint] += ddq * dt
            q[joint] += dq[joint] * dt

        # 3. Update the trajectory and time
        q_gen[k] = q
        t += dt

    return q_gen

def rollout_simple_with_coupling(model: DMPModel,
                          q0: np.ndarray,
                          g: np.ndarray,
                          tau: float,
                          dt: float) -> np.ndarray:
    """
    Rollout a DMP with coupling.
    """
    n_steps = int(tau / dt) + 1
    q_gen = np.zeros((n_steps, model.n_joints))
    q = q0.copy().astype(float)
    dq = np.zeros_like(q)
    q_gen[0] = q
    t = 0.0
    for k in range(1, n_steps):

        x = canonical_phase(t, tau=tau, alpha_canonical=model.alpha_canonical)
        
        psi = np.exp(-model.widths * (x - model.centers) ** 2)
        psi_norm = psi / (psi.sum() + 1e-10)
        
        f = x * (model.weights @ psi_norm)
        
        if model.curvature_weights is not None:
            C_curv = curvature_coupling(q, g, x, model.centers, model.widths, model.curvature_weights)

        ddq = (model.alpha_transformation * model.beta_transformation * (g - q) - model.alpha_transformation * dq + (g - q0) * f + C_curv) / (tau**2)
        dq += ddq * dt
        q += dq * dt
        q_gen[k] = q
        t += dt

    return q_gen

def rollout_rk4(
                model: DMPModel,
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
        # f_j = x * (psi_norm dot weights[j])
        return x_scalar * (weights @ psi_norm)                  # (n_joints,)

    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        """
        y' = [dq, ddq, dx]
        
        RHS = right-hand side of the ODE
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