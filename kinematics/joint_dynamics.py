from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.signal import savgol_filter


JOINT_NAMES_4DOF = [
    "elbow_flexion",
    "shoulder_elevation",
    "shoulder_azimuth",
    "shoulder_internal_rotation",
]


@dataclass(frozen=True)
class JointLimitsDeg:
    """Per‑joint kinematic limits in degrees / deg/s / deg/s^2."""

    pos_min: np.ndarray  # shape (n_joints,)
    pos_max: np.ndarray  # shape (n_joints,)
    vel_max_abs: np.ndarray  # shape (n_joints,)
    acc_max_abs: np.ndarray  # shape (n_joints,)

    @property
    def n_joints(self) -> int:
        return int(self.pos_min.shape[0])


def default_human_arm_limits_4dof() -> JointLimitsDeg:
    """Reasonable, conservative limits for 4‑DOF human arm model (degrees).

    Order matches JOINT_NAMES_4DOF.
    These can be tuned later if you calibrate per‑subject limits.
    """

    # Position ranges (deg)
    # Elbow flexion:       0   – 150  (0 = extended, 150 = flexed)
    # Shoulder elevation:  0   – 180  (0 = arm up, 180 = arm down)
    # Shoulder azimuth:   -150 – 150  (across body to abducted)
    # Shoulder int. rot.: -120 – 120
    pos_min = np.array([0.0, 0.0, -150.0, -120.0], dtype=float)
    pos_max = np.array([150.0, 180.0, 150.0, 120.0], dtype=float)

    # Velocity limits (deg/s), fairly generous for fast but plausible motion.
    vel_max_abs = np.array([450.0, 360.0, 360.0, 360.0], dtype=float)

    # Acceleration limits (deg/s^2), again conservative but not too tight.
    acc_max_abs = np.array([4000.0, 3500.0, 3500.0, 3500.0], dtype=float)

    return JointLimitsDeg(
        pos_min=pos_min,
        pos_max=pos_max,
        vel_max_abs=vel_max_abs,
        acc_max_abs=acc_max_abs,
    )


def smooth_angles_deg(
    q_deg: np.ndarray,
    method: str = "moving_average",
    window_length: int = 11,
    polyorder: int = 3,
    *,
    ma_window: int = 5,
    dt: float = 1.0,
    kalman_process_var: float = 5.0,
    kalman_meas_var: float = 25.0,
    ekf_wrap_degrees: bool = False,
) -> np.ndarray:
    """Smooth demonstrated joint angles (deg) along time.

    Args:
        q_deg: (T, n_joints) angles in degrees.
        method: Smoothing method. One of:
            - "moving_average" / "ma" (default)
            - "savitzky_golay" / "savgol"
            - "kalman"
            - "extended_kalman" / "ekf"
        window_length: Window length in samples (used by Savitzky–Golay; adapted if too long).
        polyorder: Polynomial order for Savitzky–Golay.
        ma_window: Window length for moving average.
        dt: Sample period in seconds (used by Kalman / EKF). If unknown, leaving default is OK.
        kalman_process_var: Process noise intensity (larger = smoother but can lag less/track more).
        kalman_meas_var: Measurement noise variance (larger = smoother).
        ekf_wrap_degrees: If True, EKF innovation uses wrapped angle difference in [-180, 180].
    """
    q = np.asarray(q_deg, dtype=float)
    if q.ndim != 2:
        raise ValueError(f"Expected (T, n_joints) array, got shape {q.shape}")

    T, n_joints = q.shape
    if T < 3:
        return q.copy()

    def _odd_at_least_3(w: int, max_len: int) -> int:
        w = int(w)
        if w < 3:
            w = 3
        if w > max_len:
            w = max_len
        if w % 2 == 0:
            w -= 1
        if w < 3:
            w = 3 if max_len >= 3 else (max_len | 1)
        return w

    def _moving_average_1d(x: np.ndarray, w: int) -> np.ndarray:
        w = _odd_at_least_3(w, x.shape[0])
        if w <= 1 or x.shape[0] <= 1:
            return x.copy()
        pad = w // 2
        xp = np.pad(x, (pad, pad), mode="reflect")
        kernel = np.full(w, 1.0 / w, dtype=float)
        return np.convolve(xp, kernel, mode="valid")

    def _wrap_deg_180(a: np.ndarray) -> np.ndarray:
        return (a + 180.0) % 360.0 - 180.0

    def _kalman_posvel_1d(
        z: np.ndarray,
        dt_: float,
        q_var: float,
        r_var: float,
        wrap_innovation: bool,
    ) -> np.ndarray:
        if dt_ <= 0:
            raise ValueError(f"dt must be positive for Kalman filter, got {dt_}")
        z = np.asarray(z, dtype=float)
        # State: [pos, vel]
        F = np.array([[1.0, dt_], [0.0, 1.0]], dtype=float)
        H = np.array([[1.0, 0.0]], dtype=float)
        # Continuous white-acceleration model discretization:
        Q = q_var * np.array(
            [
                [dt_**4 / 4.0, dt_**3 / 2.0],
                [dt_**3 / 2.0, dt_**2],
            ],
            dtype=float,
        )
        R = np.array([[r_var]], dtype=float)

        x = np.array([z[0], 0.0], dtype=float)
        P = np.diag([1e6, 1e6]).astype(float)

        out = np.empty_like(z)
        for k in range(z.shape[0]):
            # Predict
            x = F @ x
            P = F @ P @ F.T + Q

            # Update
            y = z[k] - (H @ x)[0]
            if wrap_innovation:
                y = float(_wrap_deg_180(np.array([y]))[0])
            S = (H @ P @ H.T + R)[0, 0]
            K = (P @ H.T)[:, 0] / S
            x = x + K * y
            P = (np.eye(2) - np.outer(K, H[0])) @ P
            out[k] = x[0]
        return out

    m = method.strip().lower().replace("-", "_").replace(" ", "_")
    if m in {"moving_average", "ma"}:
        q_smooth = np.empty_like(q)
        for j in range(n_joints):
            q_smooth[:, j] = _moving_average_1d(q[:, j], ma_window)
        return q_smooth

    if m in {"savitzky_golay", "savgol", "savitzkygolay", "savitzky"}:
        wl = _odd_at_least_3(window_length, T if T % 2 == 1 else T - 1)
        po = int(polyorder)
        if wl <= po:
            po = max(1, wl - 1)
        q_smooth = np.empty_like(q)
        for j in range(n_joints):
            q_smooth[:, j] = savgol_filter(q[:, j], wl, po, mode="interp")
        return q_smooth

    if m in {"kalman"}:
        q_smooth = np.empty_like(q)
        for j in range(n_joints):
            q_smooth[:, j] = _kalman_posvel_1d(
                q[:, j],
                dt_=dt,
                q_var=float(kalman_process_var),
                r_var=float(kalman_meas_var),
                wrap_innovation=False,
            )
        return q_smooth

    if m in {"extended_kalman", "ekf"}:
        # For this 1D measurement model (angle position), EKF reduces to KF,
        # but we keep a separate option to allow wrapped innovations for angles.
        q_smooth = np.empty_like(q)
        for j in range(n_joints):
            q_smooth[:, j] = _kalman_posvel_1d(
                q[:, j],
                dt_=dt,
                q_var=float(kalman_process_var),
                r_var=float(kalman_meas_var),
                wrap_innovation=bool(ekf_wrap_degrees),
            )
        return q_smooth

    raise ValueError(
        f"Unknown smoothing method '{method}'. "
        "Use 'moving_average'/'ma', 'savitzky_golay'/'savgol', 'kalman', or 'extended_kalman'/'ekf'."
    )


def finite_differences(
    q: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute velocity and acceleration by finite differences.

    Args:
        q: (T, n_joints) positions.
        dt: Timestep in seconds.

    Returns:
        dq: (T, n_joints) velocities.
        ddq: (T, n_joints) accelerations.
    """
    q = np.asarray(q, dtype=float)
    if q.ndim != 2:
        raise ValueError(f"Expected (T, n_joints) array, got shape {q.shape}")
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")

    dq = np.gradient(q, dt, axis=0)
    ddq = np.gradient(dq, dt, axis=0)
    return dq, ddq


@dataclass
class TrajectoryValidationReport:
    ok: bool
    reason: str
    position_violations: int
    velocity_violations: int
    acceleration_violations: int
    has_nans: bool


def validate_joint_trajectory_deg(
    q_deg: np.ndarray,
    dt: float,
    limits: Optional[JointLimitsDeg] = None,
    raise_on_error: bool = False,
    name: str = "trajectory",
) -> TrajectoryValidationReport:
    """Validate a joint‑space trajectory in degrees against simple human limits.

    Checks:
        - finite values
        - position inside per‑joint [min, max]
        - |velocity| <= vel_max_abs
        - |acceleration| <= acc_max_abs
    """
    q = np.asarray(q_deg, dtype=float)
    if q.ndim != 2:
        raise ValueError(f"Expected (T, n_joints) array, got shape {q.shape}")
    T, n_joints = q.shape

    if limits is None:
        limits = default_human_arm_limits_4dof()

    if n_joints != limits.n_joints:
        raise ValueError(
            f"{name}: expected {limits.n_joints} joints for validation, got {n_joints}"
        )

    has_nans = not np.all(np.isfinite(q))

    pos_min = limits.pos_min[None, :]
    pos_max = limits.pos_max[None, :]

    below = q < pos_min
    above = q > pos_max
    position_violations = int(np.count_nonzero(below | above))

    dq, ddq = finite_differences(q, dt)
    vel_max = limits.vel_max_abs[None, :]
    acc_max = limits.acc_max_abs[None, :]

    vel_viol = np.abs(dq) > vel_max
    acc_viol = np.abs(ddq) > acc_max

    velocity_violations = int(np.count_nonzero(vel_viol))
    acceleration_violations = int(np.count_nonzero(acc_viol))

    ok = (
        (not has_nans)
        and position_violations == 0
        and velocity_violations == 0
        and acceleration_violations == 0
    )

    if ok:
        reason = f"{name}: OK (within joint limits)"
    else:
        parts = []
        if has_nans:
            parts.append("NaNs/Infs present")
        if position_violations:
            parts.append(f"{position_violations} position violations")
        if velocity_violations:
            parts.append(f"{velocity_violations} velocity violations")
        if acceleration_violations:
            parts.append(f"{acceleration_violations} acceleration violations")
        reason = f"{name}: " + ", ".join(parts)

    report = TrajectoryValidationReport(
        ok=ok,
        reason=reason,
        position_violations=position_violations,
        velocity_violations=velocity_violations,
        acceleration_violations=acceleration_violations,
        has_nans=has_nans,
    )

    if raise_on_error and not ok:
        raise ValueError(reason)

    return report

