from re import L
import numpy as np
from scipy.spatial.transform import Rotation as R

# Trunk "up" in camera frame (OAK-D: X right, Y down, Z forward → person up = -Y)
WORLD_UP = np.array([0.0, -1.0, 0.0], dtype=np.float64)

# 0: left_shoulder, 1: left_elbow, 2: left_wrist, 3: right_shoulder, 4: left_hip, 5: right_hip

def limb_vectors(seq: np.ndarray) -> np.ndarray:
    """
    seq: (T, 4, 3) with [shoulder, elbow, wrist, right_shoulder] in meters
    returns: 
        u: (T, 3) with shoulder -> elbow vector
        v: (T, 3) with elbow -> wrist vector
    """
    S = seq[:, 0, :]
    E = seq[:, 1, :]
    W = seq[:, 2, :]
    return E - S, W - E

def _elbow_flexion(seq: np.ndarray) -> np.ndarray:
    """
    Calculate elbow flexion in degrees.
    0 degrees is when the elbow is fully extended.
    90 degrees is when the elbow is fully bent.

    Note, independent of the trunk frame.
    """
    if seq.ndim != 3 or seq.shape[2] != 3 or seq.shape[1] < 4:
        raise ValueError(f"Expected seq shape (T, N>=4, 3), got {seq.shape}")

    upper_arm, forearm = limb_vectors(seq)
    upper_arm_u, upper_arm_n = _normalize_rows(upper_arm)
    forearm_u, forearm_n = _normalize_rows(forearm)

    theta = np.full((seq.shape[0],), np.nan, dtype=np.float64)
    valid = (upper_arm_n > 1e-8) & (forearm_n > 1e-8)

    cosang = np.einsum("ij,ij->i", upper_arm_u[valid], forearm_u[valid]) # Cosine of angle between upper arm and forearm
    cosang = np.clip(cosang, -1.0, 1.0) # Make sure the cosine is between -1 and 1.

    theta[valid] = np.degrees(np.arccos(cosang)) # Angle between upper arm and forearm in degrees.
    return theta

def _shoulder_flexion(seq: np.ndarray) -> np.ndarray:
    """
    Calculate shoulder flexion in degrees.
    0 degrees is when the upper arm is parallel to the trunk frame's Y-axis (negative Y direction).
    90 degree is when the upper arm is fully extended, reaching forward in positive Z direction in trunk frame.

    Note, uses trunk frame.
    """
    if seq.ndim != 3 or seq.shape[2] != 3 or seq.shape[1] < 4:
        raise ValueError(f"Expected seq shape (T, N>=4, 3), got {seq.shape}")

    upper_arm, _ = limb_vectors(seq)

    # Define rotation matrix from world frame to trunk frame
    R_trunk = _get_trunk_rotation_matrix(seq)
    upper_arm_trunk = np.einsum("tij,tj->ti", R_trunk, upper_arm) # Equivalent to R_trunk.T @ upper_arm (NOTE: maybe other way around)
    # Change from tij -> tji to get R^T instead of R
    
    # Get angle in yz-plane (trunk frame)
    upper_arm_trunk_z = upper_arm_trunk[:, 2]
    upper_arm_trunk_y = upper_arm_trunk[:, 1]
    theta = np.degrees(np.arctan2(upper_arm_trunk_z, -upper_arm_trunk_y)) # Since WORLD_UP is -Y, we need to invert the y-axis.
    return theta

def _shoulder_abduction(seq: np.ndarray) -> np.ndarray:
    """
    Calculate shoulder abduction in degrees.

    0 degrees is when the upper arm is parallel to the trunk frame's Y-axis (negative Y direction).
    90 degrees is when the upper arm is fully abducted, reaching to the side parallel to the trunk frame's X-axis (positive X direction).
    """
    if seq.ndim != 3 or seq.shape[2] != 3 or seq.shape[1] < 4:
        raise ValueError(f"Expected seq shape (T, N>=4, 3), got {seq.shape}")

    upper_arm, _ = limb_vectors(seq)

    # Define rotation matrix from world frame to trunk frame
    R_trunk = _get_trunk_rotation_matrix(seq)
    upper_arm_trunk = np.einsum("tij,tj->ti", R_trunk, upper_arm) # Equivalent to R_trunk.T @ upper_arm
    
    # Get angle in yz-plane (trunk frame)
    upper_arm_trunk_x = upper_arm_trunk[:, 0]
    upper_arm_trunk_y = upper_arm_trunk[:, 1]
    theta = np.degrees(np.arctan2(upper_arm_trunk_x, -upper_arm_trunk_y)) # Since WORLD_UP is -Y, we need to invert the y-axis.
    return theta

def _shoulder_lateral_medial_rotation(seq: np.ndarray) -> np.ndarray:
    """
    Calculate shoulder lateral medial rotation in degrees.
    This is a proxy for the actual rotation of the shoulder around the humerus.

    0 degrees is when the (elbow-bent 90 degrees) forearm is parallel to the trunk frame's Z-axis (positive Z direction).
    -90 degrees is when the (elbow-bent 90 degrees) forearm is parallel to the trunk frame's X-axis (positive X direction). Outer/lateral rotation.
    90 degrees is when the (elbow-bent 90 degrees) forearm is parallel to the trunk frame's X-axis (negative X direction). Inner/medial rotation.

    """
    if seq.ndim != 3 or seq.ndim != 3 or seq.shape[1] < 4:
        raise ValueError(f"Expected seq shape (T, N>=4, 3), got {seq.shape}")

    T = seq.shape[0]
    
    # Get upper- and forearm
    upper_arm, forearm = limb_vectors(seq)

    # Normalize
    upper_arm_u, upper_arm_n = _normalize_rows(upper_arm)
    forearm_u, forearm_n = _normalize_rows(forearm)

    # Trunk rotation matrix: columns are trunk axes in camera frame
    R_trunk = _get_trunk_rotation_matrix(seq)

    # Transform segment directions into trunk frame
    upper_arm_trunk = np.einsum("tij,tj->ti", R_trunk, upper_arm_u)
    forearm_trunk = np.einsum("tij,tj->ti", R_trunk, forearm_u)

    # Shoulder flexion/abduction in radians from upper arm in trunk frame
    ux = upper_arm_trunk[:, 0]
    uy = upper_arm_trunk[:, 1]
    uz = upper_arm_trunk[:, 2]
    
    flex_rad = np.arctan2(uz, -uy)
    abd_rad = np.arctan2(ux, -uy)

    # Build inverse rotations frame-by-frame
    c_f = np.cos(-flex_rad)
    s_f = np.sin(-flex_rad)
    Rx = np.zeros((T, 3, 3), dtype=np.float64)
    Rx[:, 0, 0] = 1.0
    Rx[:, 1, 1] = c_f
    Rx[:, 1, 2] = -s_f
    Rx[:, 2, 1] = s_f
    Rx[:, 2, 2] = c_f

    c_a = np.cos(-abd_rad)
    s_a = np.sin(-abd_rad)
    Rz = np.zeros((T, 3, 3), dtype=np.float64)
    Rz[:, 0, 0] = c_a
    Rz[:, 0, 1] = -s_a
    Rz[:, 1, 0] = s_a
    Rz[:, 1, 1] = c_a
    Rz[:, 2, 2] = 1.0

    # Applu undo rotation to forearm in trunk frame
    R_undo = np.einsum("tij,tjk->tik", Rx, Rz)
    forearm_aligned = np.einsum("tij,tj->ti", R_undo, forearm_trunk)

    # Project forarm onto plane perpendicular to canonical upper-arm axis [0, -1, 0]. This is the xz-plane.
    p = np.stack(
        [
            forearm_aligned[:, 0],
            np.zeros(T, dtype=np.float64),
            forearm_aligned[:, 2]
        ],
        axis=1
    )
    p_u, p_n = _normalize_rows(p)

    # Signed proxy angle around upper-arm axis
    theta = np.full((T,), np.nan, dtype=np.float64)
    valid = (
        (upper_arm_n > 1e-8)
        & (forearm_n > 1e-8)
        & (p_n > 1e-8)
        & np.all(np.isfinite(upper_arm_trunk), axis=1)
        & np.all(np.isfinite(forearm_trunk), axis=1)
        & np.all(np.isfinite(p_u), axis=1)
    )

    theta[valid] = np.degrees(np.arctan2(-p_u[valid, 0], p_u[valid, 2]))
    return theta

def _get_trunk_rotation_matrix(seq: np.ndarray) -> np.ndarray:
    """
    Get the trunk rotation matrix.
    """
    if seq.ndim != 3 or seq.shape[2] != 3 or seq.shape[1] < 6:
        raise ValueError(f"Expected seq shape (T, N>=6, 3), got {seq.shape}")
    LS = np.asarray(seq[:, 0, :], dtype=np.float64) # Left Shoulder
    RS = np.asarray(seq[:, 3, :], dtype=np.float64) # Right Shoulder
    LH = np.asarray(seq[:, 4, :], dtype=np.float64) # Left Hip
    RH = np.asarray(seq[:, 5, :], dtype=np.float64) # Right Hip

    # Define centers
    C_s = 0.5 * (LS + RS) # Shoulder center
    C_h = 0.5 * (LH + RH) # Hip center
    # Hip center: assume same y as right hip and same x as shoulder center.
    # (z taken from right hip to preserve depth without needing left hip)
    #C_h = np.stack([C_s[:, 0], RH[:, 1], RH[:, 2]], axis=1)
    # Define trunk vectors and normalize
    x_trunk_norm, _ = _normalize_rows(LS - RS)
    y_temp, _ = _normalize_rows(C_s - C_h)
    z_trunk_norm, _ = _normalize_rows(np.cross(x_trunk_norm, y_temp))
    x_trunk_norm, _ = _normalize_rows(np.cross(y_temp, z_trunk_norm))
    y_trunk_norm, _ = _normalize_rows(np.cross(z_trunk_norm, x_trunk_norm))

    return np.stack([x_trunk_norm, y_trunk_norm, z_trunk_norm], axis=2)

def _normalize_rows(v: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    """
    Row-wise normalization with NaN safety.

    Args:
        v: (T, 3)
        eps: threshold below which a row is considered degenerate

    Returns:
        v_u: (T, 3) unit vectors (NaN where degenerate)
        n: (T,) norms
    """
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=1)
    v_u = np.where((n > eps)[:, None], v / np.maximum(n[:, None], eps), np.nan)
    return v_u, n

def get_angles(seq: np.ndarray) -> np.ndarray:
    """
    Get the angles for the left arm.

    seq: (T, 6, 3) with [left_shoulder, left_elbow, left_wrist, right_shoulder, left_hip, right_hip] in meters
    """
    if seq.ndim != 3 or seq.shape[2] != 3 or seq.shape[1] < 4:
        raise ValueError(f"Expected seq shape (T, N>=4, 3), got {seq.shape}")
    s_flex = _shoulder_flexion(seq)
    s_abd = _shoulder_abduction(seq)
    s_lat_med_rot = _shoulder_lateral_medial_rotation(seq)
    e_flex = _elbow_flexion(seq)
    # Canonical repo order: [elbow_flexion, shoulder_flexion, shoulder_abduction, shoulder_lat/med_rotation]
    return np.stack([e_flex, s_flex, s_abd, s_lat_med_rot], axis=1)


