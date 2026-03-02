import numpy as np

# Trunk "up" in camera frame (OAK-D: X right, Y down, Z forward → person up = -Y)
WORLD_UP = np.array([0.0, -1.0, 0.0], dtype=np.float64)


def limb_vectors(seq: np.ndarray) -> np.ndarray:
    """
    seq: (T, 4, 3) with [shoulder, elbow, wrist, right_shoulder] in meters
    returns: 
        u: (T, 3) with elbow -> shoulder vector
        v: (T, 3) with elbow -> wrist vector
    """
    S = seq[:, 0, :]
    E = seq[:, 1, :]
    W = seq[:, 2, :]
    return S-E, E-W

def elbow_flexion_deg(seq: np.ndarray) -> np.ndarray:
    """
    seq: (T, 4, 3) with [shoulder, elbow, wrist, right_shoulder] in meters
    returns: (T,) elbow angle in degrees, NaN where missing
    """

    u, v = limb_vectors(seq)

    # Normalize with NaN safety
    u_norm = np.linalg.norm(u, axis=1)
    v_norm = np.linalg.norm(v, axis=1)
    valid = (u_norm > 0) & (v_norm > 0) & np.all(np.isfinite(u), axis=1) & np.all(np.isfinite(v), axis=1)

    out = np.full((seq.shape[0],), np.nan, dtype=np.float64)
    dot = np.einsum("ij,ij->i", u, v)
    cosang = np.clip(dot / (u_norm * v_norm), -1.0, 1.0)
    out[valid] = np.degrees(np.arccos(cosang[valid]))
    return out


def _trunk_frame_from_shoulders(
    left_shoulder: np.ndarray, right_shoulder: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build right-handed trunk frame at left shoulder.
    left_shoulder, right_shoulder: (T, 3)
    Returns x_axis (T,3), y_axis (T,3), z_axis (T,3) unit vectors.
    x = right_shoulder - left_shoulder (normalized), y ≈ up, z = forward.
    """
    x = right_shoulder - left_shoulder # vector between shoulders
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    valid = (x_norm.squeeze(1) > 1e-8) & np.all(np.isfinite(x), axis=1)
    x = np.where(x_norm > 1e-8, x / x_norm, np.nan)

    # y = up (world) projected onto plane perpendicular to x
    up = np.broadcast_to(WORLD_UP, (x.shape[0], 3))
    y = up - np.einsum("ij,ij->i", up, x)[:, None] * x # project up vector onto plane perpendicular to x
    y_norm = np.linalg.norm(y, axis=1, keepdims=True)
    y = np.where(y_norm > 1e-8, y / y_norm, np.nan)
    
    # z = forward (world) projected onto plane perpendicular to x and y
    z = np.cross(x, y) # cross product of x and y to get z axis
    z_norm = np.linalg.norm(z, axis=1, keepdims=True)
    z = np.where(z_norm > 1e-8, z / z_norm, np.nan)
    return x, y, z # x, y, z axes of trunk frame


def _signed_angle_around_axis(
    v_ref: np.ndarray, v_arm: np.ndarray, axis: np.ndarray
) -> np.ndarray:
    """
    Signed angle in radians from v_ref to v_arm around axis (all (T,3)).
    v_ref: (T, 3) reference vector
    v_arm: (T, 3) arm vector
    axis: (T, 3) axis vector
    returns: (T,) signed angle in radians
    """
    # Project into plane perpendicular to axis
    axis_norm = np.linalg.norm(axis, axis=1, keepdims=True)
    axis_u = np.where(axis_norm > 1e-8, axis / axis_norm, np.nan)
   
    v_ref_p = v_ref - np.einsum("ij,ij->i", v_ref, axis_u)[:, None] * axis_u # project v_ref onto plane perpendicular to axis
    v_arm_p = v_arm - np.einsum("ij,ij->i", v_arm, axis_u)[:, None] * axis_u # project v_arm onto plane perpendicular to axis
    
    n_ref = np.linalg.norm(v_ref_p, axis=1)
    n_arm = np.linalg.norm(v_arm_p, axis=1)
    valid = (n_ref > 1e-8) & (n_arm > 1e-8)
    
    v_ref_p = np.where(valid[:, None], v_ref_p / np.maximum(n_ref[:, None], 1e-8), np.nan)
    v_arm_p = np.where(valid[:, None], v_arm_p / np.maximum(n_arm[:, None], 1e-8), np.nan)
   
    dot = np.einsum("ij,ij->i", v_ref_p, v_arm_p) # dot product of v_ref_p and v_arm_p to get the cosine of the angle
    cross = np.cross(v_ref_p, v_arm_p) # cross product of v_ref_p and v_arm_p to get the direction of the angle
    
    sign = np.sign(np.einsum("ij,ij->i", cross, axis_u))
    sign = np.where(sign == 0, 1, sign) # if sign is 0, set to 1
    out = np.full((axis.shape[0],), np.nan, dtype=np.float64)
    out[valid] = sign[valid] * np.arccos(np.clip(dot[valid], -1.0, 1.0)) # angle in radians
    return out # signed angle in radians


def shoulder_angles_3dof(seq: np.ndarray) -> np.ndarray:
    """
    Compute 3-DOF shoulder angles (elevation, plane of elevation, internal rotation).

    seq: (T, 4, 3) with [left_shoulder, left_elbow, left_wrist, right_shoulder] in meters (camera frame).
    returns: (T, 3) in degrees — [elevation_deg, azimuth_deg, internal_rotation_deg].
    NaN where keypoints are missing or degenerate.

    Convention:
    - Elevation: angle of upper arm from vertical (trunk up). 0° = arm up, 90° = horizontal, 180° = arm down.
    - Azimuth (plane of elevation): angle in the horizontal plane. 0° = forward (trunk z), positive = toward right.
    - Internal rotation: rotation of the humerus about its long axis (elbow flex axis vs reference plane).
    """
    T = seq.shape[0]
    ls = seq[:, 0, :]   # left shoulder
    rs = seq[:, 3, :]   # right shoulder

    upper_arm, forearm = limb_vectors(seq)

    x_trunk, y_trunk, z_trunk = _trunk_frame_from_shoulders(ls, rs)

    valid = (
        np.all(np.isfinite(seq), axis=(1, 2))
        & (np.linalg.norm(upper_arm, axis=1) > 1e-8)
        & (np.linalg.norm(forearm, axis=1) > 1e-8)
        & np.all(np.isfinite(x_trunk), axis=1)
        & np.all(np.isfinite(y_trunk), axis=1)
        & np.all(np.isfinite(z_trunk), axis=1)
    )

    out = np.full((T, 3), np.nan, dtype=np.float64)

    # Elevation: angle between upper arm and vertical (y_trunk)
    cos_el = np.einsum("ij,ij->i", upper_arm, y_trunk) / np.maximum(np.linalg.norm(upper_arm, axis=1), 1e-8) # cosine of the angle between upper arm and vertical
    elevation_rad = np.arccos(np.clip(cos_el, -1.0, 1.0)) # angle in radians
    out[:, 0] = np.degrees(elevation_rad) # angle in degrees

    # Azimuth: angle of upper-arm projection in horizontal (xz) plane
    ua_xz = upper_arm - np.einsum("ij,ij->i", upper_arm, y_trunk)[:, None] * y_trunk
    proj_x = np.einsum("ij,ij->i", ua_xz, x_trunk) # projection of upper arm onto x axis
    proj_z = np.einsum("ij,ij->i", ua_xz, z_trunk) # projection of upper arm onto z axis
    azimuth_rad = np.arctan2(proj_z, proj_x)
    out[:, 1] = np.degrees(azimuth_rad)

    # Internal rotation: signed angle from reference plane (upper_arm, y_trunk) to arm plane (upper_arm, forearm) about upper_arm
    n_ref = np.cross(upper_arm, y_trunk)
    n_arm = np.cross(upper_arm, forearm)
    internal_rad = _signed_angle_around_axis(n_ref, n_arm, upper_arm)
    out[:, 2] = np.degrees(internal_rad)

    out[~valid, :] = np.nan
    return out
