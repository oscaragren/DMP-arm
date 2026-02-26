import numpy as np

def elbow_flexion_deg(seq: np.ndarray) -> np.ndarray:
    """
    seq: (T, 3, 3) with [shoulder, elbow, wrist] in meters
    returns: (T,) elbow angle in degrees, NaN where missing
    """
    S = seq[:, 0, :]
    E = seq[:, 1, :]
    W = seq[:, 2, :]

    u = S - E # elbow -> shoulder vector
    v = W - E # elbow -> wrist vector

    # Normalize with NaN safety
    u_norm = np.linalg.norm(u, axis=1)
    v_norm = np.linalg.norm(v, axis=1)
    valid = (u_norm > 0) & (v_norm > 0) & np.all(np.isfinite(u), axis=1) & np.all(np.isfinite(v), axis=1)

    out = np.full((seq.shape[0],), np.nan, dtype=np.float64)
    dot = np.einsum("ij,ij->i", u, v)
    cosang = np.clip(dot / (u_norm * v_norm), -1.0, 1.0)
    out[valid] = np.degrees(np.arccos(cosang[valid]))
    return out



    