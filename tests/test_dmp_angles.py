import os
import numpy as np
from pathlib import Path

from dmp.dmp import DMPModel, fit, rollout_simple

# Default trial when no env/config is set
_DEFAULT_DATA_DIR = "test_data/processed"
_DEFAULT_SUBJECT = 1
_DEFAULT_MOTION = "reach"
_DEFAULT_TRIAL = 1


def _get_trial_dir(project_root: Path | None = None) -> Path:
    """Resolve trial directory from env DMP_TEST_* or defaults."""
    if project_root is None:
        project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / os.environ.get("DMP_TEST_DATA_DIR", _DEFAULT_DATA_DIR)
    path_env = os.environ.get("DMP_TEST_PATH")
    if path_env:
        p = Path(path_env)
        return p if p.is_absolute() else project_root / p
    subject = int(os.environ.get("DMP_TEST_SUBJECT", _DEFAULT_SUBJECT))
    motion = os.environ.get("DMP_TEST_MOTION", _DEFAULT_MOTION)
    trial = int(os.environ.get("DMP_TEST_TRIAL", _DEFAULT_TRIAL))
    return data_dir / f"subject_{subject:02d}" / motion / f"trial_{trial:03d}"


def _load_angles_demo(trial_dir: Path) -> np.ndarray:
    """
    Load elbow + shoulder angles from angles.npz and return a clean demo
    trajectory of shape (T, n_joints).
    """
    npz_path = trial_dir / "angles.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Expected angles.npz at {npz_path}. "
            "Run mapping/sequence_to_angles.py or vis/plot_left_arm_angles.py first."
        )

    data = np.load(npz_path)
    elbow_deg = data["elbow_deg"]  # (T,)
    shoulder_deg = data["shoulder_deg"]  # (T, 3)

    if shoulder_deg.ndim == 1:
        shoulder_deg = shoulder_deg[:, None]

    q_demo = np.column_stack([elbow_deg, shoulder_deg])  # (T, 4)

    # Drop any time steps with invalid values (NaNs/Infs) to keep DMP stable.
    valid = np.all(np.isfinite(q_demo), axis=1)
    q_demo = q_demo[valid]

    if q_demo.shape[0] < 10:
        raise ValueError(
            f"Not enough valid samples in angles demo after cleaning: {q_demo.shape}"
        )

    return q_demo


def test_dmp_fit_from_angles_npz():
    """
    Integration-style test: learn a DMP from recorded angles.npz
    (elbow + 3 shoulder DOFs) and verify rollout is reasonable.
    Trial is chosen via env DMP_TEST_PATH or DMP_TEST_SUBJECT/MOTION/TRIAL/DATA_DIR.
    """
    project_root = Path(__file__).resolve().parents[1]
    trial_dir = _get_trial_dir(project_root)

    assert trial_dir.exists(), (
        f"Trial directory not found: {trial_dir}. "
        "Make sure you have captured and processed a test trial first."
    )

    q_demo = _load_angles_demo(trial_dir)
    print(f"q_demo: {q_demo.shape}")
    T, n_joints = q_demo.shape
    tau = 1.0
    dt = tau / (T - 1)  # Ensure T == int(tau/dt) + 1

    model: DMPModel = fit(
        [q_demo],
        tau=tau,
        dt=dt,
        n_basis_functions=15,
        alpha_canonical=4.0,
        alpha_transformation=25.0,
        beta_transformation=6.25,
    )

    assert model.n_joints == n_joints

    q_gen = rollout_simple(model, q_demo[0], q_demo[-1], tau=tau, dt=dt)
    print(f"q_gen: {q_gen.shape}")
    # Basic sanity checks on rollout.
    assert q_gen.shape == q_demo.shape
    assert np.all(np.isfinite(q_gen)), "Generated DMP rollout contains NaN/Inf"

    # DMP should roughly reproduce the demo trajectory on real data.
    rmse = np.sqrt(np.mean((q_gen - q_demo) ** 2, axis=0))

    # Allow a fairly generous error to account for noise in captured data,
    # while still catching clearly broken behaviour.
    assert np.all(rmse < 30.0), f"RMSE too high for angles demo: {rmse}"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run DMP test on a chosen trial (angles.npz).")
    parser.add_argument("--path", type=Path, default=None, help="Trial directory (overrides subject/motion/trial)")
    parser.add_argument("--data-dir", type=Path, default=None, help="Data root (default: test_data/processed)")
    parser.add_argument("--subject", type=int, default=None, help="Subject number")
    parser.add_argument("--motion", type=str, default=None, help="Motion name")
    parser.add_argument("--trial", type=int, default=None, help="Trial number")
    args = parser.parse_args()
    if args.path is not None:
        os.environ["DMP_TEST_PATH"] = str(args.path.resolve())
    if args.data_dir is not None:
        os.environ["DMP_TEST_DATA_DIR"] = str(args.data_dir)
    if args.subject is not None:
        os.environ["DMP_TEST_SUBJECT"] = str(args.subject)
    if args.motion is not None:
        os.environ["DMP_TEST_MOTION"] = args.motion
    if args.trial is not None:
        os.environ["DMP_TEST_TRIAL"] = str(args.trial)
    test_dmp_fit_from_angles_npz()

