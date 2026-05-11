from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _np_to_jsonable(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def export_generated_trajectory(npz_path: Path, *, out_path: Path) -> Path:
    data = np.load(npz_path, allow_pickle=False)
    keys = set(data.keys())

    # Rollouts in this repo have used both names across time.
    traj_key = "q_gen_deg" if "q_gen_deg" in keys else ("q_gen_rad" if "q_gen_rad" in keys else None)
    if traj_key is None:
        raise KeyError(f"No generated trajectory found in {npz_path.name}. Keys={sorted(keys)}")

    payload: dict[str, Any] = {
        "source_npz": str(npz_path),
        "trajectory_key": traj_key,
        "q_gen": _np_to_jsonable(np.asarray(data[traj_key], dtype=float)),
    }

    if "t" in keys:
        payload["t"] = _np_to_jsonable(np.asarray(data["t"], dtype=float))
    if "dt" in keys:
        payload["dt"] = float(np.atleast_1d(data["dt"])[0])
    if "q0" in keys:
        payload["q0"] = _np_to_jsonable(np.asarray(data["q0"], dtype=float))
    if "qT" in keys:
        payload["qT"] = _np_to_jsonable(np.asarray(data["qT"], dtype=float))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def main() -> None:
    default_npz = Path("coupling/dmp_model_personalized.npz")
    #default_npz = Path("data/processed/subject_06/move_cup/trial_006/dmp_rollout_base.npz")
    default_out = Path("coupling/curvature_weights_personalized.json")

    parser = argparse.ArgumentParser(
        description="Export the generated DMP trajectory from a rollout .npz to a JSON file."
    )
    parser.add_argument(
        "--npz",
        type=Path,
        default=default_npz,
        help="Path to a DMP rollout .npz (default: subject_10 trial_006).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=default_out,
        help="Output JSON path (default: alongside the npz).",
    )
    args = parser.parse_args()

    npz_path = args.npz.expanduser().resolve()
    out_path = args.out.expanduser().resolve()
    data = np.load(npz_path, allow_pickle=False)
    #payload = data["curvature_weights"]
    payload: dict[str, Any] = {
        "source_npz": str(npz_path),
        "key": "curvature_weights",
        "curvature_weights": _np_to_jsonable(np.asarray(data["curvature_weights"], dtype=float)),
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    #export_generated_trajectory(npz_path, out_path=out_path)
    #print(f"Wrote: {out_path}")

    # print numpy array so that i can copy and paste it into the experiment.py file, with 3 decimal places
    #print(np.round(np.load(npz_path)["q_gen_deg"], 3).tolist())


if __name__ == "__main__":
    main()
