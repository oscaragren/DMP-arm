"""
Inspect .npy / .npz arrays in a trial directory.

Given a trial directory like:
  test_data/processed/subject_01/reach/trial_001

This script lists all *.npy and *.npz files directly in that directory
and prints, for each:
  - filename
  - for .npy: array shape, dtype, min/max, and a small value preview
  - for .npz: available keys, and per-key shape/dtype
"""

import argparse
from pathlib import Path

import numpy as np


def inspect_npy(path: Path) -> None:
    arr = np.load(path)
    print(f"\nFile: {path.name}")
    print("  type: .npy")
    print(f"  shape: {arr.shape}")
    print(f"  dtype: {arr.dtype}")
    if arr.size == 0:
        print("  (empty array)")
        return
    try:
        arr_f = arr.astype(float)
        print(f"  min / max: {np.nanmin(arr_f):.6g} / {np.nanmax(arr_f):.6g}")
    except Exception:
        pass
    # Show a small preview
    flat = arr.ravel()
    preview = flat[: min(10, flat.size)]
    print(f"  preview (first {preview.size} values, flattened): {preview}")


def inspect_npz(path: Path) -> None:
    data = np.load(path)
    keys = list(data.keys())
    print(f"\nFile: {path.name}")
    print("  type: .npz")
    print(f"  keys: {keys}")
    for key in keys:
        arr = data[key]
        print(f"    [{key}] shape: {arr.shape}, dtype: {arr.dtype}")


def inspect_trial_dir(trial_dir: Path) -> None:
    if not trial_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {trial_dir}")
    files = sorted(trial_dir.iterdir())
    npy_files = [f for f in files if f.suffix == ".npy"]
    npz_files = [f for f in files if f.suffix == ".npz"]

    if not npy_files and not npz_files:
        print(f"No .npy or .npz files found in {trial_dir}")
        return

    print(f"Inspecting trial directory: {trial_dir}")
    if npy_files:
        print(f"Found {len(npy_files)} .npy file(s): {[f.name for f in npy_files]}")
    if npz_files:
        print(f"Found {len(npz_files)} .npz file(s): {[f.name for f in npz_files]}")

    for f in npy_files:
        inspect_npy(f)
    for f in npz_files:
        inspect_npz(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Print shapes and basic statistics for .npy/.npz files in a trial "
            "directory (e.g. test_data/processed/subject_01/reach/trial_001)."
        )
    )
    parser.add_argument(
        "--path",
        type=Path,
        required=True,
        help="Path to trial directory containing .npy/.npz files.",
    )
    args = parser.parse_args()
    inspect_trial_dir(args.path.resolve())


if __name__ == "__main__":
    main()

