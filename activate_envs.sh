#!/usr/bin/env bash

# Usage (must be sourced):
#   source ./activate_envs.sh

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "ERROR: This script must be sourced, not executed."
  echo "Run: source ./activate_envs.sh"
  return 1 2>/dev/null || exit 1
fi

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# 1) Activate the local venv
if [[ -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.venv/bin/activate"
else
  echo "ERROR: Missing ${REPO_ROOT}/.venv/bin/activate"
  return 1
fi

# 2) Initialize conda (so `conda activate` works in non-interactive shells)
if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base 2>/dev/null || true)"
  if [[ -n "${CONDA_BASE}" && -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
  fi
fi

# 3) Activate conda env
if command -v conda >/dev/null 2>&1; then
  conda activate sim310
else
  echo "ERROR: conda not found on PATH (can't run: conda activate sim310)"
  return 1
fi
