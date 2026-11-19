#!/usr/bin/env bash
# Activate the Isaac Lab venv created by setup_linux_env.sh.
# Usage: source scripts/activate_env.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${ROOT_DIR}/env_isaaclab"

if [[ ! -f "${ENV_DIR}/bin/activate" ]]; then
  echo "ERROR: missing ${ENV_DIR}."
  echo "Run on a Linux + NVIDIA GPU host:"
  echo "  bash scripts/setup_linux_env.sh"
  return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1091
source "${ENV_DIR}/bin/activate"
export RL_UR7E_ROOT="${ROOT_DIR}"
export ISAACLAB_PATH="${ROOT_DIR}/third_party/IsaacLab"
echo "Activated env_isaaclab (Isaac Lab + Isaac Sim)"
echo "  RL_UR7E_ROOT=${RL_UR7E_ROOT}"
