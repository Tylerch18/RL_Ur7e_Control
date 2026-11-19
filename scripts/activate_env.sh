#!/usr/bin/env bash
# Usage: source scripts/activate_env.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${ROOT_DIR}/.venv"

if [[ ! -f "${ENV_DIR}/bin/activate" ]]; then
  echo "ERROR: missing ${ENV_DIR}."
  echo "Create it with:  bash scripts/setup_env.sh"
  return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1091
source "${ENV_DIR}/bin/activate"
export RL_UR7E_ROOT="${ROOT_DIR}"
echo "Activated .venv (MuJoCo UR7e)"
echo "  RL_UR7E_ROOT=${RL_UR7E_ROOT}"
