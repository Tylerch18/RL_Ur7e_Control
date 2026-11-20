#!/usr/bin/env bash
# Usage: source scripts/activate_env.sh
# Works when sourced from bash or zsh.

# Resolve this script's directory (bash: BASH_SOURCE, zsh: %x / $0).
if [[ -n "${BASH_SOURCE[0]:-}" ]]; then
  _UR7E_SCRIPT="${BASH_SOURCE[0]}"
elif [[ -n "${ZSH_VERSION:-}" ]]; then
  _UR7E_SCRIPT="${(%):-%x}"
else
  _UR7E_SCRIPT="$0"
fi

ROOT_DIR="$(cd "$(dirname "${_UR7E_SCRIPT}")/.." && pwd)"
unset _UR7E_SCRIPT
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
echo "  python: $(command -v python)"
