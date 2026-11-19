#!/usr/bin/env bash
# Create a cross-platform MuJoCo env (macOS / Linux / Windows via Git Bash or WSL).
#
#   bash scripts/setup_env.sh
#   source scripts/activate_env.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${ROOT_DIR}/.venv"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

cd "${ROOT_DIR}"

if ! command -v uv >/dev/null 2>&1; then
  echo "==> Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # shellcheck disable=SC1091
  source "${HOME}/.local/bin/env" 2>/dev/null || export PATH="${HOME}/.local/bin:${PATH}"
fi
command -v uv >/dev/null 2>&1 || {
  echo "ERROR: uv not on PATH" >&2
  exit 1
}

echo "==> Creating Python ${PYTHON_VERSION} venv at ${ENV_DIR}"
uv venv --python "${PYTHON_VERSION}" "${ENV_DIR}"

echo "==> Installing project + RL extras"
# shellcheck disable=SC1091
source "${ENV_DIR}/bin/activate"
uv pip install -e ".[rl,dev]"

echo "==> Prefetching Menagerie UR5e/UR7e meshes (one-time download)"
python - <<'PY'
import mujoco_menagerie as mm
mm.prefetch(["universal_robots_ur5e"])
print("menagerie cache:", mm.get("universal_robots_ur5e").path())
PY

echo "==> Headless smoke test"
python -m ur7e_control.scripts.spawn --headless --seconds 0.5

echo
echo "=============================================="
echo " MuJoCo UR7e env ready"
echo "=============================================="
echo "Activate:"
echo "  source scripts/activate_env.sh"
echo
echo "Interactive viewer (macOS: use mjpython):"
echo "  python -m ur7e_control.scripts.spawn"
echo "  mjpython -m ur7e_control.scripts.spawn   # macOS native viewer"
echo
echo "Train (local or remote SSH, headless):"
echo "  python -m ur7e_control.scripts.train_reach --timesteps 200000"
echo "=============================================="
