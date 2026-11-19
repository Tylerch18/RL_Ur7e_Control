#!/usr/bin/env bash
# Create an Isaac Lab + Isaac Sim environment for UR7e RL simulation.
#
# Target: Linux x86_64 with an NVIDIA GPU (RTX, 16GB+ VRAM recommended).
# Isaac Gym is deprecated — this uses Isaac Lab on Isaac Sim instead.
#
# Usage (on the Linux GPU machine):
#   cd /path/to/RL_Ur7e_Control
#   bash scripts/setup_linux_env.sh
#
# Then activate with:
#   source scripts/activate_env.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${ROOT_DIR}/env_isaaclab"
ISAACLAB_DIR="${ROOT_DIR}/third_party/IsaacLab"
PYTHON_VERSION="3.12"
ISAACSIM_VERSION="6.0.0"
TORCH_VERSION="2.10.0"
TORCHVISION_VERSION="0.25.0"

die() { echo "ERROR: $*" >&2; exit 1; }

echo "==> Checking platform..."
uname_s="$(uname -s)"
uname_m="$(uname -m)"
if [[ "${uname_s}" != "Linux" ]]; then
  die "Isaac Sim / Isaac Lab require Linux (or Windows), not ${uname_s}.
This Mac can edit code, but the env must be created on a Linux + NVIDIA GPU host
(lab workstation, cloud GPU, or NVIDIA Brev Launchable)."
fi
if [[ "${uname_m}" != "x86_64" ]]; then
  echo "WARNING: Official Isaac Sim pip packages target x86_64. Detected ${uname_m}."
  echo "         aarch64 is only supported on NVIDIA DGX Spark."
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  die "nvidia-smi not found. Install NVIDIA drivers before continuing."
fi
echo "    GPU:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

if ! command -v uv >/dev/null 2>&1; then
  echo "==> Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # shellcheck disable=SC1091
  source "${HOME}/.local/bin/env" 2>/dev/null || export PATH="${HOME}/.local/bin:${PATH}"
fi
command -v uv >/dev/null 2>&1 || die "uv not on PATH after install"

echo "==> Creating Python ${PYTHON_VERSION} venv at ${ENV_DIR}"
uv venv --python "${PYTHON_VERSION}" --seed "${ENV_DIR}"
# shellcheck disable=SC1091
source "${ENV_DIR}/bin/activate"

echo "==> Upgrading pip"
uv pip install --upgrade pip

echo "==> Installing CUDA PyTorch ${TORCH_VERSION}"
uv pip install -U \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  --index-url https://download.pytorch.org/whl/cu128

echo "==> Installing Isaac Sim ${ISAACSIM_VERSION} (this is large)"
uv pip install "isaacsim[all,extscache]==${ISAACSIM_VERSION}" \
  --extra-index-url https://pypi.nvidia.com

echo "==> Cloning / updating Isaac Lab"
mkdir -p "${ROOT_DIR}/third_party"
if [[ -d "${ISAACLAB_DIR}/.git" ]]; then
  git -C "${ISAACLAB_DIR}" pull --ff-only
else
  git clone https://github.com/isaac-sim/IsaacLab.git "${ISAACLAB_DIR}"
fi

echo "==> Installing Isaac Lab into the venv"
(
  cd "${ISAACLAB_DIR}"
  ./isaaclab.sh --install
)

echo "==> Installing this project package (editable)"
uv pip install -e "${ROOT_DIR}"

echo
echo "=============================================="
echo " Isaac Lab env ready for UR7e simulation"
echo "=============================================="
echo "Activate:"
echo "  source ${ROOT_DIR}/scripts/activate_env.sh"
echo
echo "Smoke-test Isaac Lab (headless):"
echo "  source scripts/activate_env.sh"
echo "  python third_party/IsaacLab/scripts/environments/list_envs.py"
echo
echo "Spawn UR7e (uses Isaac Sim ur5e USD — same kinematics as UR7e):"
echo "  python scripts/spawn_ur7e.py --headless"
echo "=============================================="
