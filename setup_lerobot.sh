#!/usr/bin/env bash
# FFI26 - Day 1 setup: LeRobot + SmolVLA + LIBERO on a headless Linux box.
# No Docker, no sudo needed. Safe to re-run (steps skip if already done).
#
# Usage:
#   bash setup_lerobot.sh
# Optional overrides (env vars):
#   ENV_NAME=lerobot  WORKDIR=$HOME/ffi26  LEROBOT_REF=main
#   CUDA_INDEX=https://download.pytorch.org/whl/cu128
#
# Based on the LeRobot docs "Installation" and "LIBERO" pages (checked 29/10/2026):
#   - Python >= 3.12, PyTorch >= 2.10
#   - cu128 wheels cover Blackwell (RTX 50xx), driver floor 570.86
#   - LIBERO needs Linux; headless rendering via MUJOCO_GL=egl

set -euo pipefail

ENV_NAME="${ENV_NAME:-lerobot}"
WORKDIR="${WORKDIR:-$HOME/ffi26}"
LEROBOT_REF="${LEROBOT_REF:-main}"
CUDA_INDEX="${CUDA_INDEX:-https://download.pytorch.org/whl/cu128}"
MIN_DRIVER="570.86"
# Resolve the script's own folder FIRST: later steps `cd` away, which would break a relative $0.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$WORKDIR/results" "$WORKDIR/logs"
LOG="$WORKDIR/logs/setup_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "[setup] log file: $LOG"

# ---------------------------------------------------------------- 0. GPU/driver
echo "[0/7] GPU and driver check"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found. Is the NVIDIA driver visible to your account?"; exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1 | tr -d ' ')"
if [ "$(printf '%s\n%s\n' "$MIN_DRIVER" "$DRIVER" | sort -V | head -n1)" != "$MIN_DRIVER" ]; then
  echo "WARNING: driver $DRIVER < $MIN_DRIVER (needed for cu128 wheels on RTX 50xx)."
  echo "         You have no root, so ask the lab admin to upgrade, or fall back to running on Kaggle."
fi

# ---------------------------------------------------------------- 1. conda (miniforge, user-space)
echo "[1/7] conda (miniforge)"
if ! command -v conda >/dev/null 2>&1; then
  if [ -x "$HOME/miniforge3/bin/conda" ]; then
    export PATH="$HOME/miniforge3/bin:$PATH"
  else
    INSTALLER="$WORKDIR/Miniforge3.sh"
    wget -O "$INSTALLER" "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    bash "$INSTALLER" -b -p "$HOME/miniforge3"
    export PATH="$HOME/miniforge3/bin:$PATH"
  fi
fi
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

# ---------------------------------------------------------------- 2. env + ffmpeg
echo "[2/7] conda env '$ENV_NAME' (Python 3.12) + ffmpeg"
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" python=3.12
fi
set +u; conda activate "$ENV_NAME"; set -u
# ffmpeg: needed by torchcodec (video decoding). cmake: some pip packages build from source
# (e.g. during the LeRobot extras install) and fail without it; conda-forge needs no sudo.
conda install -y ffmpeg cmake -c conda-forge

# ---------------------------------------------------------------- 3. PyTorch (cu128 for Blackwell)
echo "[3/7] PyTorch from $CUDA_INDEX"
pip install --upgrade pip
pip install torch torchvision --index-url "$CUDA_INDEX"
python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| available", torch.cuda.is_available())
major = int(torch.__version__.split(".")[0]); minor = int(torch.__version__.split(".")[1].split("+")[0])
if (major, minor) < (2, 10):
    print("WARNING: LeRobot docs ask for PyTorch >= 2.10; you have", torch.__version__)
PY

# ---------------------------------------------------------------- 4. LeRobot from source
echo "[4/7] LeRobot ($LEROBOT_REF) + extras"
cd "$WORKDIR"
if [ ! -d lerobot ]; then
  git clone https://github.com/huggingface/lerobot.git
fi
cd lerobot
git fetch --all --tags
git checkout "$LEROBOT_REF"
# smolvla = policy deps, libero = simulator, evaluation = lerobot-eval deps, training = dataset + accelerate + wandb
pip install -e ".[smolvla,libero,evaluation,training]"

# ---------------------------------------------------------------- 5. headless rendering
echo "[5/7] MUJOCO_GL=egl (persisted inside the conda env)"
conda env config vars set MUJOCO_GL=egl
export MUJOCO_GL=egl
set +u; conda deactivate; conda activate "$ENV_NAME"; set -u

# ---------------------------------------------------------------- 6. record versions for the report
echo "[6/7] recording versions into $WORKDIR/results"
{
  echo "date: $(date -Is)"
  echo "host: $(hostname)"
  echo "lerobot_ref: $LEROBOT_REF"
  echo "lerobot_commit: $(git -C "$WORKDIR/lerobot" rev-parse HEAD)"
  echo "driver: $DRIVER"
} > "$WORKDIR/results/versions.txt"
pip freeze > "$WORKDIR/results/pip_freeze.txt"
conda env export --no-builds > "$WORKDIR/results/environment.yml"

# ---------------------------------------------------------------- 7. sanity check
echo "[7/7] sanity check"
python "$SCRIPT_DIR/check_env.py" --out "$WORKDIR/results/env_report.json" || {
  echo "check_env.py reported problems. Read the messages above before continuing."; exit 1; }

echo
echo "Done. Next time:  conda activate $ENV_NAME"
echo "Then run:         bash $SCRIPT_DIR/smoke_test.sh"