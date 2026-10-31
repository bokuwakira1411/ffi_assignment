#!/usr/bin/env bash
# One round of the round-2 study: collect -> build -> train (A1) -> eval -> report.
# Usage: bash scripts/r2_run_round.sh <round_config> <eval_config> <report_config> <timing_file>
# Writes start/end unix time of every step to <timing_file>; stops at the first failing step.
set -euo pipefail
source /opt/miniconda3/etc/profile.d/conda.sh && conda activate lerobot
cd ~/ffi26/improve
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
CFG="$1"; EVAL="$2"; REP="$3"; TIM="$4"
step() { local name="$1"; shift; echo "$name start $(date +%s)" >> "$TIM"; echo "=== [$name] $(date) $*"; "$@"; echo "$name end $(date +%s)" >> "$TIM"; }
step collect python scripts/r2_collect.py --config "$CFG"
step build   python scripts/r2_build.py --config "$CFG"
step train   python scripts/train_r2.py --config "$CFG" --mode a1
step eval    python scripts/phase3_eval.py --config "$EVAL"
step report  python scripts/r2_report.py --config "$REP"
echo "=== [done] $(date)"
