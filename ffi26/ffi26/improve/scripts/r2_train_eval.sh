#!/usr/bin/env bash
# Train one variant on an existing round dataset, then eval + report.
# Usage: bash scripts/r2_train_eval.sh <round_config> <mode a1|a1emb|awr> <train_out> <eval_config> <report_config> <timing_file>
set -euo pipefail
source /opt/miniconda3/etc/profile.d/conda.sh && conda activate lerobot
cd ~/ffi26/improve
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
CFG="$1"; MODE="$2"; OUT="$3"; EVAL="$4"; REP="$5"; TIM="$6"
step() { local name="$1"; shift; echo "$name start $(date +%s)" >> "$TIM"; echo "=== [$name] $(date) $*"; "$@"; echo "$name end $(date +%s)" >> "$TIM"; }
step train  python scripts/train_r2.py --config "$CFG" --mode "$MODE" --out "$OUT"
step eval   python scripts/phase3_eval.py --config "$EVAL"
step report python scripts/r2_report.py --config "$REP"
echo "=== [done] $(date)"
