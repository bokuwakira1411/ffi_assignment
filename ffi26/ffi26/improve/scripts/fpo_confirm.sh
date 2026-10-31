#!/usr/bin/env bash
# Confirm FPO iteration 3: other-8-task forgetting check (seed 1000) + fresh eval seed 2000 vs official, then reports.
set -euo pipefail
source /opt/miniconda3/etc/profile.d/conda.sh && conda activate lerobot
cd ~/ffi26/improve
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
C=~/ffi26/results/round2/configs; TIM=outputs/fpo/timing_confirm.txt
step() { local name="$1"; shift; echo "$name start $(date +%s)" >> "$TIM"; echo "=== [$name] $(date) $*"; "$@"; echo "$name end $(date +%s)" >> "$TIM"; }
step other   python scripts/phase3_eval.py --config $C/eval_fpo_it3_other.yaml
step s2000   python scripts/phase3_eval.py --config $C/eval_seed2000.yaml
step report  python scripts/r2_report.py --config $C/report.yaml
step report2 python scripts/r2_report.py --config $C/report_seed2000.yaml
echo "=== [done] $(date)"
