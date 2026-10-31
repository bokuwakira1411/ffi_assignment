#!/usr/bin/env bash
# AWR full run on round-1 data: 6000 steps (checkpoints 2k/4k/6k) -> eval all 3 -> report.
set -euo pipefail
source /opt/miniconda3/etc/profile.d/conda.sh && conda activate lerobot
cd ~/ffi26/improve
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
C=~/ffi26/results/round2/configs; L=outputs/r2; TIM=$L/timing_awr6k_round1.txt
step() { local name="$1"; shift; echo "$name start $(date +%s)" >> "$TIM"; echo "=== [$name] $(date) $*"; "$@"; echo "$name end $(date +%s)" >> "$TIM"; }
step train  python scripts/train_r2.py --config $C/round1.yaml --mode awr --out $L/train_awr6k_round1 \
            --loss_csv ~/ffi26/results/round2/round1/loss_awr6k.csv -- \
            --steps=6000 --policy.scheduler_decay_steps=6000 --save_freq=2000
step eval   python scripts/phase3_eval.py --config $C/eval_round1_awr6k.yaml
step report python scripts/r2_report.py --config $C/report.yaml
echo "=== [done] $(date)"
