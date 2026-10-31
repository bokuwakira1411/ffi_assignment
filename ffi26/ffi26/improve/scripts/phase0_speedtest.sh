#!/usr/bin/env bash
# Phase 0: speed test lerobot/smolvla_libero on libero_10 task 0, NAS=10, relative control.
# Usage: bash scripts/phase0_speedtest.sh "4 8"   (batch sizes), N_EP=8
set -uo pipefail
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
source /opt/miniconda3/etc/profile.d/conda.sh && conda activate lerobot
BSS="${1:-4 8}"; N_EP="${N_EP:-8}"
ROOT="$HOME/ffi26/improve/outputs/speedtest"; mkdir -p "$ROOT"
RM='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'
for bs in $BSS; do
  OUT="$ROOT/bs${bs}"; rm -rf "$OUT"
  nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -lms 1000 > "$ROOT/gpu_bs${bs}.csv" &
  SMI=$!
  ( while true; do cat /proc/loadavg; sleep 5; done ) > "$ROOT/load_bs${bs}.txt" &
  LOAD=$!
  start=$(date +%s.%N)
  lerobot-eval --policy.path=lerobot/smolvla_libero --env.type=libero --env.task=libero_10 \
    --env.task_ids="[0]" --eval.batch_size=$bs --eval.n_episodes=$N_EP --env.max_parallel_tasks=1 \
    --seed=1000 --output_dir="$OUT" --rename_map="$RM" --policy.n_action_steps=10 \
    --env.control_mode=relative > "$ROOT/eval_bs${bs}.log" 2>&1
  rc=$?
  end=$(date +%s.%N)
  kill $SMI $LOAD
  peak=$(awk -F', ' '{gsub(/ MiB/,"",$2); if($2>m)m=$2} END{print m}' "$ROOT/gpu_bs${bs}.csv")
  echo "bs=$bs n_ep=$N_EP rc=$rc wall_s=$(echo "$end-$start"|bc) s_per_ep=$(echo "($end-$start)/$N_EP"|bc -l|cut -c1-6) gpu_peak_MiB=$peak" | tee -a "$ROOT/summary.txt"
done
