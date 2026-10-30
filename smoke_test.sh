#!/usr/bin/env bash
# FFI26 - smoke test: run LIBERO episodes end-to-end and save videos.
# Purpose: prove the whole chain works (checkpoint download -> policy -> simulator -> metric)
# and measure speed. A 1-episode run is NOT a result you should report.
#
# Usage:
#   conda activate lerobot
#   bash smoke_test.sh
# Overrides (env vars): POLICY, SUITE, TASK_IDS, N_EP, EVAL_BS, SEED, NAS, CM, OUT
#   NAS -> --policy.n_action_steps, CM -> --env.control_mode (flags are passed ONLY when set;
#   otherwise the checkpoint's own config is used and you must record what that was).
#   e.g. POLICY=lerobot/smolvla_base bash smoke_test.sh                              # base model
#   e.g. SUITE=libero_10 TASK_IDS="[0]" N_EP=5 EVAL_BS=5 bash smoke_test.sh          # parallel speed test
#   e.g. N_EP=5 EVAL_BS=5 CM=absolute bash smoke_test.sh                             # control-mode check

set -euo pipefail

export MUJOCO_GL="${MUJOCO_GL:-egl}"
POLICY="${POLICY:-lerobot/smolvla_libero}"
SUITE="${SUITE:-libero_10}"          # LIBERO-Long
TASK_IDS="${TASK_IDS:-[0]}"
N_EP="${N_EP:-1}"
EVAL_BS="${EVAL_BS:-1}"              # parallel envs per task; keep 1 if VRAM is short (run_queue.sh defaults to 5)
SEED="${SEED:-}"                     # optional, passed as --seed only when set (same flag run_queue.sh uses)
NAS="${NAS:-}"
CM="${CM:-}"
# The env gives cameras image/image2 but the policy expects camera1/camera2 (lerobot-eval raises
# "Feature mismatch" otherwise). Mapping copied from the train_config.json of lerobot/smolvla_libero.
# Set RENAME_MAP="" to disable, or give your own JSON (e.g. for smolvla_base, check its config first).
DEFAULT_RM='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'
RENAME_MAP="${RENAME_MAP-$DEFAULT_RM}"
OUT="${OUT:-$HOME/ffi26/results/smoke_$(date +%Y%m%d_%H%M%S)}"

# Precaution (not verified against lerobot-eval internals): keep N_EP a multiple of EVAL_BS.
if [ $(( N_EP % EVAL_BS )) -ne 0 ]; then
  echo "[smoke] WARNING: N_EP=$N_EP is not a multiple of EVAL_BS=$EVAL_BS; check how many episodes actually ran."
fi

mkdir -p "$OUT"
echo "[smoke] policy=$POLICY suite=$SUITE task_ids=$TASK_IDS episodes=$N_EP batch_size=$EVAL_BS nas=${NAS:-ckpt-default} control_mode=${CM:-ckpt-default}"
echo "[smoke] output -> $OUT"

# Record exactly which checkpoint revision we are using (needed for 'differences from original work').
python - <<PY | tee "$OUT/checkpoint_revision.txt" || true
from huggingface_hub import HfApi
info = HfApi().model_info("$POLICY")
print("repo:", info.id)
print("commit_sha:", info.sha)
print("last_modified:", info.last_modified)
PY

# For lerobot/smolvla_libero the published train_config.json says control_mode=relative and
# n_action_steps=50 (chunk_size 50). Whatever you run, write the values into the report.
cmd=(lerobot-eval
  --policy.path="$POLICY"
  --env.type=libero
  --env.task="$SUITE"
  --env.task_ids="$TASK_IDS"
  --eval.batch_size="$EVAL_BS"
  --eval.n_episodes="$N_EP"
  --env.max_parallel_tasks=1
  --output_dir="$OUT")
[ -n "$RENAME_MAP" ] && cmd+=(--rename_map="$RENAME_MAP")
[ -n "$SEED" ] && cmd+=(--seed="$SEED")
[ -n "$NAS" ] && cmd+=(--policy.n_action_steps="$NAS")
[ -n "$CM" ]  && cmd+=(--env.control_mode="$CM")
printf '%q ' "${cmd[@]}" > "$OUT/cmd.txt"; echo >> "$OUT/cmd.txt"

START=$(date +%s)
"${cmd[@]}" 2>&1 | tee "$OUT/eval.log"
END=$(date +%s)

# Wall-clock includes model load, so compare runs with the same N_EP when estimating speed.
echo "wall_clock_seconds: $(( END - START ))" | tee "$OUT/timing.txt"
echo "episodes: $N_EP  batch_size: $EVAL_BS" | tee -a "$OUT/timing.txt"

echo
echo "[smoke] finished. Look for a video under $OUT and check eval.log for the success rate."