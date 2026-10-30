#!/usr/bin/env bash
# FFI26 - experiment queue: runs a grid of lerobot-eval jobs (task x seed), resumable,
# one output folder per job. A failed job is logged and the queue keeps going.
#
# Run it inside tmux/nohup so it survives a closed terminal:
#   conda activate lerobot
#   PREFLIGHT=1 bash run_queue.sh          # print which flags this lerobot-eval version knows
#   DRY_RUN=1  bash run_queue.sh           # print the commands, run nothing
#
# Examples (each is one invocation; TAG names the experiment folder):
#   B  main eval, 3 seeds:  TAG=B_main SEEDS="1000 1001 1002" bash run_queue.sh
#   C  (base contrast: NOT runnable as is. smolvla_base outputs 6-D actions and LIBERO needs 7-D,
#       see notes/differences.md. Use the freed GPU hours for more seeds of B or for D/E.)
#   D  action-steps ablation on weak tasks (choose TASKS after B):
#                           TAG=D_nas1 TASKS="3 7" NAS=1 bash run_queue.sh
#                           TAG=D_nas50 TASKS="3 7" NAS=50 bash run_queue.sh
#   E  control-mode check:  TAG=E_abs TASKS="0 1 2" CM=absolute N_EP=10 bash run_queue.sh
#
# Env vars: POLICY SUITE TASKS SEEDS N_EP EVAL_BS NAS CM TAG ROOT
# NAS -> --policy.n_action_steps, CM -> --env.control_mode (only passed when set).
# Seeds are the OUTER loop: if time runs out you still have complete seeds.

set -uo pipefail   # no -e on purpose: one failed job must not stop the queue

export MUJOCO_GL="${MUJOCO_GL:-egl}"
POLICY="${POLICY:-lerobot/smolvla_libero}"
SUITE="${SUITE:-libero_10}"
TASKS="${TASKS:-0 1 2 3 4 5 6 7 8 9}"
SEEDS="${SEEDS:-1000}"
N_EP="${N_EP:-10}"
EVAL_BS="${EVAL_BS:-5}"       # set to the value you measured in the speed test
NAS="${NAS:-}"
CM="${CM:-}"
# Camera names: env gives image/image2, policy expects camera1/camera2 (see smoke_test.sh).
# RENAME_MAP="" disables it; override for a policy whose config uses other camera names.
DEFAULT_RM='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'
RENAME_MAP="${RENAME_MAP-$DEFAULT_RM}"
TAG="${TAG:-run}"
ROOT="${ROOT:-$HOME/ffi26/results/runs}"
DRY_RUN="${DRY_RUN:-0}"

if [ "${PREFLIGHT:-0}" = "1" ]; then
  echo "[preflight] flags this lerobot-eval accepts (empty line = flag NOT found)."
  echo "[preflight] NOTE: --env.* options only appear in --help once --env.type is given, so we pass it."
  echo "--- seed / env options";   lerobot-eval --env.type=libero --help 2>&1 | grep -i -E "seed|control_mode|episode_length|task_ids" || true
  echo "--- n_action_steps (read from the SmolVLA config source, --help may not expand policy fields)"
  LR="$(python -c 'import lerobot,os;print(os.path.dirname(lerobot.__file__))')"
  grep -n "n_action_steps" "$LR/policies/smolvla/configuration_smolvla.py" | head -3 || true
  echo "--- video / rendering related"; lerobot-eval --env.type=libero --help 2>&1 | grep -i -E "video|render" | head -5 || true
  echo "[preflight] cpu cores: $(nproc)   free disk: $(df -h "$HOME" | awk 'NR==2{print $4}')"
  exit 0
fi

POL_NAME="${POLICY//\//_}"
BASE="$ROOT/$TAG/$POL_NAME"
mkdir -p "$BASE"

# Record the exact checkpoint revision once per experiment folder.
if [ "$DRY_RUN" != "1" ] && [ ! -f "$BASE/checkpoint_revision.txt" ]; then
  python - <<PY > "$BASE/checkpoint_revision.txt" 2>&1 || true
from huggingface_hub import HfApi
i = HfApi().model_info("$POLICY")
print("repo:", i.id); print("commit_sha:", i.sha); print("last_modified:", i.last_modified)
PY
fi

echo "[queue] policy=$POLICY suite=$SUITE tasks=[$TASKS] seeds=[$SEEDS] n_ep=$N_EP bs=$EVAL_BS nas=${NAS:-default} cm=${CM:-default}"
echo "[queue] output -> $BASE"

for seed in $SEEDS; do
  for t in $TASKS; do
    OUT="$BASE/task${t}_seed${seed}"
    if [ -f "$OUT/DONE" ]; then echo "[skip] task $t seed $seed (done)"; continue; fi
    mkdir -p "$OUT"

    cmd=(lerobot-eval
      --policy.path="$POLICY"
      --env.type=libero
      --env.task="$SUITE"
      --env.task_ids="[$t]"
      --eval.batch_size="$EVAL_BS"
      --eval.n_episodes="$N_EP"
      --env.max_parallel_tasks=1
      --seed="$seed"
      --output_dir="$OUT")
    [ -n "$RENAME_MAP" ] && cmd+=(--rename_map="$RENAME_MAP")
    [ -n "$NAS" ] && cmd+=(--policy.n_action_steps="$NAS")
    [ -n "$CM" ]  && cmd+=(--env.control_mode="$CM")
    printf '%q ' "${cmd[@]}" > "$OUT/cmd.txt"; echo >> "$OUT/cmd.txt"

    if [ "$DRY_RUN" = "1" ]; then cat "$OUT/cmd.txt"; continue; fi

    echo "[run ] task $t seed $seed  ($(date +%H:%M:%S))"
    start=$(date +%s)
    "${cmd[@]}" > "$OUT/eval.log" 2>&1
    rc=$?
    secs=$(( $(date +%s) - start ))
    echo "rc=$rc seconds=$secs" > "$OUT/status.txt"
    echo "$TAG,$POL_NAME,$t,$seed,$N_EP,$EVAL_BS,${NAS:-},${CM:-},$rc,$secs" >> "$ROOT/$TAG/timing.csv"
    if [ $rc -eq 0 ]; then touch "$OUT/DONE"; echo "[ ok ] ${secs}s"; else echo "[FAIL] rc=$rc, see $OUT/eval.log"; fi
  done
done
echo "[queue] finished $TAG"