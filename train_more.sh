#!/usr/bin/env bash
# FFI26 - train_more.sh: tiep tuc fine-tune lerobot/smolvla_libero tren lerobot/libero.
# KHONG phai tai lap paper (paper: 100k step, batch 64, khoi tao tu VLM). Day la "ban cua minh":
# khoi tao tu checkpoint 25k step, optimizer + lich lr khoi dong LAI (khong resume), seed khac.
# Ten bien/co o day: doi chieu bang MODE=PREFLIGHT truoc khi tin (chua chay thu tren lerobot 0.6.2).
#
#   conda activate lerobot   (chay trong tmux)
#   MODE=PREFLIGHT bash train_more.sh            # mac dinh: co nao lerobot-train 0.6.2 co biet -> dan output
#   MODE=SPEED     bash train_more.sh            # do giay/step + VRAM (batch 32 va 64), in uoc tinh gio
#   MODE=RUN STEPS=20000 bash train_more.sh      # train that (DRY_RUN=1 chi in lenh)
# Bien: POLICY DATASET DS_REV BATCH STEPS SAVE_FREQ LOG_FREQ TRAIN_SEED NUM_WORKERS TAG ROOT SPEED_BATCHES RENAME_MAP

set -uo pipefail
command -v lerobot-train >/dev/null || { echo "ERROR: chua bat env lerobot (conda activate lerobot)"; exit 2; }

MODE="${MODE:-PREFLIGHT}"
POLICY="${POLICY:-lerobot/smolvla_libero}"
DATASET="${DATASET:-lerobot/libero}"
DS_REV="${DS_REV:-a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4}"   # revision da ghim trong probe
BATCH="${BATCH:-32}"            # giu bang recipe cua checkpoint (train_config: 32)
STEPS="${STEPS:-}"
SAVE_FREQ="${SAVE_FREQ:-5000}"  # luu moi 5000 step -> co the eval tung moc de ve duong cong
LOG_FREQ="${LOG_FREQ:-50}"
TRAIN_SEED="${TRAIN_SEED:-2000}"  # khac seed 1000 cua train_config de khong lap lai thu tu batch dau
NUM_WORKERS="${NUM_WORKERS:-16}"
TAG="${TAG:-cont}"
ROOT="${ROOT:-$HOME/ffi26/train}"
SPEED_BATCHES="${SPEED_BATCHES:-32 64}"
DRY_RUN="${DRY_RUN:-0}"
# Map camera lay tu train_config.json cua lerobot/smolvla_libero (dataset: image/image2, policy: camera1/camera2).
DEFAULT_RM='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'
RENAME_MAP="${RENAME_MAP-$DEFAULT_RM}"

mkdir -p "$ROOT/logs"

build_cmd() {  # out_dir steps batch save_freq job_name -> mang CMD
  CMD=(lerobot-train
    --policy.path="$POLICY"
    --policy.push_to_hub=false          # config goc ghi repo_id cua tac gia: KHONG duoc push
    --policy.device=cuda
    --dataset.repo_id="$DATASET"
    --dataset.revision="$DS_REV"
    --batch_size="$3" --steps="$2"
    --num_workers="$NUM_WORKERS"
    --seed="$TRAIN_SEED"
    --save_freq="$4" --log_freq="$LOG_FREQ"
    --output_dir="$1" --job_name="$5"
    --wandb.enable=false)
  if [ -n "$RENAME_MAP" ]; then CMD+=(--rename_map="$RENAME_MAP"); fi
}

wait_gpu_free() {
  if pgrep -f run_queue.sh >/dev/null; then
    echo "[wait] run_queue.sh dang chay; doi no xong de khong tranh GPU (Ctrl+C de huy)"
    while pgrep -f run_queue.sh >/dev/null; do sleep 30; done
  fi
}

preflight() {
  H="$ROOT/logs/train_help.txt"
  lerobot-train --help > "$H" 2>&1 || true
  echo "[preflight] --help luu o $H ($(wc -l < "$H") dong)"
  echo "--- so lan moi co xuat hien trong --help (0 = KHONG thay; policy.* co the khong mo rong trong --help)"
  for f in dataset.repo_id dataset.revision rename_map batch_size steps num_workers seed output_dir job_name \
           wandb.enable save_freq log_freq save_checkpoint resume policy.path; do
    printf '%-18s ' "$f"; grep -c -- "--$f" "$H" || true
  done
  LR="$(python -c 'import lerobot,os;print(os.path.dirname(lerobot.__file__))' 2>/dev/null)"
  echo "--- policy.* va lich lr trong ma nguon (LR=$LR)"
  grep -rn -E "push_to_hub|scheduler_decay|optimizer_lr|scheduler_warmup" "$LR/configs" \
       "$LR/policies/smolvla/configuration_smolvla.py" 2>/dev/null | head -30 || true
  echo "--- dau vao da ghim: policy=$POLICY dataset=$DATASET revision=$DS_REV"
  df -h "$HOME" | awk 'NR==2{print "free disk:", $4}'
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader || true
}

speed() {
  wait_gpu_free
  mkdir -p "$ROOT/speed"
  for b in $SPEED_BATCHES; do
    unset T; declare -A T=()
    ok=1
    for n in 100 300; do
      out="$ROOT/speed/$(date +%H%M%S)_b${b}_s${n}"
      build_cmd "$out" "$n" "$b" 1000000 "speed_b${b}_s${n}"
      { printf '%q ' "${CMD[@]}"; echo; } > "$out.cmd.txt"
      if [ "$DRY_RUN" = "1" ]; then cat "$out.cmd.txt"; ok=0; continue; fi
      nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 3 > "$out.vram" 2>/dev/null &
      VP=$!
      start=$(date +%s)
      "${CMD[@]}" > "$out.log" 2>&1
      rc=$?
      secs=$(( $(date +%s) - start ))
      kill "$VP" 2>/dev/null; wait "$VP" 2>/dev/null
      vmax=$(sort -n "$out.vram" 2>/dev/null | tail -1)
      echo "$b,$n,$rc,$secs,${vmax:-}" >> "$ROOT/speed/timing.csv"
      echo "[speed] batch=$b steps=$n rc=$rc secs=$secs vram_max_MiB=${vmax:-?} (gom ca tien trinh khac ~848 MiB)"
      if [ $rc -ne 0 ]; then echo "[speed] LOI (co the OOM). 8 dong cuoi:"; tail -8 "$out.log"; ok=0; break; fi
      T[$n]=$secs
    done
    if [ $ok -eq 1 ]; then
      slope=$(awk -v a="${T[100]}" -v c="${T[300]}" 'BEGIN{printf "%.3f", (c-a)/200}')
      echo "[speed] batch=$b: ~${slope} giay/step (hieu 2 lan chay 100 va 300 step, bo thoi gian khoi dong; uoc tinh THO)"
      for s in 5000 10000 25000 50000 100000; do
        awk -v s="$s" -v k="$slope" 'BEGIN{printf "          %6d step ~ %.1f gio\n", s, s*k/3600}'
      done
    fi
  done
  echo "[speed] bang tong hop: $ROOT/speed/timing.csv (batch,steps,rc,giay,VRAM_MiB). Dir speed/ co the xoa tay khi xong."
}

run() {
  [ -n "$STEPS" ] || { echo "ERROR: dat STEPS=... (vd STEPS=20000)"; exit 2; }
  OUT="$ROOT/${TAG}_s${STEPS}_b${BATCH}"
  if [ -e "$OUT" ] && [ "$DRY_RUN" != "1" ]; then
    echo "ERROR: $OUT da ton tai (se khong ghi de). Doi TAG hoac xoa tay."; exit 2
  fi
  build_cmd "$OUT" "$STEPS" "$BATCH" "$SAVE_FREQ" "$TAG"
  printf '%q ' "${CMD[@]}"; echo
  if [ "$DRY_RUN" = "1" ]; then exit 0; fi
  wait_gpu_free
  { printf '%q ' "${CMD[@]}"; echo; } > "$ROOT/logs/${TAG}_cmd.txt"
  start=$(date +%s)
  "${CMD[@]}" 2>&1 | tee "$ROOT/logs/${TAG}.log"
  rc=${PIPESTATUS[0]}
  echo "rc=$rc seconds=$(( $(date +%s) - start ))" | tee "$ROOT/logs/${TAG}_status.txt"
  echo "[run] checkpoint o: $OUT/checkpoints/ (xem bang ls; moc cuoi thuong la 'last')"
}

case "$MODE" in
  PREFLIGHT) preflight ;;
  SPEED) speed ;;
  RUN) run ;;
  *) echo "MODE khong hop le: $MODE (PREFLIGHT|SPEED|RUN)"; exit 2 ;;
esac
