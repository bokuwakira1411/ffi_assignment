#!/usr/bin/env bash
# Wait for the fpo_v2 loop to finish, then run the video replays (results/videos/configs/videos.yaml):
#   (a) official + v1 theta3, (b) the v2 checkpoint chosen in outputs/fpo_v2/selection.json, same episodes.
# Starts only if the loop finished cleanly: process gone, "=== [done]" (printed only after every step of
# fpo_v2_loop.sh succeeded under set -euo pipefail) in loop.log, selection.json and all 5 report summary.md
# exist. Otherwise nothing runs and the reason is appended to docs/WORKLOG.md.
#     bash scripts/videos_after_v2.sh <loop_pid>     (in tmux; log -> results/videos/run.log)
set -uo pipefail
source /opt/miniconda3/etc/profile.d/conda.sh && conda activate lerobot
cd ~/ffi26/improve
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
PID=$1
O=outputs/fpo_v2; V=~/ffi26/results/videos; R=~/ffi26/results/round2/fpo_v2
WL=/home/student/Lan_K68AI/temp/docs/WORKLOG.md
STATUS=$V/watcher_status.txt
log() { echo "$(date '+%F %T') $*" | tee -a "$STATUS"; }

log "waiting for fpo_v2 loop pid $PID"
while kill -0 "$PID" 2>/dev/null; do sleep 120; done
log "loop pid $PID exited"

why=()
grep -q '^=== \[done\]' $O/loop.log || why+=("không có dòng '=== [done]' trong $O/loop.log")
[ -f $O/selection.json ] || why+=("thiếu $O/selection.json")
for d in seed3000 seed4000 seed5000 seed1000_other seed1000_precision; do
  [ -f $R/$d/summary.md ] || why+=("thiếu $R/$d/summary.md")
done
if [ ${#why[@]} -gt 0 ]; then
  log "NOT STARTING: ${why[*]}"
  {
    echo; echo
    echo "## $(date '+%d/%m/%Y %H:%M'): watcher video KHÔNG chạy (loop fpo_v2 không kết thúc sạch)"
    echo
    echo "Watcher (\`improve/scripts/videos_after_v2.sh\`, tmux \`videos\`) thấy process loop (pid $PID) đã dừng nhưng:"
    for w in "${why[@]}"; do echo "- $w"; done
    echo
    echo "Không chạy gì trên GPU. 30 dòng cuối của \`$O/loop.log\`:"
    echo; echo '```'; tail -30 $O/loop.log | cut -c1-300; echo '```'
    echo; echo "\`$O/timing.txt\`:"; echo; echo '```'; cat $O/timing.txt; echo '```'
  } >> "$WL"
  echo FAILED_PRECHECK > $V/watcher_done
  exit 1
fi

# v2's GPU monitor would otherwise log the video runs into outputs/fpo_v2/gpu.csv
tmux kill-session -t fpo_v2_gpu 2>/dev/null && log "stopped tmux fpo_v2_gpu (gpu.csv rows after this time are not v2)"

log "start (a) official + v1t3"
python scripts/phase3_eval_video.py --config $V/configs/videos.yaml --only official,v1t3; rc_a=$?
log "(a) exit $rc_a"

SEL=$(python -c "import json;print(json.load(open('$O/selection.json'))['path'])")
sed -e 's|^  # - name: v2|  - name: v2|' -e 's|^  #   |    |' -e "s|V2_SEL_PATH|$SEL|" $V/configs/videos.yaml > $V/configs/videos_v2.yaml
log "v2 checkpoint: $SEL; config $V/configs/videos_v2.yaml"
log "start (b) v2"
python scripts/phase3_eval_video.py --config $V/configs/videos_v2.yaml --only v2; rc_b=$?
log "(b) exit $rc_b"

python scripts/videos_index.py --config $V/configs/videos_v2.yaml; rc_i=$?
log "index exit $rc_i"
{
  echo; echo
  echo "## $(date '+%d/%m/%Y %H:%M'): video replay xong (watcher tự chạy)"
  echo
  echo "Loop fpo_v2 kết thúc sạch. Watcher dừng tmux \`fpo_v2_gpu\` rồi chạy (a) official + v1 θ3 (exit $rc_a), (b) v2 = \`$SEL\` (exit $rc_b), index (exit $rc_i)."
  echo "Kết quả: \`results/videos/\` (\`index.md\`, \`replay_check_*.csv\`, \`episodes_*.csv\`, \`raw/\`). Log: \`results/videos/run.log\`, \`watcher_status.txt\`."
} >> "$WL"
echo "DONE a=$rc_a b=$rc_b i=$rc_i" > $V/watcher_done
log "done"
