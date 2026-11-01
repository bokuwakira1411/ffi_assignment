#!/bin/bash
# kill task0_grad if any v4 collection (iter>=1) took > 900 s of rollout; exit when task0_grad ends
cd /home/student/ffi26/improve
while tmux has-session -t task0_grad 2>/dev/null; do
  for f in outputs/fpo_v4/iter[1-9]/collect_info.json; do
    [ -f "$f" ] || continue
    s=$(python -c "import json;print(int(json.load(open('$f')).get('seconds_rollout',0)))")
    if [ "$s" -gt 900 ]; then echo "$(date) $f seconds_rollout=$s > 900 -> kill task0_grad" >> outputs/task0_diag/watchdog.log; tmux kill-session -t task0_grad; exit 0; fi
  done
  sleep 60
done
echo "$(date) task0_grad ended" >> outputs/task0_diag/watchdog.log
