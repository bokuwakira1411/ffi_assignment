#!/bin/bash
# wait for FPO v4 to finish (loop.log "=== [done]" or tmux session fpo_v4 gone), then >= 8 GB free VRAM, then
# part B on GPU: n_dec 32 (original design) and 256, theta0..theta3, 5 splits each.
cd /home/student/ffi26/improve
L=outputs/task0_diag/run_after_v4.log
echo "$(date) waiting for v4" >> $L
until grep -q "=== \[done\]" outputs/fpo_v4/loop.log || ! tmux has-session -t fpo_v4 2>/dev/null; do sleep 60; done
echo "$(date) v4 finished: $(grep '=== \[done\]' outputs/fpo_v4/loop.log || echo 'NO done marker, session fpo_v4 gone')" >> $L
while true; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  [ "$free" -ge 8192 ] && break
  echo "$(date) GPU free ${free} MiB < 8192, waiting" >> $L; sleep 120
done
echo "$(date) GPU free ${free} MiB, start" >> $L
PY=/home/student/.conda/envs/lerobot/bin/python
for n in 32 256; do
  $PY scripts/task0_diag_grad.py --device cuda --iters 0 1 2 3 --reps 5 --n_dec $n --mb 8 --threads 4 \
     --out outputs/task0_diag/grad_cos_gpu_n$n.csv > outputs/task0_diag/grad_gpu_n$n.log 2>&1
  echo "$(date) n_dec $n rc=$?" >> $L
done
echo "$(date) ALL DONE" >> $L
