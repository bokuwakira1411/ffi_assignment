#!/bin/bash
# DSRL queue (docs/WORKLOG.md 01/11/2026). Chờ loop FPO v3 kết thúc và GPU trống >= 8000 MiB, rồi:
#   1) kiểm tra tất định trên GPU (dsrl_step0_policy.py --device cuda)
#   2) song song: noise probe (dsrl_probe.py) và DSRL nhỏ (dsrl_train.py)
# Hạn: probe chỉ bắt đầu nếu trước 20:30, DSRL chỉ bắt đầu nếu trước 19:30 (hạn chót 22:00).
cd /home/student/ffi26/improve
source /opt/miniconda3/etc/profile.d/conda.sh && conda activate lerobot
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
O=outputs/dsrl
st() { echo "$(date '+%F %T') $*" >> $O/queue_status.txt; }
while true; do
  loop=$(pgrep -f "fpo_v3_loop.sh" | wc -l)
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  echo "$(date '+%F %T') waiting: v3_loop_procs=$loop gpu_free_MiB=$free" > $O/queue_wait.txt
  if [ "$loop" -eq 0 ] && [ "$free" -ge 8000 ]; then break; fi
  sleep 120
done
st "start (gpu_free_MiB=$free)"
python scripts/dsrl_step0_policy.py --device cuda > $O/step0_gpu.log 2>&1; st "step0 gpu exit $?"
now=$(date +%H%M)
if [ "$now" -lt 2030 ]; then
  python scripts/dsrl_probe.py --out $O/probe > $O/probe.log 2>&1 & P=$!; st "probe started"
else st "probe SKIPPED (start after 20:30)"; fi
if [ "$now" -lt 1930 ]; then
  python scripts/dsrl_train.py --out $O/run1 > $O/run1.log 2>&1 & D=$!; st "dsrl started"
else st "dsrl SKIPPED (start after 19:30)"; fi
[ -n "$P" ] && { wait $P; st "probe exit $?"; }
[ -n "$D" ] && { wait $D; st "dsrl exit $?"; }
st "queue done"
