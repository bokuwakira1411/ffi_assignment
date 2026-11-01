#!/bin/bash
# Chờ loop FPO v3 (scripts/fpo_v3_loop.sh) kết thúc và GPU còn >= 6000 MiB trống, rồi chạy render_commands.sh (video "mù").
# Kiểm tra mỗi 2 phút. Trạng thái: render_watch_status.txt; log render: render.log.
cd "$(dirname "$0")"
while true; do
  loop=$(pgrep -f "fpo_v3_loop.sh" | wc -l)
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  echo "$(date '+%F %T') v3_loop_procs=$loop gpu_free_MiB=$free" > render_watch_status.txt
  if [ "$loop" -eq 0 ] && [ "$free" -ge 6000 ]; then break; fi
  sleep 120
done
echo "$(date '+%F %T') start render" >> render_watch_status.txt
bash render_commands.sh > render.log 2>&1
echo "$(date '+%F %T') render exit $?" >> render_watch_status.txt
