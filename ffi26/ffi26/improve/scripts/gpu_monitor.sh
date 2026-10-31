#!/usr/bin/env bash
# Sample whole-GPU memory used (MiB) every 2 s into $1 (csv: unix_time,mib). Stop with tmux kill-session.
while true; do echo "$(date +%s),$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" >> "$1"; sleep 2; done
