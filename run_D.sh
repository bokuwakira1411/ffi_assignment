#!/usr/bin/env bash
# Ablation D: NAS=1 va NAS=50, 10 task, cung seed, cung N_EP.
# Seed la vong ngoai, moi seed chay ca hai NAS lien nhau de luon co cap day du.
# Bien: D_SEEDS (mac dinh "1000"), D_NEP (mac dinh 50), D_NAS (mac dinh "1 50")
command -v lerobot-eval >/dev/null || { echo "ERROR: chua bat env lerobot (conda activate lerobot)"; exit 2; }
cd ~/Lan_K68AI/temp || exit 2
D_SEEDS="${D_SEEDS:-1000}"; D_NEP="${D_NEP:-50}"; D_NAS="${D_NAS:-1 50}"
echo "[D] seeds=[$D_SEEDS] n_ep=$D_NEP nas=[$D_NAS]  $(date)"
if [ "${DRY_RUN:-0}" != "1" ]; then
  while pgrep -f run_queue.sh >/dev/null; do sleep 30; done   # doi queue B_main xong
fi
for s in $D_SEEDS; do
  for nas in $D_NAS; do
    TAG=D_nas${nas} NAS=$nas CM=relative N_EP=$D_NEP SEEDS="$s" bash run_queue.sh
  done
done
echo "[D] finished $(date)"
