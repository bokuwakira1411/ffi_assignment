#!/usr/bin/env bash
# Phase 2 (A.1) fine-tune from lerobot/smolvla_libero on the merged demos+rollouts dataset.
# Only the action expert + state projection are trained (VLM and vision encoder frozen).
# Usage: bash scripts/phase2_train.sh <dataset_root> <repo_id> <output_dir> [steps] [batch_size]
set -euo pipefail
source /opt/miniconda3/etc/profile.d/conda.sh && conda activate lerobot
DS_ROOT="$1"; REPO_ID="$2"; OUT="$3"; STEPS="${4:-2000}"; BS="${5:-32}"
RM='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'
lerobot-train \
  --policy.path=lerobot/smolvla_libero \
  --policy.push_to_hub=false \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --policy.train_state_proj=true \
  --policy.optimizer_lr=1e-5 \
  --policy.scheduler_warmup_steps=100 \
  --policy.scheduler_decay_steps="$STEPS" \
  --policy.scheduler_decay_lr=2.5e-6 \
  --dataset.repo_id="$REPO_ID" \
  --dataset.root="$DS_ROOT" \
  --rename_map="$RM" \
  --batch_size="$BS" \
  --steps="$STEPS" \
  --save_freq=1000 \
  --log_freq=50 \
  --num_workers=8 \
  --seed=1000 \
  --accelerator.mixed_precision=bf16 \
  --wandb.enable=false \
  --output_dir="$OUT"
