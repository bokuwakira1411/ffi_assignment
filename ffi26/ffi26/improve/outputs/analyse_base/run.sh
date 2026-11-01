#!/bin/bash
# Runs the three Analyse base-model evals in sequence; log in outputs/analyse_base/run.log
source /opt/miniconda3/etc/profile.d/conda.sh && conda activate lerobot
cd ~/ffi26/improve
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
for s in 3000 4000 5000; do
  echo "eval_s$s start $(date +%s)"
  python scripts/phase3_eval.py --config outputs/analyse_base/configs/eval_base_s$s.yaml || exit 1
  echo "eval_s$s end $(date +%s)"
done
