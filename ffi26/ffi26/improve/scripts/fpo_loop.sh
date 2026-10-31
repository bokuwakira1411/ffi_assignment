#!/usr/bin/env bash
# FPO loop (plan: results/round2/fpo_plan.md): for k = 0..K-1: collect with theta_k -> train -> theta_{k+1};
# then eval theta_1..theta_K (eval_fpo.yaml) and report. Timing -> outputs/fpo/timing.txt.
set -euo pipefail
source /opt/miniconda3/etc/profile.d/conda.sh && conda activate lerobot
cd ~/ffi26/improve
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
C=~/ffi26/results/round2/configs; O=outputs/fpo; TIM=$O/timing.txt
mkdir -p $O
step() { local name="$1"; shift; echo "$name start $(date +%s)" >> "$TIM"; echo "=== [$name] $(date) $*"; "$@"; echo "$name end $(date +%s)" >> "$TIM"; }
K=$(python -c "import yaml;print(yaml.safe_load(open('$C/fpo.yaml'))['iterations'])")
POL=$(python -c "import yaml;print(yaml.safe_load(open('$C/fpo.yaml'))['init_policy'])")
for ((k=0; k<K; k++)); do
  if [ ! -f $O/iter$k/train/pretrained_model/model.safetensors ]; then
    [ -f $O/iter$k/collect_info.json ] || step collect$k python scripts/fpo_collect.py --config $C/fpo.yaml --iter $k --policy $POL --out $O/iter$k
    step train$k python scripts/fpo_train.py --config $C/fpo.yaml --policy $POL --data $O/iter$k --out $O/iter$k/train
  fi
  POL=$HOME/ffi26/improve/$O/iter$k/train/pretrained_model
done
step eval   python scripts/phase3_eval.py --config $C/eval_fpo.yaml
step report python scripts/r2_report.py --config $C/report.yaml
echo "=== [done] $(date)"
