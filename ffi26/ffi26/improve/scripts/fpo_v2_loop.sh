#!/usr/bin/env bash
# FPO v2 loop (config results/round2/configs/fpo_v2.yaml, expert_fp32: true):
#   for k = 0..K-1: collect with theta_k (fpo_collect.py) -> train (fpo_train_v2.py) -> theta_{k+1};
#   collect iteration K with theta_K (no training; only for the collapse rule);
#   select the checkpoint by the pre-registered collapse rule (collection success only) -> outputs/fpo_v2/selection.json;
#   eval (expert fp32, all arms): official vs v1 theta3 vs v2 selected, tasks 0, 8 x 50 at seeds 3000/4000/5000;
#   seed 1000: other 8 tasks x 20 for all arms + official tasks 0, 8 (precision sanity row); reports.
# Timing -> outputs/fpo_v2/timing.txt; per-iteration summary -> outputs/fpo_v2/iter_summary.csv.
set -euo pipefail
source /opt/miniconda3/etc/profile.d/conda.sh && conda activate lerobot
cd ~/ffi26/improve
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
C=~/ffi26/results/round2/configs; CFG=$C/fpo_v2.yaml; O=outputs/fpo_v2; TIM=$O/timing.txt
mkdir -p $O/configs
step() { local name="$1"; shift; echo "$name start $(date +%s)" >> "$TIM"; echo "=== [$name] $(date) $*"; "$@"; echo "$name end $(date +%s)" >> "$TIM"; }
K=$(python -c "import yaml;print(yaml.safe_load(open('$CFG'))['iterations'])")
POL=$(python -c "import yaml;print(yaml.safe_load(open('$CFG'))['init_policy'])")
for ((k=0; k<K; k++)); do
  if [ ! -f $O/iter$k/train/pretrained_model/model.safetensors ]; then
    [ -f $O/iter$k/collect_info.json ] || step collect$k python scripts/fpo_collect.py --config $CFG --iter $k --policy $POL --out $O/iter$k
    step train$k python scripts/fpo_train_v2.py --config $CFG --policy $POL --data $O/iter$k --out $O/iter$k/train
  fi
  POL=$HOME/ffi26/improve/$O/iter$k/train/pretrained_model
done
[ -f $O/iter$K/collect_info.json ] || step collect$K python scripts/fpo_collect.py --config $CFG --iter $K --policy $POL --out $O/iter$K

# collapse rule (pre-registered): theta_K unless success(k+1) < success(k) - 15 points; then theta_k (first drop)
python - "$CFG" "$O" <<'PY'
import json, os, sys, yaml
cfg, O = yaml.safe_load(open(sys.argv[1])), sys.argv[2]
K = cfg["iterations"]
s = [json.load(open(f"{O}/iter{k}/collect_info.json")) for k in range(K + 1)]
pct = [100.0 * x["n_success"] / x["n_rollouts"] for x in s]
chosen, why = K, "no drop > 15 points"
for k in range(K):
    if pct[k + 1] < pct[k] - 15:
        chosen, why = k, f"success iteration {k+1} = {pct[k+1]:.1f}% < iteration {k} = {pct[k]:.1f}% - 15"
        break
home = os.path.expanduser("~/ffi26/improve")
path = cfg["init_policy"] if chosen == 0 else f"{home}/{O}/iter{chosen - 1}/train/pretrained_model"
sel = {"collect_success_pct": pct, "collect_success": [f"{x['n_success']}/{x['n_rollouts']}" for x in s],
       "chosen_theta": chosen, "path": path, "reason": why}
json.dump(sel, open(f"{O}/selection.json", "w"), indent=1)
print("[selection]", json.dumps(sel))
PY
SEL=$(python -c "import json;print(json.load(open('$O/selection.json'))['path'])")
for s in 1000 3000 4000 5000; do sed "s#V2_SEL_PATH#$SEL#" $C/eval_fpo_v2_s$s.tmpl.yaml > $O/configs/eval_fpo_v2_s$s.yaml; done
for s in 3000 4000 5000 1000; do step eval_s$s python scripts/phase3_eval.py --config $O/configs/eval_fpo_v2_s$s.yaml; done
for s in 3000 4000 5000; do step report_s$s python scripts/r2_report.py --config $C/report_fpo_v2_s$s.yaml; done
step report_other python scripts/r2_report.py --config $C/report_fpo_v2_other.yaml
step report_sanity python scripts/r2_report.py --config $C/report_fpo_v2_sanity.yaml
echo "=== [done] $(date)"
