#!/usr/bin/env bash
# FPO v4 loop (config results/round2/configs/fpo_v4.yaml; = fpo_v3_loop.sh, same trainer fpo_train_v3.py with the v4 flags):
#   iteration 0 collection is REUSED from outputs/fpo_v2/iter0 (theta0, same collection config / seeds; decisions/
#   and logs/ symlinked, small files copied), after checking the collect / reward / n_action_steps / expert_fp32 /
#   init_policy fields of both configs are identical;
#   for k = 0..K-1: [collect with theta_k (fpo_collect.py), k > 0] -> train (fpo_train_v3.py) -> theta_{k+1};
#   collect iteration K with theta_K (no training; collapse rule only); select by the pre-registered collapse rule;
#   eval v4 ONLY (official / v2 / v3 episodes reused from outputs/fpo_v2/eval_s*, outputs/fpo_v3/eval_s*, same protocol):
#   tasks 0, 8 x 50 at seeds 3000 / 4000 / 5000 FIRST, then seed 1000 other 8 tasks x 20 (forgetting); reports
#   (r2_report.py, reference official) + pooled / paired stats vs official and vs v3 (fpo_v4_pooled.py).
# Timing -> outputs/fpo_v4/timing.txt; per-iteration summary -> outputs/fpo_v4/iter_summary.csv.
set -euo pipefail
source /opt/miniconda3/etc/profile.d/conda.sh && conda activate lerobot
cd ~/ffi26/improve
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
C=~/ffi26/results/round2/configs; CFG=$C/fpo_v4.yaml; O=outputs/fpo_v4; TIM=$O/timing.txt
mkdir -p $O/configs
step() { local name="$1"; shift; echo "$name start $(date +%s)" >> "$TIM"; echo "=== [$name] $(date) $*"; "$@"; echo "$name end $(date +%s)" >> "$TIM"; }
K=$(python -c "import yaml;print(yaml.safe_load(open('$CFG'))['iterations'])")
POL=$(python -c "import yaml;print(yaml.safe_load(open('$CFG'))['init_policy'])")

# iteration 0 collection: reuse v2's (identical config fields checked here; v4 = v3 = v2 for these fields)
if [ ! -f $O/iter0/collect_info.json ]; then
  python - "$C/fpo_v2.yaml" "$CFG" <<'PY'
import sys, yaml
a, b = (yaml.safe_load(open(p)) for p in sys.argv[1:])
for k in ("collect", "reward", "n_action_steps", "expert_fp32", "init_policy"):
    assert a[k] == b[k], f"config field {k} differs between v2 and v3: cannot reuse v2 iteration 0"
print("[iter0] v2 and v3 collection config identical -> reuse outputs/fpo_v2/iter0")
PY
  mkdir -p $O/iter0
  ln -s ~/ffi26/improve/outputs/fpo_v2/iter0/decisions $O/iter0/decisions
  ln -s ~/ffi26/improve/outputs/fpo_v2/iter0/logs $O/iter0/logs
  cp outputs/fpo_v2/iter0/{groups.csv,rollouts.csv,segments.csv} $O/iter0/
  cp outputs/fpo_v2/iter0/collect_info.json $O/iter0/collect_info.json
  echo "reused_from_v2_iter0 $(date +%s)" >> "$TIM"
fi

for ((k=0; k<K; k++)); do
  if [ ! -f $O/iter$k/train/pretrained_model/model.safetensors ]; then
    [ -f $O/iter$k/collect_info.json ] || step collect$k python scripts/fpo_collect.py --config $CFG --iter $k --policy $POL --out $O/iter$k
    step train$k python scripts/fpo_train_v3.py --config $CFG --policy $POL --data $O/iter$k --out $O/iter$k/train
  fi
  POL=$HOME/ffi26/improve/$O/iter$k/train/pretrained_model
done
[ -f $O/iter$K/collect_info.json ] || step collect$K python scripts/fpo_collect.py --config $CFG --iter $K --policy $POL --out $O/iter$K

# collapse rule (pre-registered, = v2): theta_K unless success(k+1) < success(k) - 15 points; then theta_k (first drop)
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
for s in 1000 3000 4000 5000; do sed "s#V4_SEL_PATH#$SEL#" $C/eval_fpo_v4_s$s.tmpl.yaml > $O/configs/eval_fpo_v4_s$s.yaml; done
# v4 eval order (requested): trained tasks 0, 8 at seeds 3000 / 4000 / 5000 FIRST, then forgetting (seed 1000)
for s in 3000 4000 5000; do
  step eval_s$s python scripts/phase3_eval.py --config $O/configs/eval_fpo_v4_s$s.yaml
  step report_s$s python scripts/r2_report.py --config $C/report_fpo_v4_s$s.yaml || echo "[warn] report_s$s failed"
done
step eval_s1000 python scripts/phase3_eval.py --config $O/configs/eval_fpo_v4_s1000.yaml
step report_other python scripts/r2_report.py --config $C/report_fpo_v4_other.yaml || echo "[warn] report_other failed"
[ -f scripts/fpo_v4_pooled.py ] && { step pooled python scripts/fpo_v4_pooled.py || echo "[warn] pooled failed"; }
echo "=== [done] $(date)"
