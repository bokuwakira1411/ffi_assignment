"""Relative weight change of FPO checkpoints vs the official one, per tensor: |W - W0| / |W0| (Frobenius). CPU only.
Also: the size of the PPO trust region in CFM-loss units, from c = ratio_scale_used (train_info.json):
a sample is clipped when |c * (l_old - l_new)| > log(1.2) ~ 0.18.
Output: ~/ffi26/results/round2/fpo_v3/weight_delta.csv (v2, v3, v4) (+ printed summary).

    CUDA_VISIBLE_DEVICES="" python scripts/fpo_weight_delta.py
"""

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from rollout import checkpoint_file  # noqa: E402

O = ROOT / "outputs"
CK = {f"{r}_theta{k + 1}": O / f"fpo_{r}/iter{k}/train/pretrained_model" for r in ("v2", "v3", "v4") for k in range(4)}


def main():
    w0 = {k: v.float() for k, v in load_file(str(checkpoint_file("lerobot/smolvla_libero"))).items()}
    rows = []
    for name, p in CK.items():
        w = load_file(str(p / "model.safetensors"))
        rel = {k: float((w[k].float() - w0[k]).norm() / (w0[k].norm() + 1e-12)) for k in w0}
        ch = {k: r for k, r in rel.items() if r > 0}
        ex = [r for k, r in ch.items() if "lm_expert" in k]
        num = sum(float((w[k].float() - w0[k]).norm() ** 2) for k in ch)
        den = sum(float(w0[k].norm() ** 2) for k in ch)
        rows.append({"checkpoint": name, "tensors_changed": len(ch), "of": len(w0), "expert_changed": len(ex),
                     "rel_median": np.median(list(ch.values())), "rel_max": max(ch.values()),
                     "rel_global_trainable": math.sqrt(num / den)})
    df = pd.DataFrame(rows)
    out = Path.home() / "ffi26/results/round2/fpo_v3/weight_delta.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    for r in ("v2", "v3", "v4"):
        i = json.load(open(O / f"fpo_{r}/iter0/train/train_info.json"))
        c, lo = i["ratio_scale_used"], i["mean_l_old"]
        print(f"{r} iter0: c = {c:.2f}, mean l_old = {lo:.4f}; clipped when |l_old - l_new| > {math.log(1.2) / c:.4f} "
              f"= {100 * math.log(1.2) / c / lo:.0f} % of mean l_old (with c = 1 the threshold would be {math.log(1.2):.3f})")


if __name__ == "__main__":
    main()
