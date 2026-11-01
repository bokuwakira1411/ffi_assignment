"""FPO v4 sanity check before training (CPU, read-only). Uses src/fpo_adv.py = the code the trainer calls.
  1. unit tests: a fake group with 1 short successful rollout (3 decisions) + 1 long failed rollout (10 decisions):
     with adv_weighting "rollout" the two rollouts have the same total weight; "decision" mode reproduces the v3 formula
     bit for bit; with "segment" (v4) every segment (fake segments.csv in a temp dir) has total weight 1.
  2. iteration-0 data (outputs/fpo_v2/iter0, = v3 iter0 = v4 iter0), filter of fpo_train.load_decisions, per task:
     % decisions of successful rollouts with A^ < 0, % of failed rollouts with A^ > 0, rollout-weighted mean A^,
     for v2 (A / std over both tasks), v3 (decision, mean), v4-rollout (rejected) and v4-segment (rollout, none).
Output: ~/ffi26/results/round2/fpo_v4/sanity.md (Vietnamese), sanity.csv.

    python scripts/fpo_v4_sanity.py
"""

import glob
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fpo_adv import normalise, rollout_weights, sign_stats  # noqa: E402

H = Path.home() / "ffi26"
OUT = H / "results/round2/fpo_v4"
IT0 = ROOT / "outputs/fpo_v2/iter0"


def unit_test():
    d = {"group": np.zeros(13, np.int64), "env": np.array([0] * 3 + [1] * 10), "adv": np.array([0.5] * 3 + [-0.2] * 10,
                                                                                               np.float32)}
    w = rollout_weights(d, "rollout")
    s0, s1 = w[d["env"] == 0].sum(), w[d["env"] == 1].sum()
    assert abs(s0 - s1) < 1e-12 and abs(s0 - 1) < 1e-12, (s0, s1)
    ah, _ = normalise(d, rollout_weights(d, "decision"), "mean")
    ref = np.clip((d["adv"] - float(d["adv"].mean())) / float(d["adv"].std()), -5, 5).astype(np.float32)
    assert np.array_equal(ah, ref)
    ah4, st = normalise(d, w, "none")
    # weighted mean of A: (0.5 + -0.2) / 2 = 0.15 -> each rollout counts once
    assert abs(st["weighted_mean"] - 0.15) < 1e-6, st
    assert (ah4[:3] > 0).all() and (ah4[3:] < 0).all()
    # batch renormalisation as in the trainer: mean of w over a batch = 1
    idx = np.array([0, 1, 3, 4, 5, 6])
    wn = w[idx] / w[idx].mean()
    assert abs(wn.mean() - 1) < 1e-12
    # segment weighting: rollout 0 = 1 segment (steps 0-29), rollout 1 = 2 segments (0-19: 2 decisions, 20-99: 8)
    with tempfile.TemporaryDirectory() as td:
        pd.DataFrame({"name": ["x_j0", "x_j1", "x_j1"], "group": [0, 0, 0], "start": [0, 0, 20],
                      "end": [30, 20, 100]}).to_csv(Path(td) / "segments.csv", index=False)
        ds = dict(d, step=np.array([0, 10, 20] + list(range(0, 100, 10))))
        wsg = rollout_weights(ds, "segment", td)
    segsum = [wsg[:3].sum(), wsg[3:5].sum(), wsg[5:].sum()]
    assert all(abs(x - 1) < 1e-12 for x in segsum), segsum
    return {"segment: tổng w của 3 segment (kỳ vọng 1 / 1 / 1)": " / ".join(f"{x:.3f}" for x in segsum),
            "tổng w rollout thành công ngắn (3 decision)": s0, "tổng w rollout thất bại dài (10 decision)": s1,
            "weighted mean A (kỳ vọng 0,15)": st["weighted_mean"], "chế độ decision/mean = công thức v3": True}


def load_iter0():
    per = {}
    for f in sorted(glob.glob(str(IT0 / "decisions" / "g*.npz"))):
        z = np.load(f)
        m = (z["rejected"] == 0) & (z["adv"] != 0)
        if m.any():
            per.setdefault(int(z["task_id"][0]), []).append({k: z[k][m] for k in ("adv", "group", "env", "step")})
    return {t: {k: np.concatenate([x[k] for x in L]) for k in L[0]} for t, L in per.items()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ut = unit_test()
    data = load_iter0()
    sd_all = float(np.concatenate([d["adv"] for d in data.values()]).std())
    rows = []
    for t, d in sorted(data.items()):
        for name, wm, cm in (("v2 (A / std chung)", None, None), ("v3 (decision, trừ mean_t)", "decision", "mean"),
                             ("v4-rollout (bỏ, không chạy)", "rollout", "none"),
                             ("v4-segment (chạy)", "segment", "none")):
            w = rollout_weights(d, wm or "decision", IT0)
            ah = np.clip(d["adv"] / sd_all, -5, 5).astype(np.float32) if wm is None else normalise(d, w, cm)[0]
            s = sign_stats(d, ah, w, IT0)
            rows.append({"task": t, "cách": name, "decision": s["n"],
                         "TC: Â < 0": f"{s['succ_hat_neg_pct']:.1f}%", "TB: Â > 0": f"{s['fail_hat_pos_pct']:.1f}%",
                         "Â TB theo segment": round(s["hat_mean_segment"], 3),
                         "Â TB theo rollout": round(s["hat_mean_rollout"], 3),
                         "Â TB theo decision": round(s["hat_mean_decision"], 3),
                         "tổng w rollout TC / TB": f"{s['w_sum_success_rollouts']:.0f} / {s['w_sum_fail_rollouts']:.0f}",
                         "số rollout TC / TB": f"{s['n_success_rollouts']} / {s['n_fail_rollouts']}",
                         "TC: A thô < 0": f"{s['succ_raw_neg_pct']:.1f}%", "TB: A thô > 0": f"{s['fail_raw_pos_pct']:.1f}%"})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "sanity.csv", index=False)
    md = lambda d: "\n".join(["| " + " | ".join(map(str, d.columns)) + " |", "|" + "---|" * len(d.columns)]  # noqa: E731
                             + ["| " + " | ".join(str(v) for v in r) + " |" for r in d.itertuples(index=False)])
    L = ["# FPO v4: sanity check trước khi train (CPU)", "",
         "Script `improve/scripts/fpo_v4_sanity.py`, dùng `src/fpo_adv.py` (cùng code với trainer). Dữ liệu iter 0: "
         "`outputs/fpo_v2/iter0` (= v3 iter 0 = v4 iter 0), lọc như `load_decisions`. TC = rollout thành công, TB = thất bại.", "",
         "## 1. Unit test (nhóm giả: 1 rollout thành công 3 decision, 1 thất bại 10 decision; segment giả): ĐẠT", "",
         "\n".join(f"- {k}: {v}" for k, v in ut.items()), "",
         "## 2. Iter 0: dấu của Â theo cách chuẩn hoá", "",
         "Số v3 phải trùng với `results/round2/fpo_v3/adv_check.md` (iter 0: TC Â < 0 = 1,1 % / 1,6 %, TB Â > 0 = 52,6 % / 37,0 %).",
         "", md(df), ""]
    (OUT / "sanity.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
