"""FPO v4: paired comparison v4 vs official and v4 vs v3, recomputed from the raw per-episode files (ep*.meta.json).

Arms (all expert fp32, same protocol): official, v2 from outputs/fpo_v2/eval_s*; v3 from outputs/fpo_v3/eval_s*;
v4 from outputs/fpo_v4/eval_s*/fpo_v4_sel. Trained tasks 0, 8 x 50 init, seeds 3000 / 4000 / 5000 (total, per task,
held-out / training init, task x split, per seed); forgetting: other 8 tasks x 20, seed 1000.
Pairs = same (seed, task, init). Exact two-sided McNemar. Missing arms / episodes are reported, never imputed.
Then the pre-registered predictions (docs/WORKLOG.md, written before training) are checked with the criteria fixed there:
  (a) task 0: McNemar v4 vs official p > 0.05 AND v4 successes > v3 (46/150);
  (b) task 8: McNemar v4 vs official p < 0.05 AND v4 > official;
  (c) forgetting total: v4 >= v3 (115/160).
Outputs: ~/ffi26/results/round2/fpo_v4/pooled.md (Vietnamese), pooled.json, episodes_all.csv.

    python scripts/fpo_v4_pooled.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from fpo_v3_pooled import OTHER_SEED, OTHERS, TRAIN_SEEDS, TRAINED, mcnemar_exact, pct  # noqa: E402

H = Path.home() / "ffi26"
ARMS = {"official": H / "improve/outputs/fpo_v2/eval_s{seed}/official",
        "v2": H / "improve/outputs/fpo_v2/eval_s{seed}/fpo_v2_sel",
        "v3": H / "improve/outputs/fpo_v3/eval_s{seed}/fpo_v3_sel",
        "v4": H / "improve/outputs/fpo_v4/eval_s{seed}/fpo_v4_sel"}
OUT = H / "results/round2/fpo_v4"


def load() -> pd.DataFrame:
    rows = []
    for arm, root in ARMS.items():
        for seed, tasks in [(s, TRAINED) for s in TRAIN_SEEDS] + [(OTHER_SEED, OTHERS)]:
            for t in tasks:
                for f in sorted((Path(str(root).format(seed=seed)) / f"task{t}").glob("ep*.meta.json")):
                    m = json.load(open(f))
                    rows.append({"arm": arm, "seed": seed, "task": t, "init": int(f.name[2:5]),
                                 "success": bool(m["success"])})
    return pd.DataFrame(rows)


def paired(ep, a, b, mask):
    e = ep[mask(ep)]
    x = e[e.arm == a].set_index(["seed", "task", "init"]).success
    y = e[e.arm == b].set_index(["seed", "task", "init"]).success
    common = x.index.intersection(y.index)
    oa, ob = int((x[common] & ~y[common]).sum()), int((~x[common] & y[common]).sum())
    return {"n_pairs": len(common), "k_a": int(x[common].sum()), "k_b": int(y[common].sum()), "only_a": oa,
            "only_b": ob, "p": mcnemar_exact(oa, ob)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(H / "results/round2/configs/fpo_v4.yaml"))
    train_init = {t["task_id"]: set(t["init_ids"]) for t in cfg["collect"]["tasks"]}
    ep = load()
    ep["train_init"] = [r.task in train_init and r.init in train_init[r.task] for r in ep.itertuples()]
    ep.to_csv(OUT / "episodes_all.csv", index=False)
    counts = ep.groupby(["arm", "seed"]).size().astype(int)
    res: dict = {"counts": {f"{a} s{s}": int(n) for (a, s), n in counts.items()}}
    L = ["# FPO v4: so sánh theo cặp với official và v3", "",
         "Tính lại từ file gốc `ep*.meta.json` (script `improve/scripts/fpo_v4_pooled.py`). official, v2: "
         "`outputs/fpo_v2/eval_s*`; v3: `outputs/fpo_v3/eval_s*`; v4: `outputs/fpo_v4/eval_s*/fpo_v4_sel`; không chạy lại "
         "arm cũ. Cặp = cùng (seed, task, init). McNemar chính xác hai phía. \"chỉ A / chỉ B\" = số cặp chỉ A thành công / "
         "chỉ B thành công. p theo tập con và theo task không hiệu chỉnh nhiều lần.", "",
         "Số episode theo arm và seed: " + ", ".join(f"{k}: {v}" for k, v in res["counts"].items()), ""]
    tr = lambda e: e.seed.isin(TRAIN_SEEDS) & e.task.isin(TRAINED)  # noqa: E731
    subsets = {"tổng (task 0 + 8)": tr, "task 0": lambda e: tr(e) & (e.task == 0),
               "task 8": lambda e: tr(e) & (e.task == 8), "held-out init": lambda e: tr(e) & ~e.train_init,
               "init đã train": lambda e: tr(e) & e.train_init}
    for t in TRAINED:
        for sp, lab in ((False, "held-out"), (True, "đã train")):
            subsets[f"task {t}, {lab}"] = (lambda t, sp: lambda e: tr(e) & (e.task == t) & (e.train_init == sp))(t, sp)
    for s in TRAIN_SEEDS:
        subsets[f"seed {s}"] = (lambda s: lambda e: tr(e) & (e.seed == s))(s)
    head = ("| tập | official | v2 θ4 | v3 θ4 | v4 | v4 vs official: chỉ v4 / chỉ off., p | v4 vs v3: chỉ v4 / chỉ v3, p |\n"
            "|---|---|---|---|---|---|---|")
    L += ["## Task đã train (0, 8) × 50 init × seed 3000/4000/5000", "", head]

    def block(name, m, key):
        e = ep[m(ep)]
        k = {a: (int(e[e.arm == a].success.sum()), int((e.arm == a).sum())) for a in ARMS}
        po, pv = paired(ep, "v4", "official", m), paired(ep, "v4", "v3", m)
        res.setdefault(key, {})[name] = {"counts": k, "v4_vs_official": po, "v4_vs_v3": pv}
        L.append(f"| {name} | {pct(*k['official'])} | {pct(*k['v2'])} | {pct(*k['v3'])} | {pct(*k['v4'])} | "
                 f"{po['only_a']}/{po['only_b']}, p = {po['p']:.3f} | {pv['only_a']}/{pv['only_b']}, p = {pv['p']:.3f} |")

    for name, m in subsets.items():
        block(name, m, "trained")
    ot = lambda e: (e.seed == OTHER_SEED) & e.task.isin(OTHERS)  # noqa: E731
    L += ["", "## Quên: 8 task không train × 20 init, seed 1000", "", head.replace("| tập |", "| task |")]
    for name, m in [("tổng 8 task", ot)] + [(f"task {t}", (lambda t: lambda e: ot(e) & (e.task == t))(t)) for t in OTHERS]:
        block(name, m, "forgetting")

    # pre-registered predictions
    L += ["", "## Đối chiếu với dự đoán ghi trước (docs/WORKLOG.md, tiêu chí cố định trước khi train)", ""]
    tr0, tr8 = res["trained"].get("task 0"), res["trained"].get("task 8")
    fg = res.get("forgetting", {}).get("tổng 8 task")
    pred = {}

    def done(r, arm, n):
        return r is not None and r["counts"][arm][1] == n

    if done(tr0, "v4", 150) and done(tr0, "v3", 150):
        k4, k3, p = tr0["counts"]["v4"][0], tr0["counts"]["v3"][0], tr0["v4_vs_official"]["p"]
        pred["a"] = (p > 0.05 and k4 > k3, f"task 0: v4 {k4}/150, v3 {k3}/150, official {tr0['counts']['official'][0]}/150; "
                                          f"v4 vs official p = {p:.3f}")
    if done(tr8, "v4", 150):
        k4, ko, p = tr8["counts"]["v4"][0], tr8["counts"]["official"][0], tr8["v4_vs_official"]["p"]
        pred["b"] = (p < 0.05 and k4 > ko, f"task 8: v4 {k4}/150, official {ko}/150; v4 vs official p = {p:.3f}")
    if done(fg, "v4", 160) and done(fg, "v3", 160):
        k4, k3 = fg["counts"]["v4"][0], fg["counts"]["v3"][0]
        pred["c"] = (k4 >= k3, f"quên: v4 {k4}/160, v3 {k3}/160 (official {fg['counts']['official'][0]}/160)")
    crit = {"a": "task 0: p(v4 vs official) > 0,05 và v4 > v3", "b": "task 8: p(v4 vs official) < 0,05 và v4 > official",
            "c": "quên: v4 ≥ v3 (tổng 8 task)"}
    L += ["| dự đoán | tiêu chí | số liệu | kết quả |", "|---|---|---|---|"]
    for k in "abc":
        if k in pred:
            ok, txt = pred[k]
            L.append(f"| ({k}) | {crit[k]} | {txt} | **{'ĐÚNG' if ok else 'SAI'}** |")
        else:
            L.append(f"| ({k}) | {crit[k]} | chưa đủ episode | chưa đối chiếu |")
    res["predictions"] = {k: {"correct": v[0], "data": v[1]} for k, v in pred.items()}
    (OUT / "pooled.md").write_text("\n".join(L) + "\n")
    json.dump(res, open(OUT / "pooled.json", "w"), indent=1, ensure_ascii=False)
    print("\n".join(L))


if __name__ == "__main__":
    main()
