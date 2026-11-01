"""FPO v3: paired comparison v3 vs official and v3 vs v2, recomputed from the raw per-episode files (ep*.meta.json).

  * trained tasks 0, 8 x 50 init, seeds 3000 / 4000 / 5000 (outputs/fpo_v2/eval_s{seed}/{official,fpo_v2_sel},
    outputs/fpo_v3/eval_s{seed}/fpo_v3_sel): per seed and pooled; total, per task, held-out / training init
    (training init = collect.init_ids of fpo_v3.yaml, = v2);
  * forgetting, other 8 tasks x 20, seed 1000 (eval_s1000): total and per task.
Pairs = same (seed, task, init) (episode k: init state k, env seed seed + k). Exact two-sided McNemar on the
discordant pairs (binomial, p = 0.5). Missing arms / episodes are reported, never imputed.
Outputs: ~/ffi26/results/round2/fpo_v3/pooled.md (Vietnamese), pooled.json, episodes_all.csv.

    python scripts/fpo_v3_pooled.py
"""

from __future__ import annotations

import json
from math import comb
from pathlib import Path

import pandas as pd
import yaml

H = Path.home() / "ffi26"
ARMS = {"official": H / "improve/outputs/fpo_v2/eval_s{seed}/official",
        "v2": H / "improve/outputs/fpo_v2/eval_s{seed}/fpo_v2_sel",
        "v3": H / "improve/outputs/fpo_v3/eval_s{seed}/fpo_v3_sel"}
OUT = H / "results/round2/fpo_v3"
TRAIN_SEEDS, OTHER_SEED = (3000, 4000, 5000), 1000
TRAINED, OTHERS = (0, 8), (1, 2, 3, 4, 5, 6, 7, 9)


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def load() -> pd.DataFrame:
    rows = []
    for arm, root in ARMS.items():
        for seed, tasks in [(s, TRAINED) for s in TRAIN_SEEDS] + [(OTHER_SEED, OTHERS)]:
            for t in tasks:
                d = Path(str(root).format(seed=seed)) / f"task{t}"
                for f in sorted(d.glob("ep*.meta.json")):
                    m = json.load(open(f))
                    rows.append({"arm": arm, "seed": seed, "task": t, "init": int(f.name[2:5]),
                                 "success": bool(m["success"]), "length": int(m["length"])})
    return pd.DataFrame(rows)


def pct(k, n):
    return f"{k}/{n} = {100 * k / n:.1f}%" if n else "—"


def paired(ep: pd.DataFrame, a: str, b: str, mask) -> dict:
    e = ep[mask(ep)]
    x = e[e.arm == a].set_index(["seed", "task", "init"]).success
    y = e[e.arm == b].set_index(["seed", "task", "init"]).success
    common = x.index.intersection(y.index)
    only_a = int((x[common] & ~y[common]).sum())
    only_b = int((~x[common] & y[common]).sum())
    return {"n_pairs": len(common), "k_a": int(x[common].sum()), "k_b": int(y[common].sum()),
            "only_a": only_a, "only_b": only_b, "p": mcnemar_exact(only_a, only_b)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(H / "results/round2/configs/fpo_v3.yaml"))
    train_init = {t["task_id"]: set(t["init_ids"]) for t in cfg["collect"]["tasks"]}
    ep = load()
    ep["train_init"] = [r.task in train_init and r.init in train_init[r.task] for r in ep.itertuples()]
    ep.to_csv(OUT / "episodes_all.csv", index=False)
    res: dict = {"counts": ep.groupby(["arm", "seed"]).size().astype(int).reset_index().values.tolist()}
    L = ["# FPO v3: so sánh theo cặp với official và v2", "",
         "Tính lại từ file gốc `ep*.meta.json` (script `improve/scripts/fpo_v3_pooled.py`). official và v2: episode của "
         "`outputs/fpo_v2/eval_s*` (không chạy lại); v3: `outputs/fpo_v3/eval_s*/fpo_v3_sel`. Cặp = cùng (seed, task, "
         "init). McNemar chính xác hai phía. \"chỉ A / chỉ B\" = số cặp chỉ A thành công / chỉ B thành công.", "",
         "Số episode theo arm và seed: " + ", ".join(f"{a} s{s}: {n}" for a, s, n in res["counts"]), ""]

    tr = lambda e: e.seed.isin(TRAIN_SEEDS) & e.task.isin(TRAINED)  # noqa: E731
    subsets = {"tổng (task 0 + 8)": tr,
               "task 0": lambda e: tr(e) & (e.task == 0),
               "task 8": lambda e: tr(e) & (e.task == 8),
               "held-out init": lambda e: tr(e) & ~e.train_init,
               "init đã train": lambda e: tr(e) & e.train_init}
    for s in TRAIN_SEEDS:
        subsets[f"seed {s}"] = (lambda s: lambda e: tr(e) & (e.seed == s))(s)
    L += ["## Task đã train (0, 8) × 50 init × seed 3000/4000/5000, expert fp32", "",
          "| tập | official | v2 θ4 | v3 | v3 vs official: chỉ v3 / chỉ off., p | v3 vs v2: chỉ v3 / chỉ v2, p |",
          "|---|---|---|---|---|---|"]
    res["trained"] = {}
    for name, m in subsets.items():
        e = ep[m(ep)]
        k = {a: (int(e[e.arm == a].success.sum()), int((e.arm == a).sum())) for a in ARMS}
        po, pv = paired(ep, "v3", "official", m), paired(ep, "v3", "v2", m)
        res["trained"][name] = {"counts": k, "v3_vs_official": po, "v3_vs_v2": pv}
        L.append(f"| {name} | {pct(*k['official'])} | {pct(*k['v2'])} | {pct(*k['v3'])} | "
                 f"{po['only_a']}/{po['only_b']}, p = {po['p']:.3f} | {pv['only_a']}/{pv['only_b']}, p = {pv['p']:.3f} |")

    ot = lambda e: (e.seed == OTHER_SEED) & e.task.isin(OTHERS)  # noqa: E731
    L += ["", "## Quên: 8 task không train × 20 init, seed 1000, expert fp32", "",
          "| task | official | v2 θ4 | v3 | v3 vs official: chỉ v3 / chỉ off., p | v3 vs v2: chỉ v3 / chỉ v2, p |",
          "|---|---|---|---|---|---|"]
    res["forgetting"] = {}
    for name, m in [("tổng 8 task", ot)] + [(f"task {t}", (lambda t: lambda e: ot(e) & (e.task == t))(t))
                                            for t in OTHERS]:
        e = ep[m(ep)]
        k = {a: (int(e[e.arm == a].success.sum()), int((e.arm == a).sum())) for a in ARMS}
        po, pv = paired(ep, "v3", "official", m), paired(ep, "v3", "v2", m)
        res["forgetting"][name] = {"counts": k, "v3_vs_official": po, "v3_vs_v2": pv}
        L.append(f"| {name} | {pct(*k['official'])} | {pct(*k['v2'])} | {pct(*k['v3'])} | "
                 f"{po['only_a']}/{po['only_b']}, p = {po['p']:.3f} | {pv['only_a']}/{pv['only_b']}, p = {pv['p']:.3f} |")
    L += ["", "p theo từng task là so sánh nhiều lần (8 task), không hiệu chỉnh; chỉ dùng để mô tả."]
    (OUT / "pooled.md").write_text("\n".join(L) + "\n")
    json.dump(res, open(OUT / "pooled.json", "w"), indent=1, ensure_ascii=False)
    print("\n".join(L))


if __name__ == "__main__":
    main()
