"""FPO v3: task x init-split table (tasks 0, 8 separately; held-out vs training init), official / v2 / v3, exact
McNemar v3 vs official and v3 vs v2 per cell. Pooled over seeds 3000 / 4000 / 5000 and per seed. CPU only, reads the
raw ep*.meta.json through fpo_v3_pooled.load() (same arms, pairs = same (seed, task, init)).
Training init = collect.init_ids of results/round2/configs/fpo_v3.yaml (= v2).
Outputs: ~/ffi26/results/round2/fpo_v3/split_task_init.md (Vietnamese), split_task_init.csv.

    python scripts/fpo_v3_split.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from fpo_v3_pooled import ARMS, H, OUT, TRAIN_SEEDS, load, paired, pct  # noqa: E402


def main():
    cfg = yaml.safe_load(open(H / "results/round2/configs/fpo_v3.yaml"))
    train_init = {t["task_id"]: set(t["init_ids"]) for t in cfg["collect"]["tasks"]}
    ep = load()
    ep = ep[ep.seed.isin(TRAIN_SEEDS) & ep.task.isin(train_init)].copy()
    ep["split"] = ["train" if r.init in train_init[r.task] else "heldout" for r in ep.itertuples()]
    rows, L = [], ["# FPO v3: tách task × init (held-out / init đã train)", "",
                   "Script `improve/scripts/fpo_v3_split.py`; dữ liệu = file gốc `ep*.meta.json` như `pooled.md` "
                   "(official, v2: `outputs/fpo_v2/eval_s*`; v3: `outputs/fpo_v3/eval_s*/fpo_v3_sel`). Init đã train = "
                   "`collect.init_ids` trong `fpo_v3.yaml` (16 init/task, = v2), còn lại 34 init/task là held-out. "
                   "Cặp = cùng (seed, task, init). McNemar chính xác hai phía; \"chỉ A / chỉ B\" = số cặp chỉ A thành công "
                   "/ chỉ B thành công. 4 ô × 2 so sánh, không hiệu chỉnh nhiều lần.", ""]
    for scope, seeds in [("gộp 3 seed", TRAIN_SEEDS)] + [(f"seed {s}", (s,)) for s in TRAIN_SEEDS]:
        L += [f"## {scope}", "", "| task | init | official | v2 θ4 | v3 θ4 | v3 vs official: chỉ v3 / chỉ off., p | "
              "v3 vs v2: chỉ v3 / chỉ v2, p |", "|---|---|---|---|---|---|---|"]
        for t in sorted(train_init):
            for sp in ("heldout", "train"):
                m = lambda e, t=t, sp=sp, seeds=seeds: (e.task == t) & (e.split == sp) & e.seed.isin(seeds)  # noqa: E731
                e = ep[m(ep)]
                k = {a: (int(e[e.arm == a].success.sum()), int((e.arm == a).sum())) for a in ARMS}
                po, pv = paired(ep, "v3", "official", m), paired(ep, "v3", "v2", m)
                rows.append({"scope": scope, "task": t, "split": sp,
                             **{f"{a}_succ": k[a][0] for a in ARMS}, **{f"{a}_n": k[a][1] for a in ARMS},
                             "v3_vs_off_only_v3": po["only_a"], "v3_vs_off_only_off": po["only_b"], "v3_vs_off_p": po["p"],
                             "v3_vs_v2_only_v3": pv["only_a"], "v3_vs_v2_only_v2": pv["only_b"], "v3_vs_v2_p": pv["p"]})
                L.append(f"| {t} | {'held-out' if sp == 'heldout' else 'đã train'} | {pct(*k['official'])} | "
                         f"{pct(*k['v2'])} | {pct(*k['v3'])} | {po['only_a']}/{po['only_b']}, p = {po['p']:.3f} | "
                         f"{pv['only_a']}/{pv['only_b']}, p = {pv['p']:.3f} |")
        L.append("")
    pd.DataFrame(rows).to_csv(OUT / "split_task_init.csv", index=False)
    (OUT / "split_task_init.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
