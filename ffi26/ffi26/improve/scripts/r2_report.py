"""Round-2 study report: success (Wilson 95% CI), held-out split, paired McNemar vs the official
checkpoint, forgetting on the other 8 tasks, failure types, and the positive-vs-negative gap check.
Everything is recomputed from the per-episode eval files (<dir>/task{t}/ep{k}.npz|.meta.json).

    python scripts/r2_report.py --config ~/ffi26/results/round2/configs/report.yaml
Config: models: [{name, label, dirs: {<task>: <dir containing ep*.npz>} or root: <dir with task*/>}],
        trained_tasks, other_tasks, heldout_from: <round-1 collect config>, reference: <model name>,
        gap_pairs: [[pos_name, neg_name], ...], out_dir.
Outputs (out_dir): episodes.csv, per_model_task.csv, summary.csv, failure_types.csv, summary.md (Vietnamese).
"""

import argparse
import json
import sys
from math import comb, sqrt
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"), str(Path(__file__).resolve().parent)]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from failure_labels import label_episode  # noqa: E402
from phase1_logging import load_episode  # noqa: E402
from reward_r2 import detect_events  # noqa: E402

PRIMARY_VI = {"A": "A gắp hỏng", "B": "B kẹt chuyển subtask", "C": "C nhầm vật", "D": "D làm hỏng tiến độ",
              "E": "E đứng yên", "O": "O khác"}


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def load_dir(d: Path, task: int, model: str) -> list[dict]:
    rows = []
    for npz in sorted(d.glob("ep*.npz")):
        log, actions, meta = load_episode(npz)
        succ = bool(meta["success"])
        lab = label_episode(log, actions, succ)
        ev = pd.Series([e["type"] for e in detect_events(log, actions, succ)], dtype=object).value_counts()
        rows.append({"model": model, "task": task, "init": int(npz.stem[2:]), "success": succ,
                     "primary": lab["primary"], "subtask_completion": lab["subtask_completion"],
                     **{f"n_{x}": int(ev.get(x, 0)) for x in ("drop", "slip", "collision")}})
    return rows


def fmt(k, n):
    if n == 0:
        return "—"
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {100*k/n:.0f}% ({lo:.0f}–{hi:.0f})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    out = Path(cfg["out_dir"]).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    trained, others = cfg["trained_tasks"], cfg["other_tasks"]
    col = yaml.safe_load(open(Path(cfg["heldout_from"]).expanduser()))
    used = {t["task_id"]: set(t["init_ids"]) for t in col["tasks"]}
    heldout = {t: set(range(50)) - used.get(t, set()) for t in trained}

    rows = []
    for m in cfg["models"]:
        for t in trained + others:
            d = Path(m["dirs"][t]).expanduser() if "dirs" in m and t in m["dirs"] else \
                Path(m.get("root", "/nonexistent")).expanduser() / f"task{t}"
            if d.exists():
                rows += load_dir(d, t, m["name"])
    ep = pd.DataFrame(rows)
    ep["heldout"] = [r.task in trained and r.init in heldout[r.task] for r in ep.itertuples()]
    ep.to_csv(out / "episodes.csv", index=False)

    per = ep.groupby(["model", "task"]).agg(n=("success", "size"), k=("success", "sum"),
                                             subtask=("subtask_completion", "mean")).reset_index()
    per.to_csv(out / "per_model_task.csv", index=False)

    ref = cfg["reference"]
    labels = {m["name"]: m.get("label", m["name"]) for m in cfg["models"]}
    summ = []
    for m in cfg["models"]:
        e = ep[ep.model == m["name"]]
        if e.empty:
            continue
        tr, ho, ot = e[e.task.isin(trained)], e[e.task.isin(trained) & e.heldout], e[e.task.isin(others)]
        row = {"model": m["name"], "label": labels[m["name"]],
               "trained_k": int(tr.success.sum()), "trained_n": len(tr),
               "heldout_k": int(ho.success.sum()), "heldout_n": len(ho),
               "other_k": int(ot.success.sum()), "other_n": len(ot)}
        for t in trained:
            et = tr[tr.task == t]
            row[f"task{t}_k"], row[f"task{t}_n"] = int(et.success.sum()), len(et)
        # paired vs reference on trained tasks (same init state + env seed)
        r = ep[(ep.model == ref) & ep.task.isin(trained)].set_index(["task", "init"]).success
        x = tr.set_index(["task", "init"]).success
        common = x.index.intersection(r.index)
        b = int((x[common] & ~r[common]).sum())
        c = int((~x[common] & r[common]).sum())
        row.update({"only_model": b, "only_ref": c, "mcnemar_p": mcnemar_exact(b, c) if m["name"] != ref else np.nan})
        ro = ep[(ep.model == ref) & ep.task.isin(others)].set_index(["task", "init"]).success
        xo = ot.set_index(["task", "init"]).success
        co = xo.index.intersection(ro.index)
        row["other_ref_k_same_eps"] = int(ro[co].sum())
        summ.append(row)
    summ = pd.DataFrame(summ)
    summ.to_csv(out / "summary.csv", index=False)

    ft = (ep[~ep.success & ep.task.isin(trained)].groupby(["model", "task", "primary"]).size()
          .unstack("primary", fill_value=0))
    ev = ep[ep.task.isin(trained)].groupby(["model", "task"])[["n_drop", "n_slip", "n_collision"]].sum()
    ft = ft.join(ev, how="outer").fillna(0).astype(int)
    ft.to_csv(out / "failure_types.csv")

    # markdown (Vietnamese)
    L = ["# Tóm tắt kết quả round 2 (RL có điều kiện theo advantage)", "",
         "Nguồn: tính lại từ file từng episode (`episodes.csv`); CI = Wilson 95%. "
         f"Held-out = init state không dùng để thu rollout ({len(next(iter(heldout.values())))} / task). "
         f"McNemar chính xác so với `{ref}` trên cùng init state + seed env (task đã train).", "",
         "| model | task " + " | task ".join(map(str, trained)) + " | task đã train (tất cả) | held-out | 8 task khác "
         f"| chỉ model / chỉ {ref} | McNemar p |",
         "|---|" + "---|" * (len(trained) + 5)]
    for r in summ.itertuples():
        cells = [fmt(getattr(r, f"task{t}_k"), getattr(r, f"task{t}_n")) for t in trained]
        p = "—" if np.isnan(r.mcnemar_p) else f"{r.mcnemar_p:.2f}"
        L.append(f"| {r.label} | " + " | ".join(cells) + f" | {fmt(r.trained_k, r.trained_n)} | {fmt(r.heldout_k, r.heldout_n)}"
                 f" | {fmt(r.other_k, r.other_n)} | {r.only_model} / {r.only_ref} | {p} |")
    L += ["", "## Kiểm tra khoảng cách positive vs negative", ""]
    for pos, neg in cfg.get("gap_pairs", []):
        a, b_ = summ.set_index("model").loc[pos], summ.set_index("model").loc[neg]
        gap = 100 * (a.trained_k / a.trained_n - b_.trained_k / b_.trained_n)
        x = ep[(ep.model == pos) & ep.task.isin(trained)].set_index(["task", "init"]).success
        y = ep[(ep.model == neg) & ep.task.isin(trained)].set_index(["task", "init"]).success
        ci = x.index.intersection(y.index)
        bb, cc = int((x[ci] & ~y[ci]).sum()), int((~x[ci] & y[ci]).sum())
        flag = " **⚠ khoảng cách ≈ 0: VLM đóng băng có thể không truyền nhãn sang action expert**" if abs(gap) < 5 else ""
        L.append(f"- {labels[pos]} − {labels[neg]}: {gap:+.1f} điểm % (cặp: {bb} chỉ positive / {cc} chỉ negative, "
                 f"McNemar p {mcnemar_exact(bb, cc):.2f}).{flag}")
    L += ["", "## Kiểu thất bại trên task đã train (nhãn chính, chỉ episode thất bại) và số sự kiện", "",
          "Cột A–O: số episode thất bại theo nhãn chính (`src/failure_labels.py`); n_drop/n_slip/n_collision: số sự kiện "
          "(`src/reward_r2.py`) trên mọi episode.", ""]
    L.append(ft.reset_index().rename(columns=PRIMARY_VI).to_markdown(index=False) if hasattr(ft, "to_markdown") and _has_tabulate()
             else _md(ft.reset_index().rename(columns=PRIMARY_VI)))
    (out / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def _has_tabulate():
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
