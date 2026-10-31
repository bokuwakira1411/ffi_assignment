"""Phase 3 analysis: success with 95% Wilson CI, subtask completion, failure types, paired test.

    python scripts/phase3_analyze.py --eval outputs/eval_r1 --out outputs \
        --ref_dir ~/ffi26/results/runs/D_nas10/lerobot_smolvla_libero
Writes <out>/results_per_model_task.csv, results_failure_types.csv, results_paired.csv,
results_episodes.csv, plots/success_ci.png, plots/failure_types.png.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

from failure_labels import label_episode  # noqa: E402
from phase1_analyze import wilson  # noqa: E402
from phase1_logging import load_episode  # noqa: E402

# reference palette (dataviz skill, references/palette.md), categorical slots in fixed order
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
LABELS = ["A", "B", "C", "D", "E", "O"]
LABEL_NAMES = {"A": "A grasp", "B": "B stuck after subtask", "C": "C wrong object", "D": "D undo progress",
               "E": "E stall", "O": "O other"}


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9c8c3")
    ax.tick_params(colors=INK2)
    ax.grid(axis="x", color="#ebeae6", linewidth=0.8)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref_dir", default=None, help="lerobot-eval baseline runs (task{t}_seed1000) for reference")
    args = ap.parse_args()
    ev, out = Path(args.eval).expanduser(), Path(args.out).expanduser()
    (out / "plots").mkdir(parents=True, exist_ok=True)

    rows = []
    for npz in sorted(ev.glob("*/task*/ep*.npz")):
        log, actions, meta = load_episode(npz)
        lab = label_episode(log, actions, meta["success"])
        rows.append({"model": npz.parents[1].name, "task_id": int(npz.parent.name[4:]), "episode": int(npz.stem[2:]),
                     "success": meta["success"], "length": meta["length"], **lab})
    df = pd.DataFrame(rows)
    df.to_csv(out / "results_episodes.csv", index=False)
    models = [m for m in ["baseline", "a1_r1", "a1_r2"] if m in df.model.unique()]

    summ = []
    for (m, t), g in df.groupby(["model", "task_id"]):
        k, n = int(g.success.sum()), len(g)
        lo, hi = wilson(k, n)
        rec = {"model": m, "task_id": t, "n": n, "n_success": k, "success_rate": k / n, "ci95_lo": lo, "ci95_hi": hi,
               "subtask_completion_mean": g.subtask_completion.mean(), "length_success_mean": g[g.success].length.mean()}
        if args.ref_dir:
            p = Path(args.ref_dir).expanduser() / f"task{t}_seed1000" / "eval_info.json"
            if p.exists():
                s = json.load(open(p))["per_task"][0]["metrics"]["successes"][:n]
                rec["ref_lerobot_eval_same_init_states"] = f"{sum(s)}/{len(s)}"
        summ.append(rec)
    summ = pd.DataFrame(summ)
    summ.to_csv(out / "results_per_model_task.csv", index=False)

    ft = (df[~df.success].groupby(["model", "task_id", "primary"]).size().unstack(fill_value=0)
          .reindex(columns=LABELS, fill_value=0).reset_index())
    ft.to_csv(out / "results_failure_types.csv", index=False)

    # paired comparison on the same (task, episode) = same init state + env seed: exact McNemar
    paired = []
    if "baseline" in models:
        for m in [x for x in models if x != "baseline"]:
            for t in sorted(set(df[df.model == m].task_id) & set(df[df.model == "baseline"].task_id)):
                a = df[(df.model == "baseline") & (df.task_id == t)].set_index("episode").success
                b = df[(df.model == m) & (df.task_id == t)].set_index("episode").success
                idx = a.index.intersection(b.index)
                a, b = a[idx], b[idx]
                n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
                p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
                paired.append({"model": m, "task_id": t, "n_pairs": len(idx), "baseline_success": int(a.sum()),
                               "model_success": int(b.sum()), "only_model_succeeds": n01,
                               "only_baseline_succeeds": n10, "mcnemar_exact_p": p})
    paired = pd.DataFrame(paired)
    paired.to_csv(out / "results_paired.csv", index=False)

    # ---- plot 1: success rate with 95% Wilson CI (dot + interval), one row per task
    tasks = sorted(df.task_id.unique())
    fig, ax = plt.subplots(figsize=(7.5, 0.9 + 0.75 * len(tasks) * len(models) / 2 + 0.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    ylabels, y = [], 0
    for t in tasks:
        for i, m in enumerate(models):
            r = summ[(summ.model == m) & (summ.task_id == t)]
            if r.empty:
                continue
            r = r.iloc[0]
            ax.plot([r.ci95_lo * 100, r.ci95_hi * 100], [y, y], color=SERIES[i], linewidth=2, solid_capstyle="round")
            ax.plot(r.success_rate * 100, y, "o", color=SERIES[i], markersize=8, markeredgecolor=SURFACE,
                    markeredgewidth=2)
            ax.text(r.ci95_hi * 100 + 1.5, y, f"{r.n_success}/{r.n} = {r.success_rate*100:.0f}%", va="center",
                    fontsize=8, color=INK)
            ylabels.append((y, f"task {t} · {m}"))
            y += 1
        y += 0.6
    ax.set_yticks([p for p, _ in ylabels], [s for _, s in ylabels], fontsize=8, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 115)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xlabel("success rate (%) · dot = estimate, bar = 95% Wilson CI", fontsize=8, color=INK2)
    ax.set_title("LIBERO-10 success by model and task (same init states and seeds)", fontsize=10, color=INK, loc="left")
    handles = [plt.Line2D([], [], marker="o", color=SERIES[i], linewidth=2, label=m) for i, m in enumerate(models)]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "plots/success_ci.png", facecolor=SURFACE)
    plt.close(fig)

    # ---- plot 2: failure types, stacked horizontal bars of failure counts
    tgt = ft[ft.task_id.isin(paired.task_id.unique() if len(paired) else tasks)]
    fig, ax = plt.subplots(figsize=(7.5, 0.9 + 0.45 * len(tgt) + 0.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    order = [(t, m) for t in sorted(tgt.task_id.unique()) for m in models if ((tgt.task_id == t) & (tgt.model == m)).any()]
    for yi, (t, m) in enumerate(order):
        r = tgt[(tgt.task_id == t) & (tgt.model == m)].iloc[0]
        left = 0
        for j, lab in enumerate(LABELS):
            v = int(r[lab])
            if v == 0:
                continue
            ax.barh(yi, v, left=left, height=0.6, color=SERIES[j], edgecolor=SURFACE, linewidth=2)
            if v >= 2:
                ax.text(left + v / 2, yi, str(v), ha="center", va="center", fontsize=7, color=INK)
            left += v
        ax.text(left + 0.4, yi, f"{left} failures", va="center", fontsize=8, color=INK2)
    ax.set_yticks(range(len(order)), [f"task {t} · {m}" for t, m in order], fontsize=8, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("failed episodes (primary label)", fontsize=8, color=INK2)
    ax.set_title("Failure types on target tasks", fontsize=10, color=INK, loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=SERIES[j], label=LABEL_NAMES[lab]) for j, lab in enumerate(LABELS)]
    ax.legend(handles=handles, frameon=False, fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.25))
    xmax = max([1] + [int(tgt[LABELS].iloc[i].sum()) for i in range(len(tgt))])
    ax.set_xlim(0, xmax * 1.25)
    fig.tight_layout()
    fig.savefig(out / "plots/failure_types.png", facecolor=SURFACE)
    plt.close(fig)

    print(summ.to_string())
    print(ft.to_string())
    print(paired.to_string())


if __name__ == "__main__":
    main()
