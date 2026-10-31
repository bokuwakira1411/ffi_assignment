"""Phase 1 analysis: failure labels, subtask completion, reward sanity checks.

    python scripts/phase1_analyze.py --logged outputs/baseline/logged --out outputs/baseline \
        --baseline_dir ~/ffi26/results/runs/D_nas10/lerobot_smolvla_libero
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from failure_labels import label_episode  # noqa: E402
from reward import RewardConfig, episode_return  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase1_logging import load_episode  # noqa: E402


def md(df, floatfmt=".3f"):
    """Markdown table without the optional 'tabulate' dependency."""
    def f(v):
        return format(v, floatfmt) if isinstance(v, (float, np.floating)) else str(v)
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(f(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logged", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--baseline_dir", default=None, help="lerobot-eval runs task{t}_seed{s}/eval_info.json")
    ap.add_argument("--baseline_seeds", default="1000,1001,1002")
    args = ap.parse_args()
    logged, out = Path(args.logged).expanduser(), Path(args.out).expanduser()
    cfg = RewardConfig()
    cfg_full = RewardConfig(grasp_bonus=0.1, wrong_obj_cost=0.1, stall_cost=0.05)

    rows = []
    for npz in sorted(logged.glob("task*/ep*.npz")):
        log, actions, meta = load_episode(npz)
        task_id = int(npz.parent.name[4:])
        lab = label_episode(log, actions, meta["success"])
        rows.append({
            "task_id": task_id, "episode": int(npz.stem[2:]), "init_id": meta["init_id"], "seed": meta["seed"],
            "success": meta["success"], "length": meta["length"],
            "G_base": episode_return(log, cfg), "G_full": episode_return(log, cfg_full),
            **lab,
        })
    df = pd.DataFrame(rows).sort_values(["task_id", "episode"])
    df.to_csv(out / "episodes_labeled.csv", index=False)

    # ---- per-task summary
    summ = []
    for t, g in df.groupby("task_id"):
        k, n = int(g.success.sum()), len(g)
        lo, hi = wilson(k, n)
        fails = g[~g.success]
        rec = {"task_id": t, "n": n, "n_success": k, "success_rate": k / n, "ci95_lo": lo, "ci95_hi": hi,
               "subtask_completion_mean": g.subtask_completion.mean(),
               "subtask_completion_fail_mean": fails.subtask_completion.mean() if len(fails) else np.nan,
               "n_fail": len(fails)}
        for lab in ["A", "B", "C", "D", "E", "O"]:
            rec[f"primary_{lab}"] = int((fails.primary == lab).sum())
        for lab in ["A", "B", "C", "D", "E"]:
            rec[f"any_{lab}"] = int(fails[f"lab_{lab}"].sum())
        summ.append(rec)
    summ = pd.DataFrame(summ)
    summ.to_csv(out / "per_task_summary.csv", index=False)

    # ---- reward sanity checks
    lines = []
    # (i) does Phi separate failure types
    sep = df.groupby(["task_id", "primary"]).agg(n=("success", "size"), phi_T=("phi_T", "mean"),
                                                 phi_max=("phi_max", "mean"), G_base=("G_base", "mean"),
                                                 len_mean=("length", "mean")).reset_index()
    sep.to_csv(out / "reward_by_primary_label.csv", index=False)
    lines.append("## (i) Phi / return by primary label\n\n" + md(sep))
    # (ii) ranking
    rk = []
    for t, g in df.groupby("task_id"):
        s, f = g[g.success], g[~g.success]
        sep_ok = (s.G_base.min() > f.G_base.max()) if len(s) and len(f) else None
        # rank correlation between G and (success, phi_T)
        from scipy.stats import spearmanr
        rho_succ = spearmanr(g.G_base, g.success.astype(int)).correlation if g.success.nunique() > 1 else np.nan
        rho_phi = spearmanr(f.G_base, f.phi_T).correlation if len(f) > 2 and f.phi_T.nunique() > 1 else np.nan
        rk.append({"task_id": t, "all_success_G_above_all_failure_G": sep_ok,
                   "min_G_success": s.G_base.min() if len(s) else np.nan,
                   "max_G_failure": f.G_base.max() if len(f) else np.nan,
                   "spearman(G, success)": rho_succ, "spearman(G, phi_T) among failures": rho_phi,
                   "distinct G among failures": f.G_base.round(4).nunique(), "n_fail": len(f)})
    rk = pd.DataFrame(rk)
    lines.append("## (ii) Does G rank episodes sensibly?\n\n" + md(rk))
    # (iii) mixed-outcome groups from the 3-seed lerobot-eval baseline (same init states, different seeds)
    if args.baseline_dir:
        bd = Path(args.baseline_dir).expanduser()
        mix = []
        for t in sorted(df.task_id.unique()):
            S = []
            for s in args.baseline_seeds.split(","):
                p = bd / f"task{t}_seed{s}" / "eval_info.json"
                S.append(json.load(open(p))["per_task"][0]["metrics"]["successes"])
            per = np.array(S).sum(axis=0)
            n_seeds = len(S)
            mix.append({"task_id": t, "n_init_states": len(per), "n_seeds": n_seeds,
                        "all_fail": int((per == 0).sum()), "mixed": int(((per > 0) & (per < n_seeds)).sum()),
                        "all_success": int((per == n_seeds).sum()),
                        "mixed_init_ids": " ".join(str(i) for i in np.where((per > 0) & (per < n_seeds))[0])})
        mix = pd.DataFrame(mix)
        mix.to_csv(out / "mixed_groups_from_baseline.csv", index=False)
        lines.append("## (iii) Init states with mixed success over baseline seeds\n\n" + md(mix))
    (out / "reward_sanity.md").write_text("\n\n".join(lines) + "\n")
    print(summ.to_string())
    print("\n\n".join(lines))


if __name__ == "__main__":
    main()
