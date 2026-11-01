"""Report figures (CPU only, no new runs). Every value is recomputed from raw experiment files; nothing is interpolated.

Figures (report/figures/*.pdf):
  fig_nas.pdf        Reproduce: libero_10 success per task for n_action_steps 1 / 10 / 50 (seed 1000, 50 init / task),
                     raw: ~/ffi26/results/runs/D_nas{1,10,50}/lerobot_smolvla_libero/task{k}_seed1000/eval_info.json.
  fig_round1.pdf     Improve, first screen (seed 1000, tasks 0+8, 50 init each, expert bf16 at eval as in round 1):
                     ~/ffi26/results/round2/episodes.csv (one row per episode). McNemar vs official on (task, init).
  fig_fpo_final.pdf  Improve, final protocol (expert fp32): tasks 0 / 8 x 50 init x seeds 3000/4000/5000 and forgetting
                     (8 other tasks x 20, seed 1000). Raw ep*.meta.json of improve/outputs/fpo_v2/eval_s*/{official,
                     fpo_v1_it3,fpo_v2_sel}, fpo_v3/eval_s*/fpo_v3_sel, fpo_v4/eval_s*/fpo_v4_sel.
  fig_training.pdf   FPO training dynamics: rolling-10 clip_frac in iteration 0 (same data for v2/v3/v4), gradient steps per
                     iteration (train_info.json), relative weight change vs theta0 (results/round2/fpo_v3/weight_delta.csv).
  fig_advantage.pdf  Advantage length bias, iteration 0 (shared by v2/v3/v4): segment adv vs number of decisions in the
                     segment (segments.csv + rollouts.csv + groups.csv), and the sign of A^ under v2 / v3 / v4
                     normalisation (results/round2/fpo_v4/sanity.csv).
Error bars: Wilson 95 %. McNemar: exact two-sided on discordant pairs.

    conda activate lerobot && python report/scripts/make_figures.py
"""

from __future__ import annotations

import json
from math import comb
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

H = Path.home() / "ffi26"
OUT = Path(__file__).resolve().parents[1] / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
                     "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
C = {"official": "#5b5b5b", "v1": "#9ecae1", "v2": "#4292c6", "v3": "#08519c", "v4": "#e6550d",
     "nas1": "#bdbdbd", "nas10": "#e6550d", "nas50": "#6baed6", "pos": "#31a354", "neg": "#de2d26"}
LOG = []  # every plotted number, written to figures/figure_data.json


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def bar_ci(ax, x, k, n, color, width, label=None, hatch=None):
    p = np.array(k) / np.array(n)
    lo, hi = zip(*[wilson(a, b) for a, b in zip(k, n)])
    ax.bar(x, 100 * p, width, color=color, label=label, edgecolor="white", linewidth=0.4, hatch=hatch)
    ax.errorbar(x, 100 * p, yerr=[100 * (p - np.array(lo)), 100 * (np.array(hi) - p)], fmt="none", ecolor="black",
                elinewidth=0.6, capsize=1.5)


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


# ------------------------------------------------------------------ 1. NAS ablation
def fig_nas():
    names = ["soup+sauce", "cheese+butter", "stove+moka", "bowl+drawer", "2 mugs/plates", "book→caddy",
             "mug+pudding", "soup+cheese", "2 moka pots", "mug→microwave"]
    res = {}
    for nas in (1, 10, 50):
        res[nas] = [json.load(open(H / f"results/runs/D_nas{nas}/lerobot_smolvla_libero/task{t}_seed1000/eval_info.json"))
                    ["per_task"][0]["n_success"] for t in range(10)]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.3), gridspec_kw={"width_ratios": [4.2, 1]})
    x = np.arange(10)
    for i, nas in enumerate((1, 10, 50)):
        bar_ci(a, x + (i - 1) * 0.27, res[nas], [50] * 10, C[f"nas{nas}"], 0.27, f"NAS = {nas}")
    a.set_xticks(x, [str(t) for t in range(10)])
    a.set_xlabel("task libero_10")
    a.set_ylabel("tỉ lệ thành công (%)")
    a.set_ylim(0, 105)
    a.legend(ncol=3, loc="upper left", frameon=False, bbox_to_anchor=(0, 1.16))
    tot = {nas: sum(v) for nas, v in res.items()}
    for i, nas in enumerate((1, 10, 50)):
        bar_ci(b, [i], [tot[nas]], [500], C[f"nas{nas}"], 0.7)
        b.text(i, 100 * tot[nas] / 500 + 5, f"{100 * tot[nas] / 500:.1f}", ha="center", fontsize=7)
    b.set_xticks(range(3), ["1", "10", "50"])
    b.set_xlabel("NAS")
    b.set_ylim(0, 105)
    b.set_title("tổng (500 ep)", fontsize=8)
    fig.savefig(OUT / "fig_nas.pdf")
    LOG.append({"figure": "fig_nas", "per_task_n50": res, "total_n500": tot})


# ------------------------------------------------------------------ 2. round-1 screen (seed 1000, bf16)
def fig_round1():
    ep = pd.read_csv(H / "results/round2/episodes.csv")
    tr = ep[ep.task.isin([0, 8])]
    order = [("official", "official"), ("a1_r1_pos", "A1 prompt +"), ("a1_r1_none", "A1 prompt ∅"),
             ("a1_r1_neg", "A1 prompt −"), ("a1emb_r1_pos", "A1-emb +"), ("a1emb_r1_none", "A1-emb ∅"),
             ("a1emb_r1_neg", "A1-emb −"), ("awr_r1", "AWR"), ("awr6k_r1_s2000", "AWR 6k@2k"),
             ("awr6k_r1_s4000", "AWR 6k@4k"), ("awr6k_r1_s6000", "AWR 6k@6k"), ("fpo_it1", "FPO v1 θ1"),
             ("fpo_it2", "FPO v1 θ2"), ("fpo_it3", "FPO v1 θ3"), ("fpo_it4", "FPO v1 θ4")]
    ref = tr[tr.model == "official"].set_index(["task", "init"]).success
    rows = []
    for m, lab in order:
        g = tr[tr.model == m].set_index(["task", "init"]).success
        common = g.index.intersection(ref.index)
        b, c = int((g[common] & ~ref[common]).sum()), int((~g[common] & ref[common]).sum())
        rows.append({"model": m, "label": lab, "k": int(g.sum()), "n": len(g), "only_model": b, "only_off": c,
                     "p": mcnemar(b, c)})
    df = pd.DataFrame(rows)
    fam = ["official"] + ["A1"] * 3 + ["A1e"] * 3 + ["AWR"] * 4 + ["FPO"] * 4
    col = {"official": C["official"], "A1": "#c7e9c0", "A1e": "#74c476", "AWR": "#fdae6b", "FPO": C["v2"]}
    fig, ax = plt.subplots(figsize=(7.0, 2.2))
    x = np.arange(len(df))
    for i, r in df.iterrows():
        bar_ci(ax, [i], [r.k], [r.n], col[fam[i]], 0.72)
        s = stars(r.p) if r.model != "official" else ""
        if s:
            ax.text(i, 100 * r.k / r.n + 12, s + ("↑" if r.only_model > r.only_off else "↓"), ha="center", fontsize=7)
    ax.axhline(100 * df.k[0] / df.n[0], color=C["official"], ls="--", lw=0.7)
    ax.set_xticks(x, df.label, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("task 0 + 8, thành công (%)")
    ax.set_ylim(0, 80)
    ax.set_title("Vòng sàng lọc: seed 1000, 100 ep (task 0, 8 × 50 init), expert bf16 khi eval; * McNemar p < 0,05 so với official",
                 loc="left", fontsize=7.5)
    fig.savefig(OUT / "fig_round1.pdf")
    LOG.append({"figure": "fig_round1", "rows": df.to_dict("records")})


# ------------------------------------------------------------------ 3. final FPO comparison (fp32, 3 seeds)
ARMS = {"official": "improve/outputs/fpo_v2/eval_s{s}/official", "v1": "improve/outputs/fpo_v2/eval_s{s}/fpo_v1_it3",
        "v2": "improve/outputs/fpo_v2/eval_s{s}/fpo_v2_sel", "v3": "improve/outputs/fpo_v3/eval_s{s}/fpo_v3_sel",
        "v4": "improve/outputs/fpo_v4/eval_s{s}/fpo_v4_sel"}
LAB = {"official": "official", "v1": "FPO v1 θ3", "v2": "FPO v2 θ4", "v3": "FPO v3 θ4", "v4": "FPO v4 θ4"}


def load_final():
    rows = []
    for arm, root in ARMS.items():
        for s, tasks in [(3000, (0, 8)), (4000, (0, 8)), (5000, (0, 8)), (1000, (1, 2, 3, 4, 5, 6, 7, 9))]:
            for t in tasks:
                for f in sorted((H / root.format(s=s) / f"task{t}").glob("ep*.meta.json")):
                    rows.append({"arm": arm, "seed": s, "task": t, "init": int(f.name[2:5]),
                                 "success": bool(json.load(open(f))["success"])})
    return pd.DataFrame(rows)


def fig_fpo_final():
    ep = load_final()
    tr = ep[ep.seed != 1000]
    fg = ep[ep.seed == 1000]
    arms = list(ARMS)
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={"width_ratios": [1.3, 2]})
    groups = [("task 0", tr[tr.task == 0]), ("task 8", tr[tr.task == 8]), ("0 + 8", tr),
              ("quên", fg)]
    w = 0.16
    log = {}
    for gi, (gname, g) in enumerate(groups):
        ref = g[g.arm == "official"].set_index(["seed", "task", "init"]).success
        for ai, arm in enumerate(arms):
            s = g[g.arm == arm].set_index(["seed", "task", "init"]).success
            if not len(s):
                continue
            xx = gi + (ai - 2) * w
            bar_ci(a, [xx], [int(s.sum())], [len(s)], C[arm], w, LAB[arm] if gi == 0 else None)
            common = s.index.intersection(ref.index)
            bb, cc = int((s[common] & ~ref[common]).sum()), int((~s[common] & ref[common]).sum())
            p = mcnemar(bb, cc)
            log[f"{gname}|{arm}"] = {"k": int(s.sum()), "n": len(s), "only_arm": bb, "only_official": cc, "p": p}
            if arm != "official" and stars(p):
                lo, hi = wilson(int(s.sum()), len(s))
                a.text(xx, 100 * hi + 1.5, stars(p).replace("*", "∗"), ha="center", va="bottom", fontsize=5.5,
                       rotation=90, color=C["pos"] if bb > cc else C["neg"])
    a.set_xticks(range(len(groups)), [g for g, _ in groups])
    a.set_ylabel("tỉ lệ thành công (%)")
    a.set_ylim(0, 100)
    a.set_title("(a) 150 ep / arm / task (3 seed); quên: 160 ep", loc="left", fontsize=7.5)
    tasks = (1, 2, 3, 4, 5, 6, 7, 9)
    for ai, arm in enumerate(arms):
        k = [int(fg[(fg.arm == arm) & (fg.task == t)].success.sum()) for t in tasks]
        b.bar(np.arange(8) + (ai - 2) * w, [100 * v / 20 for v in k], w, color=C[arm], edgecolor="white", linewidth=0.4)
        log[f"forget_per_task|{arm}"] = dict(zip(map(str, tasks), k))
    b.set_xticks(range(8), [f"task {t}" for t in tasks], fontsize=7)
    b.set_ylim(0, 105)
    b.set_title("(b) quên theo task (20 ep / task, seed 1000; không vẽ CI)", loc="left", fontsize=7.5)
    b.set_ylabel("tỉ lệ thành công (%)")
    h, l = a.get_legend_handles_labels()
    fig.legend(h, l, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.07), frameon=False, fontsize=7.5)
    fig.tight_layout(w_pad=1.5)
    fig.text(0.01, -0.04, "Sao: McNemar chính xác so với official (cặp cùng seed, task, init); xanh = tốt hơn, đỏ = kém hơn; "
                          "* p<0,05, ** p<0,01, *** p<0,001; không hiệu chỉnh nhiều phép so sánh.", fontsize=6.5)
    fig.savefig(OUT / "fig_fpo_final.pdf")
    LOG.append({"figure": "fig_fpo_final", "cells": log})


# ------------------------------------------------------------------ 4. training dynamics
def fig_training():
    O = H / "improve/outputs"
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.2, 2.1), gridspec_kw={"width_ratios": [1.4, 1, 1]})
    for r in ("v2", "v3", "v4"):
        lg = pd.read_csv(O / f"fpo_{r}/iter0/train/train_log.csv")
        a.plot(lg.step, lg.clip_frac.rolling(10, min_periods=1).mean(), color=C[r], lw=1.1, label=LAB[r].split(" θ")[0])
    a.axhline(0.2, color="black", ls=":", lw=0.7)
    a.text(2, 0.21, "ngưỡng early stop 0,2", fontsize=6.5)
    a.set_xlabel("bước gradient (iter 0, cùng dữ liệu)")
    a.set_ylabel("clip_frac (TB 10 bước)")
    a.legend(frameon=False, loc="lower right")
    a.set_title("(a) clip_frac", loc="left", fontsize=8)
    steps = {"v1": [json.load(open(O / f"fpo/iter{k}/train/train_info.json"))["steps"] for k in range(4)]}
    for r in ("v2", "v3", "v4"):
        steps[r] = [json.load(open(O / f"fpo_{r}/iter{k}/train/train_info.json"))["steps"] for k in range(4)]
    for i, r in enumerate(("v1", "v2", "v3", "v4")):
        b.bar(np.arange(4) + (i - 1.5) * 0.2, steps[r], 0.2, color=C[r], label=f"{r} ({sum(steps[r])})")
    b.set_xticks(range(4), [str(k) for k in range(4)])
    b.set_xlabel("iteration")
    b.set_ylabel("số bước")
    b.set_ylim(0, 200)
    b.legend(frameon=False, fontsize=6, ncol=2, loc="upper right", columnspacing=0.6, handlelength=1.0,
             title="tổng số bước", title_fontsize=6)
    b.set_title("(b) bước / iteration*", loc="left", fontsize=8)
    wd = pd.read_csv(H / "results/round2/fpo_v3/weight_delta.csv")
    for r in ("v2", "v3", "v4"):
        g = wd[wd.checkpoint.str.startswith(r)]
        c.plot(range(1, 5), 100 * g.rel_median, "o-", ms=3, color=C[r], lw=1.1, label=r)
    c.set_xticks(range(1, 5), [f"θ{k}" for k in range(1, 5)])
    c.set_ylabel("‖W−W₀‖/‖W₀‖, trung vị (%)")
    c.set_title("(c) thay đổi trọng số", loc="left", fontsize=8)
    c.legend(frameon=False)
    fig.text(0.38, -0.06, "* v1: batch 32; v2–v4: batch 128 (v3, v4: 64 / task).", fontsize=6.5)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(OUT / "fig_training.pdf")
    LOG.append({"figure": "fig_training", "steps": steps, "weight_rel_median": wd.set_index("checkpoint").rel_median.to_dict()})


# ------------------------------------------------------------------ 5. advantage length bias
def fig_advantage():
    it0 = H / "improve/outputs/fpo_v2/iter0"
    seg = pd.read_csv(it0 / "segments.csv")
    ro = pd.read_csv(it0 / "rollouts.csv")
    gr = pd.read_csv(it0 / "groups.csv")
    seg = seg.merge(gr[["group", "rejected"]], on="group")
    seg = seg[~seg.rejected.astype(bool)].merge(ro[["name", "success"]], on="name")
    # decisions in [start, end): decision steps are multiples of n_action_steps = 10
    seg["n_dec"] = (np.ceil(seg.end / 10) - np.ceil(seg.start / 10)).astype(int)
    seg = seg[seg.n_dec > 0]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.4), gridspec_kw={"width_ratios": [1.1, 1.4]})
    rng = np.random.default_rng(0)
    for succ, col, lab in ((True, C["pos"], "rollout thành công"), (False, C["neg"], "rollout thất bại")):
        g = seg[seg.success == succ]
        a.scatter(g.n_dec + rng.uniform(-0.3, 0.3, len(g)), g.adv, s=4, alpha=0.45, color=col, label=lab, lw=0)
    rcoef = float(np.corrcoef(seg.n_dec, seg.adv)[0, 1])
    a.axhline(0, color="black", lw=0.5)
    a.set_xlabel("số decision trong segment")
    a.set_ylabel("advantage A của segment")
    a.legend(frameon=False, loc="lower left", markerscale=2.5)
    a.set_title(f"(a) iter 0, task 0 + 8: corr = {rcoef:.2f}", loc="left", fontsize=8)
    sc = pd.read_csv(H / "results/round2/fpo_v4/sanity.csv")
    pick = {"v2 (A / std chung)": "v2", "v3 (decision, trừ mean_t)": "v3", "v4-segment (chạy)": "v4"}
    sc = sc[sc["cách"].isin(pick)]
    labels, w = [], 0.13
    xs = np.arange(2)
    for ti, t in enumerate((0, 8)):
        for vi, (name, v) in enumerate(pick.items()):
            r = sc[(sc.task == t) & (sc["cách"] == name)].iloc[0]
            tc = float(str(r["TC: Â < 0"]).rstrip("%"))
            tb = float(str(r["TB: Â > 0"]).rstrip("%"))
            off = ti * 3.5 + vi
            b.bar(off - w, tc, 2 * w, color=C[v], hatch="", edgecolor="white")
            b.bar(off + w, tb, 2 * w, color=C[v], hatch="////", edgecolor="white")
            b.text(off, max(tc, tb) + 2, f"Â seg {r['Â TB theo segment']:+.2f}", ha="center", fontsize=5.8)
            labels.append((off, f"{v}\ntask {t}"))
    b.set_xticks([o for o, _ in labels], [l for _, l in labels], fontsize=6.8)
    b.set_ylabel("% decision")
    from matplotlib.patches import Patch
    b.legend(handles=[Patch(facecolor="#999999", label="TC có Â < 0"), Patch(facecolor="#999999", hatch="////",
                                                                            label="TB có Â > 0")],
             frameon=False, loc="upper right")
    b.set_ylim(0, 65)
    b.set_title("(b) dấu của Â theo cách chuẩn hoá (iter 0; nhãn: Â TB theo segment)", loc="left", fontsize=8)
    fig.tight_layout(w_pad=1.5)
    fig.savefig(OUT / "fig_advantage.pdf")
    LOG.append({"figure": "fig_advantage", "corr_ndec_adv": rcoef, "n_segments": len(seg),
                "sanity_rows": sc.to_dict("records")})


if __name__ == "__main__":
    for f in (fig_nas, fig_round1, fig_fpo_final, fig_training, fig_advantage):
        f()
        print("ok", f.__name__)
    json.dump(LOG, open(OUT / "figure_data.json", "w"), indent=1, ensure_ascii=False, default=str)
