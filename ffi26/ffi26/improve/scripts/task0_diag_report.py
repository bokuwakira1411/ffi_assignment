"""Tables of results/round2/task0_diag.md from outputs/task0_diag/*.csv (task0_diag_labels.py, task0_diag_grad.py).
Writes outputs/task0_diag/tables.md (pasted into the report) and outputs/task0_diag/<grad file>_summary.csv.

    python scripts/task0_diag_report.py          (from ~/ffi26/improve)
"""

from pathlib import Path

import pandas as pd

D = Path(__file__).resolve().parents[1] / "outputs/task0_diag"


def md(d: pd.DataFrame) -> str:
    cols = list(d.columns)
    return "\n".join(["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
                     + ["| " + " | ".join(str(v) for v in r) + " |" for r in d.itertuples(index=False)])


def nz(d: pd.DataFrame) -> pd.DataFrame:  # drop all-zero label columns
    return d.loc[:, [c for c in d.columns if not (c.startswith(("chính ", "có ")) and (d[c] == 0).all())]]


def main():
    L = []
    for name, title in [("A1_task0", "A1. Task 0, eval seed 3000/4000/5000 (150 episode/arm)"),
                        ("A2_task8", "A2a. Task 8, eval seed 3000/4000/5000 (150 episode/arm)"),
                        ("A2_task4", "A2b. Task 4, eval seed 1000 (20 episode/arm)"),
                        ("A2_forget", "A2c. Forgetting: 8 task × 20 episode, seed 1000"),
                        ("A3_collect_task0", "A3a. Collection v3, task 0 (128 rollout/iteration; iter 4 = θ4)"),
                        ("A3_collect_task8", "A3b. Collection v3, task 8 (128 rollout/iteration)")]:
        f = D / f"{name}.csv"
        if f.exists():
            L += [f"### {title}", "", f"Nguồn: `improve/outputs/task0_diag/{name}.csv`", "", md(nz(pd.read_csv(f))), ""]
    for gname, gtitle in [("grad_cos", "CPU, n = 32/nửa, DỪNG giữa chừng (làm chậm v4)"),
                          ("grad_cos_gpu_n32", "GPU, n = 32/nửa (thiết kế gốc)"),
                          ("grad_cos_gpu_n256", "GPU, n = 256/nửa")]:
        g = D / f"{gname}.csv"
        if not g.exists():
            continue
        df = pd.read_csv(g)
        cols = ["cos_0a_0b", "cos_8a_8b", "cos_0a_8a", "cos_0b_8b", "cos_0a_8b", "cos_0b_8a", "cos_0_8",
                "cos_0_demo", "cos_8_demo", "norm_0", "norm_8", "norm_0a", "norm_0b", "norm_8a", "norm_8b", "norm_demo"]
        df["cos_cross_half"] = df[["cos_0a_8a", "cos_0b_8b", "cos_0a_8b", "cos_0b_8a"]].mean(axis=1)
        cols.insert(6, "cos_cross_half")
        s = df.groupby("iter")[cols].agg(["mean", "std"])
        s.columns = [f"{a}_{b}" for a, b in s.columns]
        s.insert(0, "n_rep", df.groupby("iter").size())
        s.to_csv(D / f"{gname}_summary.csv")
        pm = lambda c: [f"{m:+.3f} ± {sd:.3f}" for m, sd in zip(s[f"{c}_mean"], s[f"{c}_std"])]  # noqa: E731
        pn = lambda c: [f"{m:.2f} ± {sd:.2f}" for m, sd in zip(s[f"{c}_mean"], s[f"{c}_std"])]  # noqa: E731
        t = pd.DataFrame({"θ_k (iter k)": [f"θ{k}" for k in s.index], "n chia": s.n_rep.tolist(),
                          "đối chứng cos(g0a, g0b)": pm("cos_0a_0b"), "đối chứng cos(g8a, g8b)": pm("cos_8a_8b"),
                          "cos(g0 nửa, g8 nửa)": pm("cos_cross_half"), "cos(g0, g8)": pm("cos_0_8"),
                          "cos(g0, g_demo)": pm("cos_0_demo"), "cos(g8, g_demo)": pm("cos_8_demo"),
                          "‖g0‖": pn("norm_0"), "‖g8‖": pn("norm_8"), "‖g_demo‖": pn("norm_demo")})
        L += [f"### B. Cosine gradient, {gtitle} (mean ± std qua các cách chia)", "",
              f"Nguồn: `improve/outputs/task0_diag/{gname}.csv` (từng lần chia), `{gname}_summary.csv`", "", md(t), ""]
    (D / "tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
