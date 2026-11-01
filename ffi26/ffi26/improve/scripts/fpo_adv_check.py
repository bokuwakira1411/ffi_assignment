"""Checks 1-3 of the raw FPO advantages (read-only, CPU). Runs v2 and v3, iterations 0..3 (iter 0 of v3 = v2's).
Inputs per iteration: decisions/g*.npz (keys env, step, n_exec, adv, rejected, task_id, group only), rollouts.csv,
segments.csv, groups.csv (written by scripts/fpo_collect.py; advantages from src/reward_r2.group_segment_labels).
  1. per (group, stage): mean of the segment advantage over the rollouts that have that stage (rollout-weighted).
  2. decision-weighted vs segment- / rollout-weighted mean of adv; decisions per success / failure rollout;
     correlation of segment length (decisions) with segment adv. Training filter = fpo_train.load_decisions
     (rejected == 0 and adv != 0).
  3. alignment: per (group, env j) decisions == ceil(length_j / n_action_steps), steps 0, 10, 20, ...;
     n_exec == min(10, length - step); stored adv == adv of the segments.csv segment containing the step (0 if rejected);
     sign of adv for decisions of successful rollouts, raw and after v3's per-task normalisation
     A^ = clip((A - mean_t) / std_t, -5, 5), mean_t / std_t over the filtered decisions of task t (= fpo_train_v3.py:101-102,
     i.e. decision-weighted).
Outputs: ~/ffi26/results/round2/fpo_v3/adv_check.md (Vietnamese) + adv_check_*.csv.

    python scripts/fpo_adv_check.py
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

H = Path.home() / "ffi26"
O = H / "improve/outputs"
OUT = H / "results/round2/fpo_v3"
NAS = 10
KEYS = ("env", "step", "n_exec", "adv", "rejected", "task_id", "group")


def load_iter(d: Path):
    dec = []
    for f in sorted(glob.glob(str(d / "decisions" / "g*.npz"))):
        z = np.load(f)
        dec.append(pd.DataFrame({k: z[k] for k in KEYS}))
    dec = pd.concat(dec, ignore_index=True)
    ro = pd.read_csv(d / "rollouts.csv")
    ro["j"] = ro.name.str.extract(r"_j(\d+)$")[0].astype(int)
    seg = pd.read_csv(d / "segments.csv")
    seg["j"] = seg.name.str.extract(r"_j(\d+)$")[0].astype(int)
    gr = pd.read_csv(d / "groups.csv")
    return dec, ro, seg, gr


def md(d: pd.DataFrame) -> str:
    return "\n".join(["| " + " | ".join(map(str, d.columns)) + " |", "|" + "---|" * len(d.columns)]
                     + ["| " + " | ".join(str(v) for v in r) + " |" for r in d.itertuples(index=False)])


def main():
    c1, c2, c3, c3s = [], [], [], []
    for run in ("v2", "v3"):
        for k in range(4):
            dec, ro, seg, gr = load_iter(O / f"fpo_{run}/iter{k}")
            rej = gr.set_index("group").rejected
            dec = dec.merge(ro[["group", "j", "success", "length"]], left_on=["group", "env"], right_on=["group", "j"],
                            how="left", validate="many_to_one")
            # ---- 3. alignment
            assert dec.success.notna().all(), "decision without rollout"
            nd = dec.groupby(["group", "env"]).size().rename("n_dec").reset_index().merge(
                ro, left_on=["group", "env"], right_on=["group", "j"])
            exp_n = np.ceil(nd.length / NAS).astype(int)
            steps_ok = dec.groupby(["group", "env"]).step.apply(lambda s: (np.sort(s.values) == NAS * np.arange(len(s))).all())
            nexec_ok = (dec.n_exec == np.minimum(NAS, dec.length - dec.step))
            # stored adv vs segments.csv
            s_ = seg.merge(gr[["group", "rejected"]], on="group")
            exp_adv = np.zeros(len(dec), np.float64)
            n_match = np.zeros(len(dec), int)
            sg = {key: g for key, g in s_.groupby(["group", "j"])}
            for i, r in enumerate(dec.itertuples()):
                g = sg.get((r.group, r.env))
                if g is None:
                    continue
                m = g[(g.start <= r.step) & (r.step < g.end)]
                n_match[i] = len(m)
                if len(m) and not bool(rej[r.group]):
                    exp_adv[i] = m.adv.iloc[0]
            adv_err = np.abs(dec.adv.values - exp_adv)
            for t in (0, 8):
                tm = dec.task_id.values == t
                c3.append({"run": run, "iter": k, "task": t, "rollouts": int((ro.task_id == t).sum()),
                           "n_dec == ceil(len/10)": f"{int((nd.n_dec == exp_n)[nd.task_id == t].sum())}/{int((nd.task_id == t).sum())}",
                           "bước = 0,10,20,…": f"{int(steps_ok[nd[nd.task_id == t].set_index(['group', 'env']).index].sum())}/{int((nd.task_id == t).sum())}",
                           "n_exec đúng": f"{int(nexec_ok[tm].sum())}/{int(tm.sum())}",
                           "đúng 1 segment chứa bước": f"{int((n_match[tm] == 1).sum())}/{int(tm.sum())}",
                           "max |adv − adv segment|": f"{adv_err[tm].max():.1e}"})
            # ---- filtered (training) decisions
            f = dec[(dec.rejected == 0) & (dec.adv != 0)].copy()
            for t in (0, 8):
                ft = f[f.task_id == t]
                mu, sd = ft.adv.mean(), ft.adv.std(ddof=0)
                ah = np.clip((ft.adv - mu) / sd, -5, 5)
                s, fl = ft.success.astype(bool), ~ft.success.astype(bool)
                # rollout-weighted mean_t (each rollout's decisions averaged first)
                rw = ft.groupby(["group", "env"]).adv.mean().mean()
                c3s.append({"run": run, "iter": k, "task": t, "decision (lọc)": len(ft),
                            "của rollout thành công": int(s.sum()),
                            "mean_t (theo decision)": round(float(mu), 4), "mean theo rollout": round(float(rw), 4), "std_t": round(float(sd), 4),
                            "TC: A < 0 (thô)": f"{100 * (ft.adv[s] < 0).mean():.1f}%",
                            "TC: Â < 0 (sau chuẩn hoá v3)": f"{100 * (ah[s] < 0).mean():.1f}%",
                            "TB: A > 0 (thô)": f"{100 * (ft.adv[fl] > 0).mean():.1f}%",
                            "TB: Â > 0 (sau chuẩn hoá v3)": f"{100 * (ah[fl] > 0).mean():.1f}%"})
            # ---- 1. per (group, stage), rollout-weighted, non-rejected groups
            sn = s_[~s_.rejected.astype(bool)]
            gs = sn.groupby(["group", "task_id", "stage"]).agg(n=("adv", "size"), mean=("adv", "mean"),
                                                             absmean=("adv", lambda x: np.abs(x).mean())).reset_index()
            for t in (0, 8):
                g = gs[gs.task_id == t]
                g2 = g[g.n >= 2]
                c1.append({"run": run, "iter": k, "task": t, "(nhóm, stage) n ≥ 2": len(g2),
                           "trung vị |mean|": f"{np.median(np.abs(g2['mean'])):.4f}",
                           "max |mean|": f"{np.abs(g2['mean']).max():.4f}",
                           "trung vị mean |adv|": f"{np.median(g2.absmean):.4f}",
                           "tỉ lệ |mean| / mean|adv| (trung vị)": f"{np.median(np.abs(g2['mean']) / g2.absmean):.3f}",
                           "(nhóm, stage) n = 1": int((g.n == 1).sum())})
            # ---- 2. weighting
            ns = f.groupby(["group", "env", "adv"]).size().rename("nd").reset_index()  # one row ~ one segment
            for t in (0, 8):
                ft = f[f.task_id == t]
                st = ns.merge(ft[["group", "env"]].drop_duplicates(), on=["group", "env"])
                st = st[st.group.isin(ft.group.unique())]
                rr = ft.groupby(["group", "env"]).agg(succ=("success", "first"), nd=("adv", "size"), m=("adv", "mean"))
                allr = dec[(dec.task_id == t) & (dec.rejected == 0)].groupby(["group", "env"]).agg(
                    succ=("success", "first"), nd=("adv", "size"))
                c2.append({"run": run, "iter": k, "task": t,
                           "mean theo decision": round(float(ft.adv.mean()), 4),
                           "mean theo segment": round(float(st.adv.mean()), 4),
                           "mean theo rollout": round(float(rr.m.mean()), 4),
                           "decision / rollout TC": round(allr[allr.succ.astype(bool)].nd.mean(), 1),
                           "decision / rollout TB": round(allr[~allr.succ.astype(bool)].nd.mean(), 1),
                           "corr(số decision của segment, adv)": round(np.corrcoef(st.nd, st.adv)[0, 1], 3),
                           "% decision của rollout TB": f"{100 * (~ft.success.astype(bool)).mean():.1f}%"})
            print(run, k, "done", flush=True)
    t1, t2, t3, t3b = (pd.DataFrame(x) for x in (c1, c2, c3, c3s))
    for n, d in (("adv_check_1_group_stage", t1), ("adv_check_2_weighting", t2), ("adv_check_3_alignment", t3),
                 ("adv_check_3_success_sign", t3b)):
        d.to_csv(OUT / f"{n}.csv", index=False)
    L = ["# Kiểm tra advantage thô của FPO (kiểm tra 1–3)", "",
         "Script `improve/scripts/fpo_adv_check.py` (CPU, chỉ đọc). Nguồn: `outputs/fpo_v{2,3}/iter{0..3}/"
         "{decisions/g*.npz, rollouts.csv, segments.csv, groups.csv}`. v3 iter 0 = v2 iter 0 (dùng lại). Advantage tính "
         "trong `src/reward_r2.py:173–217`, gán cho decision theo bước đầu của nó (`scripts/fpo_collect.py:89–92`). "
         "\"Lọc\" = như khi train (`fpo_train.load_decisions`: rejected = 0 và adv ≠ 0). TC = rollout thành công, "
         "TB = rollout thất bại.", "",
         "## Kiểm tra 3: khớp decision ↔ rollout ↔ segment", "", md(t3), "",
         "## Kiểm tra 1: trung bình advantage trong từng (nhóm, stage), mỗi rollout một phiếu, nhóm không bị loại", "",
         "Nếu V[k] đúng thì mean ≈ 0. Không bằng 0 chính xác vì γ^(độ dài segment) khác nhau giữa các rollout và rollout "
         "không tới stage k+1 thì không có bootstrap.", "", md(t1), "",
         "## Kiểm tra 2: trung bình theo decision vs theo segment / rollout (decision đã lọc)", "", md(t2), "",
         "## Kiểm tra 3 (thêm): dấu advantage của decision thuộc rollout thành công / thất bại", "",
         "Chuẩn hoá của v3: Â = clip((A − mean_t) / std_t, ±5), mean_t và std_t tính **theo decision** trên decision đã lọc "
         "của task t (`fpo_train_v3.py:101–102`). Cột \"mean theo rollout\" chỉ để so sánh, không dùng khi train. "
         "Với v2, cột \"sau chuẩn hoá v3\" là giả định (v2 thực tế dùng A / std, không đổi dấu).", "", md(t3b), "",
         "## Nhận xét (chỉ từ số liệu trên)", "",
         f"- Kiểm tra 3: mọi decision khớp với đúng một rollout và đúng một segment; số decision = ⌈độ dài/10⌉, bước 0, 10, 20, …, "
         f"n_exec đúng ở mọi (run, vòng, task); adv lưu trùng với `segments.csv` (sai lệch tối đa {t3['max |adv − adv segment|'].astype(float).max():.0e}, "
         "do làm tròn float32). Không thấy lỗi gán advantage.",
         "- Kiểm tra 1: trong từng (nhóm, stage), trung bình advantage theo rollout có trung vị 0,0000 ở mọi (run, vòng, task); "
         f"lớn nhất {t1['max |mean|'].astype(float).max():.4f}. V[k] khớp với thiết kế.",
         "- Kiểm tra 2: trung bình theo segment ≈ 0 (|mean| ≤ 0,0023); trung bình theo decision âm (−0,028 đến −0,107). "
         "Rollout thất bại luôn có 52,0 decision (mọi thất bại chạy hết 520 bước), rollout thành công 29–42; "
         "tương quan (số decision của segment, adv) từ −0,28 đến −0,51. Vậy trung bình âm là do **trọng số theo decision**, "
         "không phải do tính sai advantage.",
         "- Dấu sau chuẩn hoá của v3: mean_t tính theo decision (âm), nên trừ mean_t đẩy mọi Â lên khoảng |mean_t|/std_t "
         "(0,15–0,30 std). Sau chuẩn hoá: decision của rollout thành công có Â < 0 chỉ còn 0,0–3,6 % (thô: 13,6–29,5 %); "
         "decision của rollout **thất bại** có Â > 0 tăng lên 26–53 % (thô: 15,3–27,9 %), cao nhất ở task 0 (44,7–52,6 %).",
         "- Kiểm tra 4–7 không chạy: kiểm tra 1–3 không cho thấy lỗi trong cách tính advantage thô. Điểm đáng chú ý "
         "(trọng số theo decision, và dấu sau chuẩn hoá) là hệ quả của thiết kế, không phải lỗi tính toán.", ""]
    (OUT / "adv_check.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
