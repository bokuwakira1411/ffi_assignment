"""FPO v2 vs v3: gradient steps, samples used, and log-based diagnostics of why v3 ran more steps. CPU only, no training.

Inputs: outputs/fpo_v{2,3}/iter{0..3}/train/{train_info.json,train_log.csv} and the stored advantages
outputs/fpo_v{2,3}/iter{k}/decisions/g*.npz (only keys adv / rejected / task_id are read; same filter as
fpo_train.load_decisions: rejected == 0 and adv != 0). Iteration 0 uses the SAME decisions for v2 and v3 (v3 reused
v2 iter0), same theta0, same ratio scale c, so it is the cleanest comparison.
Samples per step (from the training code): PPO batch_size decisions (v2: one task per step; v3: B/2 per task),
rollout-BC min(bc_batch, n_success) decisions, v3 also demo_batch demo samples.
Advantage used in the loss: v2 A^ = clip(A / std(A over both tasks), -5, 5) (fpo_train_v2.py:220);
v3 A^ = clip((A - mean_t) / std_t, -5, 5) (fpo_train_v3.py:102).
Outputs: ~/ffi26/results/round2/fpo_v3/grad_steps.md (Vietnamese), grad_steps.csv, grad_steps_iter0_curve.csv.

    python scripts/fpo_v3_grad_steps.py
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

H = Path.home() / "ffi26"
O = H / "improve/outputs"
OUT = H / "results/round2/fpo_v3"
RUNS = {"v2": O / "fpo_v2", "v3": O / "fpo_v3"}


def advs(it_dir: Path) -> dict[int, np.ndarray]:
    per: dict[int, list] = {}
    for f in sorted(glob.glob(str(it_dir / "decisions" / "g*.npz"))):
        z = np.load(f)
        a, m = z["adv"], (z["rejected"] == 0) & (z["adv"] != 0)
        if m.any():
            per.setdefault(int(z["task_id"][0]), []).append(a[m])
    return {t: np.concatenate(v) for t, v in per.items()}


def md(d: pd.DataFrame) -> str:
    return "\n".join(["| " + " | ".join(map(str, d.columns)) + " |", "|" + "---|" * len(d.columns)]
                     + ["| " + " | ".join(str(v) for v in r) + " |" for r in d.itertuples(index=False)])


def main():
    rows, adv_rows, logs = [], [], {}
    for run, root in RUNS.items():
        for k in range(4):
            info = json.load(open(root / f"iter{k}/train/train_info.json"))
            log = pd.read_csv(root / f"iter{k}/train/train_log.csv")
            logs[(run, k)] = log
            f = info["fpo"] if "fpo" in info else {}
            B = info["batch_size"]
            nsucc = info["success_decisions"]
            nsucc = sum(nsucc.values()) if isinstance(nsucc, dict) else nsucc
            ndec = info["decisions"]
            ndec = sum(ndec.values()) if isinstance(ndec, dict) else ndec
            bc = 32 if nsucc >= 32 else nsucc  # bc_batch 32 (fpo_v2.yaml / fpo_v3.yaml); v3: 16 per task
            demo = 32 if run == "v3" else 0
            s = info["steps"]
            rows.append({"run": run, "iter": k, "decisions": ndec, "steps": s, "steps_per_epoch": info["steps_per_epoch"],
                         "epochs": round(info["epochs_used"], 3), "ppo_samples": s * B, "bc_samples": s * bc,
                         "demo_samples": s * demo, "clip_frac_mean": round(info["clip_frac_mean"], 4),
                         "clip_frac_last10": round(info["clip_frac_last10"], 4),
                         "kl_end": round(info["kl_end"]["kl_proxy"], 4), "train_s": round(info["seconds_total"])})
            a = advs(root / f"iter{k}")
            sd_all = float(np.concatenate(list(a.values())).std())
            for t, x in sorted(a.items()):
                h = np.clip(x / sd_all, -5, 5) if run == "v2" else np.clip((x - x.mean()) / x.std(), -5, 5)
                adv_rows.append({"run": run, "iter": k, "task": t, "n": len(x), "A_mean": round(float(x.mean()), 4),
                                 "A_std": round(float(x.std()), 4), "Ahat_mean": round(float(h.mean()), 3),
                                 "Ahat_absmean": round(float(np.abs(h).mean()), 3), "Ahat_std": round(float(h.std()), 3),
                                 "frac_Ahat_neg": round(float((h < 0).mean()), 3)})
    st = pd.DataFrame(rows)
    ad = pd.DataFrame(adv_rows)
    tot = st.groupby("run")[["steps", "ppo_samples", "bc_samples", "demo_samples", "train_s"]].sum().reset_index()
    st.to_csv(OUT / "grad_steps.csv", index=False)
    ad.to_csv(OUT / "grad_steps_adv.csv", index=False)

    # iteration 0 curves (same data): rolling-10 clip_frac, ratio drift, grad norm
    cur = []
    for run in RUNS:
        lg = logs[(run, 0)].copy()
        lg["clip10"] = lg.clip_frac.rolling(10, min_periods=1).mean()
        for s in [5, 10, 15, 20, 23, 30, 50, 80, 110, 131]:
            r = lg[lg.step == s]
            if len(r):
                r = r.iloc[0]
                cur.append({"run": run, "step": s, "clip_frac": round(r.clip_frac, 3), "clip_frac_roll10": round(r.clip10, 3),
                            "ratio_mean": round(r.ratio_mean, 4), "std_logr": round(r.std_logr, 4),
                            "dl_absmean": f"{r.dl_absmean:.2e}", "grad_norm": round(r.grad_norm, 2)})
    cur = pd.DataFrame(cur)
    cur.to_csv(OUT / "grad_steps_iter0_curve.csv", index=False)

    # v2 iteration 0 per task (one task per step) vs v3 per-task clip_frac columns
    pt = []
    for run in RUNS:
        for k in range(4):
            lg = logs[(run, k)]
            r = {"run": run, "iter": k}
            if "task" in lg:
                for t in (0, 8):
                    g = lg[lg.task == t]
                    r[f"clip_t{t}_mean"] = round(g.clip_frac.mean(), 3)
                    r[f"ratio_mean_t{t}"] = round(g.ratio_mean.mean(), 4)
            else:
                for t in (0, 8):
                    r[f"clip_t{t}_mean"] = round(lg[f"clip_frac_t{t}"].mean(), 3)
                r["ratio_mean_t0"] = r["ratio_mean_t8"] = "— (không log theo task)"
            r["ratio_mean_all"] = round(lg.ratio_mean.mean(), 4)
            r["ratio_mean_last10"] = round(lg.ratio_mean.tail(10).mean(), 4)
            r["grad_norm_mean"] = round(lg.grad_norm.mean(), 2)
            r["std_logr_last10"] = round(lg.std_logr.tail(10).mean(), 4)
            pt.append(r)
    pt = pd.DataFrame(pt)
    pt.to_csv(OUT / "grad_steps_per_task.csv", index=False)

    L = ["# FPO v2 vs v3: số bước gradient, số mẫu, và vì sao v3 chạy nhiều bước hơn", "",
         "Script `improve/scripts/fpo_v3_grad_steps.py` (CPU, không train). Nguồn: `outputs/fpo_v{2,3}/iter{k}/train/"
         "train_info.json`, `train_log.csv`, advantage trong `iter{k}/decisions/g*.npz` (lọc như `load_decisions`). "
         "Mẫu mỗi bước theo code: PPO = 128 decision; BC rollout = 32; v3 thêm 32 mẫu demo. Iter 0 dùng CÙNG decision, "
         "cùng θ0, cùng c = 24,61 cho v2 và v3, nên là so sánh sạch nhất.", "",
         "## Tổng", "", md(tot), "", "## Theo vòng", "", md(st), "",
         "## Advantage dùng trong loss (Â), theo vòng × task", "",
         "v2: Â = clip(A / std(A cả hai task), ±5), KHÔNG trừ trung bình; v3: Â = clip((A − mean_t) / std_t, ±5).", "",
         md(ad), "", "## Iter 0 (cùng dữ liệu): theo bước", "", md(cur), "",
         "## clip_frac theo task và độ lệch ratio, theo vòng", "",
         "v2: mỗi bước một task (cột `task` của log); v3: cột `clip_frac_t0/t8` (nửa batch mỗi task).", "", md(pt), ""]
    # iteration 0, matched step ranges: clip_frac of each task's samples (v2: steps of that task; v3: that task's half)
    v2, v3 = logs[("v2", 0)], logs[("v3", 0)]
    mt = []
    for lo, hi in [(1, 10), (11, 23), (24, 50), (51, 100), (101, 131)]:
        a, b = v2[(v2.step >= lo) & (v2.step <= hi)], v3[(v3.step >= lo) & (v3.step <= hi)]
        f2 = lambda t: f"{a[a.task == t].clip_frac.mean():.3f} (n = {int((a.task == t).sum())})" if (a.task == t).any() else "— (đã dừng)"  # noqa: E731
        mt.append({"bước": f"{lo}–{hi}", "v2 bước task 0": f2(0), "v2 bước task 8": f2(8),
                   "v3 nửa task 0": f"{b.clip_frac_t0.mean():.3f}", "v3 nửa task 8": f"{b.clip_frac_t8.mean():.3f}",
                   "v3 cả batch": f"{b.clip_frac.mean():.3f}"})
    win = []
    for k in range(4):
        w = logs[("v2", k)].tail(10)
        win.append({"vòng": k, "số bước task 0 / 10": int((w.task == 0).sum()),
                    "clip bước task 0": round(w[w.task == 0].clip_frac.mean(), 3),
                    "clip bước task 8": round(w[w.task == 8].clip_frac.mean(), 3)})
    big = [f"{r} iter{k}: " + ", ".join(f"bước {int(x.step)} ({x.grad_norm:.0f})" for x in logs[(r, k)].itertuples()
                                       if x.grad_norm > 100) for r in RUNS for k in range(4)
           if (logs[(r, k)].grad_norm > 100).any()]
    allg = {r: pd.concat([logs[(r, k)] for k in range(4)]).grad_norm for r in RUNS}
    clipped = "; ".join(f"{r}: {100 * (g > 10).mean():.0f} % số bước có grad_norm > 10, trung vị {g.median():.1f}"
                        for r, g in allg.items())
    L += ["## Iter 0, cùng khoảng bước: clip_frac theo task", "",
          "v2: trung bình trên các bước của task đó; v3: cột `clip_frac_t0/t8` (nửa batch của task).", "", md(pd.DataFrame(mt)), "",
          "## v2: cửa sổ 10 bước cuối (quyết định early stop), theo task", "", md(pd.DataFrame(win)), "",
          "Bước có grad_norm > 100 (trước khi clip về 10): " + "; ".join(big) + ".", "",
          "## Kết luận (chỉ từ log, không có ablation)", "",
          f"- Tổng: v2 {int(tot.set_index('run').steps['v2'])} bước, v3 {int(tot.set_index('run').steps['v3'])} bước; "
          "mẫu PPO / BC rollout / demo ở bảng Tổng. Cả 8 vòng dừng do early stop (clip_frac 10 bước cuối > 0,2), "
          "không vòng nào chạm trần 3 epoch.",
          "- Iter 0 (cùng dữ liệu, cùng θ0, cùng c): ở bước 11–23, clip_frac của TỪNG task trong v3 chỉ khoảng một nửa v2. "
          "Vậy v3 không chỉ \"trung bình hoá\" hai task: ratio của mỗi task trôi chậm hơn mỗi bước. Riêng việc trộn task "
          "(batch cân bằng) không giải thích được điều này.",
          "- Thành phần cửa sổ của v2 có góp phần: bước task 0 clip nhiều hơn bước task 8 (bảng trên), và cửa sổ cuối "
          "của iter 0 có 7/10 bước task 0.",
          "- Hai cơ chế khớp với log nhưng KHÔNG tách được nếu không train lại: (a) v2 không trừ trung bình advantage "
          "(Â trung bình −0,31 ở task 0, 63 % mẫu Â < 0), nên mỗi bước đẩy l_new của đa số mẫu theo cùng một hướng; v3 "
          "trừ trung bình theo task (Â trung bình 0). (b) v3 thêm λ_demo × L_CFM(demo) (32 mẫu, λ 5): AdamW + clip "
          "grad norm 10 (" + clipped + ") nên một phần bước cập nhật "
          "dành cho demo thay vì PPO.",
          "- Gai grad_norm xuất hiện ở cả hai (v2 iter0 bước 14–15 ngay trước early stop; v3 iter0 bước 84), nên gai "
          "không phải khác biệt hệ thống. Xác nhận nguyên nhân cần ablation (v3 bỏ demo; v3 không trừ trung bình), "
          "chưa chạy (yêu cầu: không train thêm).", ""]
    (OUT / "grad_steps.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
