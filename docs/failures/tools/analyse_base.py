"""Analyse (main_revised.tex): một bộ dữ liệu duy nhất cho SmolVLA gốc, NAS=10, expert fp32, có log mô phỏng từng bước.

Dữ liệu: 10 task x 50 init x seed 3000/4000/5000 = 1500 episode.
  task 0, 8:      improve/outputs/fpo_v2/eval_s{seed}/official   (arm baseline của eval FPO v2)
  task 1-7, 9:    improve/outputs/analyse_base/eval_s{seed}/official   (chạy 02/11/2026, cùng thiết lập)
Nhãn: improve/src/failure_labels.label_episode_v2 (giống scripts/failure_labels_run.py), sự kiện slip/drop/collision:
improve/src/reward_r2.detect_events. Tâm vùng đích (cho nhãn G) ước lượng lại trên chính 1500 episode này.
Ghi: improve/outputs/analyse_base/{labels.csv, summary.md}. Không sửa file nào khác.
Chạy: cd ffi26/ffi26/improve && ~/.conda/envs/lerobot/bin/python ../../../docs/failures/tools/analyse_base.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.home() / "ffi26/improve"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from failure_labels import PRIORITY_V2, label_episode_v2  # noqa: E402
from failure_labels_run import region_centres, mcnemar_exact  # noqa: E402,F401
from phase1_logging import load_episode  # noqa: E402
from reward_r2 import detect_events  # noqa: E402

OUT = ROOT / "outputs/analyse_base"
SEEDS = (3000, 4000, 5000)
SRC = {t: ROOT / ("outputs/fpo_v2/eval_s{s}/official" if t in (0, 8) else "outputs/analyse_base/eval_s{s}/official")
       for t in range(10)}
N_INIT = 50
LABS = ["A", "C", "F", "G", "D"]
NAME = {"A": "gắp hỏng", "C": "chạm/cầm vật ngoài goal", "F": "làm rơi", "G": "đặt gần đích nhưng chưa đúng",
        "D": "làm mất tiến độ"}


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def load():
    eps, missing = [], []
    for t, root in SRC.items():
        for s in SEEDS:
            d = Path(str(root).format(s=s)) / f"task{t}"
            for k in range(N_INIT):
                npz = d / f"ep{k:03d}.npz"
                if not npz.exists():
                    missing.append(str(npz))
                    continue
                log, act, meta = load_episode(npz)
                eps.append({"seed": s, "task": t, "init": k, "success": bool(meta["success"]), "log": log, "act": act,
                            "task_text": meta["task"], "npz": str(npz)})
    return eps, missing


def main():
    eps, missing = load()
    if missing:
        sys.exit(f"thiếu {len(missing)} episode, ví dụ {missing[:3]}")
    reg = region_centres(eps)
    rxy = {t: {(r.region, r.object): (r.x, r.y) for r in g.itertuples()} for t, g in reg.groupby("task")}
    rows = []
    for e in eps:
        lab = label_episode_v2(e["log"], e["act"], e["success"], region_xy=rxy.get(e["task"]))
        ev = {x["type"] for x in detect_events(e["log"], e["act"], e["success"])}
        rows.append({"seed": e["seed"], "task": e["task"], "init": e["init"], "task_text": e["task_text"],
                     "success": e["success"], "length": len(e["act"]), **lab,
                     **{f"ev_{k}": k in ev for k in ("slip", "drop", "collision")}, "npz": e["npz"]})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "labels.csv", index=False)

    df["partial"] = (df.subtask_completion > 0) & (df.subtask_completion < 1)  # xong một phần goal ở cuối episode
    df["none_done"] = df.subtask_completion == 0
    df["any_mech"] = df[[f"lab_{k}" for k in LABS]].any(axis=1)
    f, s = df[~df.success], df[df.success]

    L = ["# Analyse: SmolVLA gốc, 10 task × 50 init × 3 seed (1500 episode)", "",
         "Script: `docs/failures/tools/analyse_base.py`. Thiết lập: `lerobot/smolvla_libero`, NAS=10, expert fp32, "
         "seed 3000/4000/5000; task 0, 8 từ `outputs/fpo_v2/eval_s*/official`, task khác từ "
         "`outputs/analyse_base/eval_s*/official`. Nhãn: `label_episode_v2`; sự kiện: `detect_events`.", "",
         "## 1. Tỉ lệ thành công theo task", "", "| task | câu lệnh | thành công /150 | % (Wilson 95%) |", "|---|---|---|---|"]
    for t, g in df.groupby("task"):
        k, n = int(g.success.sum()), len(g)
        lo, hi = wilson(k, n)
        L.append(f"| {t} | {g.task_text.iloc[0]} | {k} | {100 * k / n:.1f} ({lo:.1f}–{hi:.1f}) |")
    k, n = int(df.success.sum()), len(df)
    lo, hi = wilson(k, n)
    L += [f"| tổng | | {k}/{n} | {100 * k / n:.1f} ({lo:.1f}–{hi:.1f}) |", ""]

    L += ["## 2. Thất bại: xong một phần hay chưa xong gì (cuối episode)", "",
          "`subtask_completion` = tỉ lệ predicate goal (sai lúc đầu) đúng ở bước cuối.", "",
          "| task | thất bại | xong một phần | chưa xong gì |", "|---|---|---|---|"]
    for t, g in f.groupby("task"):
        L.append(f"| {t} | {len(g)} | {int(g.partial.sum())} | {int(g.none_done.sum())} |")
    L += [f"| tổng | {len(f)} | {int(f.partial.sum())} | {int(f.none_done.sum())} |", ""]

    L += ["## 3. Lỗi thao tác (đa nhãn): số episode có nhãn, thất bại / thành công", "",
          "| nhãn | " + " | ".join(f"task {t}" for t in range(10)) + " | tổng thất bại | tổng thành công |",
          "|---|" + "---|" * 12]
    items = [(f"lab_{k}", f"{k} {NAME[k]}") for k in LABS] + [
        ("ev_slip", "slip (trượt khỏi kẹp)"), ("ev_drop", "drop"), ("ev_collision", "collision"),
        ("any_mech", "có ≥1 nhãn A/C/F/G/D")]
    for col, name in items:
        per = [f"{int(f[f.task == t][col].sum())}/{int(s[s.task == t][col].sum())}" for t in range(10)]
        L.append(f"| {name} | " + " | ".join(per) + f" | {int(f[col].sum())}/{len(f)} | {int(s[col].sum())}/{len(s)} |")
    L += ["", "Ô task: thất bại có nhãn / thành công có nhãn.", ""]

    L += ["## 4. Nhãn chính (một nhãn mỗi thất bại, thứ tự ưu tiên " + " > ".join(PRIORITY_V2) + " > O)", ""]
    pc = pd.crosstab(f.task, f.primary)
    L += ["| task | " + " | ".join(pc.columns) + " |", "|---|" + "---|" * len(pc.columns)]
    for t, r in pc.iterrows():
        L.append(f"| {t} | " + " | ".join(str(int(v)) for v in r) + " |")
    L.append("")

    L += ["## 5. Độ ổn định theo cảnh (500 cảnh × 3 seed)", ""]
    sc = df.groupby(["task", "init"]).success.sum()
    L += [f"- thành công cả 3 seed: {int((sc == 3).sum())}; thất bại cả 3: {int((sc == 0).sum())}; "
          f"lẫn lộn: {int(((sc > 0) & (sc < 3)).sum())}", ""]

    (OUT / "summary.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
