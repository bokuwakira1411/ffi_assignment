"""Đếm các loại lỗi thao tác (L2) của checkpoint official, dùng cho docs/analyse.md §2.1 và report/main.tex Bảng L2.

Đầu vào: improve/outputs/failure_labels/labels.csv (nhãn tự động v2) và các file .npz trong cột `npz`
(eval fpo_v2, arm official: NAS=10, expert fp32; task 0, 8 x 50 init x seed 3000/4000/5000; 8 task khác x 20 init, seed 1000).
Chỉ đọc, không ghi file. Chạy: cd ffi26/ffi26/improve && ~/.conda/envs/lerobot/bin/python ../../../docs/failures/tools/l2_counts.py
"""

import collections
import csv
import sys
from pathlib import Path

import numpy as np

sys.path[:0] = ["src", "scripts"]
from phase1_logging import load_episode  # noqa: E402
from reward_r2 import detect_events  # noqa: E402

MECH = ["A", "C", "F", "G", "D"]
rows = [r for r in csv.DictReader(open("outputs/failure_labels/labels.csv")) if r["arm"] == "official"]


def group(r):
    return {"0": "task0", "8": "task8"}.get(r["task"], "8 task khác")


def lab(r, k):
    return r[f"lab_{k}"] == "True"


stats = collections.defaultdict(collections.Counter)
objs = collections.defaultdict(collections.Counter)
g_dist = collections.defaultdict(list)
for r in rows:
    succ = r["success"] == "True"
    outcome = "thành công" if succ else "thất bại"
    log, act, _ = load_episode(Path(r["npz"]))
    types = {e["type"] for e in detect_events(log, act, succ)}
    for k in (group(r), "tổng"):
        kk = (k, outcome)
        s = stats[kk]
        s["n"] += 1
        for m in MECH + ["E_contact", "E_free", "B"]:
            s[m] += lab(r, m)
        s["có nhãn A/C/F/G/D"] += any(lab(r, m) for m in MECH)
        s["không nhấc được vật goal nào"] += r["lifted_goal_objs"] == ""
        s["nhấc cả hai vật goal (task 8)"] += r["lifted_goal_objs"] == "moka_pot_1;moka_pot_2"
        s["có slip"] += "slip" in types
        if r["C_objs"]:
            names = list(log["obj_names"])
            grasp = np.asarray(log["grasp"], bool)
            s["C: có cầm vật ngoài goal"] += any(grasp[:, names.index(o)].any() for o in r["C_objs"].split(";"))
        if not succ:
            for col in ("A_objs", "C_objs", "F_objs", "G_objs", "D_preds"):
                objs[(kk[0], col)].update(o for o in r[col].split(";") if o)
            g_dist[kk[0]] += [float(x) for x in r["G_dist"].split(";") if x]

for kk in sorted(stats):
    print(kk, dict(stats[kk]))
for k in sorted(objs):
    print(k, dict(objs[k].most_common()))
for k, v in sorted((k, v) for k, v in g_dist.items() if v):
    v = sorted(v)
    print("G_dist", k, f"n={len(v)} min={v[0]:.3f} trung vị={v[len(v) // 2]:.3f} max={v[-1]:.3f}")
