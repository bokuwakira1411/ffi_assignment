"""Does FPO v2 theta4 pick up task-0 objects on tasks 1 / 7 (forgetting eval, seed 1000)?

Task-0 objects = goal objects of task 0 (alphabet_soup_1, tomato_sauce_1; basket excluded: shared container) that
are NOT goal objects of the task being evaluated (task 1: both; task 7: tomato_sauce_1 only, alphabet soup is a
task-7 goal). Per episode, from the original eval logs (outputs/fpo_v2/eval_s1000/{official,fpo_v2_sel}/task{t};
the video replays matched them 40 / 40) and the detectors of src/reward.py:
  touched = grasp or gripper contact at any step; lifted = lifted_mask (held and >= 3 cm above reset height);
  moved = moved_mask (touched, then displaced > MOVE_THRESHOLD).
These are automatic detectors (not checked by watching every video). Writes results/videos/v2_forget/index.md.

    python scripts/v2_forget_wrong_object.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from reward import lifted_mask, moved_mask  # noqa: E402

H = Path.home() / "ffi26"
ROOT = H / "improve/outputs/fpo_v2/eval_s1000"
ARMS = {"official": "official", "v2": "fpo_v2_sel"}
OUT = H / "results/videos/v2_forget"
TASK0 = ("alphabet_soup_1", "tomato_sauce_1")


def main():
    rows = []
    for arm, d in ARMS.items():
        for t in (1, 7):
            for f in sorted((ROOT / d / f"task{t}").glob("ep*.npz")):
                m = json.load(open(f.with_suffix(".meta.json")))
                z = np.load(f)
                log = {k: z[k] for k in ("obj_pos", "grasp", "contact")}
                names = m["obj_names"]
                wrong = [n for n in TASK0 if n not in m["goal_objs"] and n in names]
                touched = (log["grasp"] | log["contact"]).any(0)
                lifted, moved = lifted_mask(log).any(0), moved_mask(log).any(0)
                ix = {n: names.index(n) for n in wrong}
                rows.append({"arm": arm, "task": t, "init": int(f.name[2:5]), "success": bool(m["success"]),
                             "wrong_objs": ",".join(wrong),
                             "touched_task0": ",".join(n for n, i in ix.items() if touched[i]),
                             "lifted_task0": ",".join(n for n, i in ix.items() if lifted[i]),
                             "moved_task0": ",".join(n for n, i in ix.items() if moved[i]),
                             "lifted_any": ",".join(n for i, n in enumerate(names) if lifted[i]),
                             "touched_any": ",".join(n for i, n in enumerate(names) if touched[i])})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "wrong_object.csv", index=False)

    L = ["# Video: v2 θ4 thất bại ở task 1 và 7 (eval quên, seed 1000)", "",
         "Config: `results/videos/v2_forget/configs/videos_v2_forget.yaml`; script `improve/scripts/phase3_eval_video.py` "
         "(replay chính xác, expert fp32, batch 8, 20 episode/task). Kiểm tra tất định: "
         "`replay_check_v2.csv`, 0/40 episode replay khác bản gốc `outputs/fpo_v2/eval_s1000/fpo_v2_sel`.", "",
         "Vật của task 0 = vật mục tiêu của task 0 (`alphabet_soup_1`, `tomato_sauce_1`; bỏ `basket_1` vì dùng chung) "
         "mà KHÔNG phải vật mục tiêu của task đang chạy: task 1 (cream cheese + butter → basket): cả hai; task 7 "
         "(alphabet soup + cream cheese → basket): chỉ `tomato_sauce_1`.", "",
         "Bộ phát hiện tự động (`src/reward.py`, tính từ log gốc, CHƯA xem video để xác nhận từng episode): "
         "chạm = grasp hoặc tiếp xúc gripper ở bất kỳ bước nào; nhấc = đang giữ và cao hơn vị trí reset ≥ 3 cm; "
         "dịch chuyển = đã chạm rồi bị dời > ngưỡng `MOVE_THRESHOLD`. Script: `improve/scripts/v2_forget_wrong_object.py`; "
         "bảng đầy đủ: `wrong_object.csv`.", "",
         "## Tổng hợp (mọi episode, không chỉ thất bại)", "",
         "| arm | task | episode | thành công | chạm vật task 0 | nhấc vật task 0 | dịch chuyển vật task 0 |", "|---|---|---|---|---|---|---|"]
    for (arm, t), g in df.groupby(["arm", "task"]):
        n = len(g)
        c = lambda col: f"{int((g[col] != '').sum())}/{n}"  # noqa: E731
        L.append(f"| {arm} | {t} | {n} | {int(g.success.sum())}/{n} | {c('touched_task0')} | {c('lifted_task0')} | "
                 f"{c('moved_task0')} |")
    L += ["", "Chỉ episode thất bại:", "", "| arm | task | thất bại | chạm vật task 0 | nhấc vật task 0 |", "|---|---|---|---|---|"]
    for (arm, t), g in df[~df.success].groupby(["arm", "task"]):
        n = len(g)
        L.append(f"| {arm} | {t} | {n} | {int((g.touched_task0 != '').sum())}/{n} | {int((g.lifted_task0 != '').sum())}/{n} |")
    L += ["", "## Video (v2 θ4, episode thất bại)", "",
          "| task | init | file | chạm vật task 0 | nhấc vật task 0 | vật đã nhấc (mọi vật) | official cùng episode |",
          "|---|---|---|---|---|---|---|"]
    off = df[df.arm == "official"].set_index(["task", "init"]).success
    for r in df[(df.arm == "v2") & ~df.success].itertuples():
        fn = f"v2_t{r.task}_init{r.init}_s1000_fail.mp4"
        L.append(f"| {r.task} | {r.init} | [{fn}]({fn}) | {r.touched_task0 or '—'} | {r.lifted_task0 or '—'} | "
                 f"{r.lifted_any or '—'} | {'thành công' if off[(r.task, r.init)] else 'thất bại'} |")
    (OUT / "index.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
