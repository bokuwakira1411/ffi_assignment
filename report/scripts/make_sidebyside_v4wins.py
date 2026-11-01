"""Side-by-side videos: SmolVLA gốc (thất bại) | FPO v4 (thành công) on the same episode (CPU only).

Input: ~/ffi26/results/videos/4_official_fail_v4_success/_meta/replay/{arm}_t{task}_init{k}_s{seed}_{success|fail}.mp4,
written by improve/scripts/phase3_eval_video.py with _meta/configs/videos.yaml (42 episodes: official failed and FPO v4
succeeded in the final eval). Frame = 60 px text header + agent view 256x256 | wrist view 256x256, 20 fps.
A replay is used only if its outcome equals the original (_meta/replay/replay_check_{arm}.csv) and is the expected one
(official fail, v4 success); otherwise the script stops. A replay whose length differs from the original (outcome equal)
is kept and flagged in index.md (1/84 here: v4 t8 init17 s5000, 397 -> 390 steps).
Outputs (~/ffi26/results/videos/4_official_fail_v4_success/):
  sidebyside/t{t}_init{k}_s{seed}.mp4   agent view only, scaled x2: SmolVLA gốc | FPO v4, synchronised by step;
                                       a finished episode holds its last frame; border red = fail, green = success
  index.md                             list of the 42 episodes (Vietnamese)

    conda activate lerobot && python report/scripts/make_sidebyside_v4wins.py
"""

from pathlib import Path

import imageio.v3 as iio
import matplotlib
import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw, ImageFont

V = Path.home() / "ffi26/results/videos/4_official_fail_v4_success"
R = V / "_meta/replay"
ARMS = {"official": "SmolVLA gốc", "v4": "FPO v4"}
EXPECT = {"official": False, "v4": True}
HEADER, W, SCALE, BAR = 60, 256, 2, 48  # 2*256 + 48 = 560, divisible by 16 for libx264
GREEN, RED = np.array([49, 163, 84], np.uint8), np.array([222, 45, 38], np.uint8)
FONT = ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"), 18)


def video(arm, t, k, s):
    f = list(R.glob(f"{arm}_t{t}_init{k}_s{s}_*.mp4"))
    assert len(f) == 1, (arm, t, k, s, f)
    return f[0], f[0].stem.endswith("success")


def panel(frame, arm, ok, step, T, w=6):
    img = frame[HEADER:, :W].repeat(SCALE, 0).repeat(SCALE, 1)
    col = GREEN if ok else RED
    img[:w], img[-w:], img[:, :w], img[:, -w:] = col, col, col, col
    bar = Image.new("RGB", (img.shape[1], BAR), "black")
    d = ImageDraw.Draw(bar)
    d.text((10, 14), f"{ARMS[arm]}: {'thành công' if ok else 'thất bại'}", font=FONT, fill=tuple(int(c) for c in col))
    d.text((img.shape[1] - 10, 14), f"bước {min(step, T - 1) + 1}/{T}", font=FONT, fill="white", anchor="ra")
    return np.concatenate([np.asarray(bar), img], axis=0)


def main():
    cfg = yaml.safe_load(open(V / "_meta/configs/videos.yaml"))
    eps = [(e["task"], e["init"], e["seed"]) for e in cfg["episodes"]]
    checks = pd.concat([pd.read_csv(R / f"replay_check_{a}.csv") for a in ARMS])
    (V / "sidebyside").mkdir(exist_ok=True)
    rows, notes = [], []
    for t, k, s in eps:
        clips, lens = {}, {}
        for arm in ARMS:
            c = checks[(checks.arm == arm) & (checks.seed == s) & (checks.task == t) & (checks.init == k)]
            assert len(c) == 1 and c.success_replay.iloc[0] == c.success_orig.iloc[0], f"{arm} t{t} init{k} s{s}: outcome differs"
            if not bool(c.match.iloc[0]):
                notes.append(f"{ARMS[arm]}, task {t}, init {k}, seed {s}: replay {c.length_replay.iloc[0]} bước, "
                             f"bản gốc {c.length_orig.iloc[0]} bước (cùng kết quả)")
            path, ok = video(arm, t, k, s)
            assert ok == EXPECT[arm], (arm, t, k, s, ok)
            clips[arm] = (iio.imread(path, index=None), ok)
            lens[arm] = len(clips[arm][0])
        T = max(lens.values())
        frames = [np.concatenate([panel(fr[min(i, len(fr) - 1)], arm, ok, i, len(fr)) for arm, (fr, ok) in clips.items()],
                                 axis=1) for i in range(T)]
        name = f"t{t}_init{k}_s{s}.mp4"
        iio.imwrite(V / "sidebyside" / name, np.stack(frames), fps=20, codec="libx264")
        rows.append({"task": t, "init": k, "seed": s, "official_steps": lens["official"],
                     "v4_steps": lens["v4"], "video": name})
        print("ok", name, lens, flush=True)

    df = pd.DataFrame(rows)
    L = ["# SmolVLA gốc thất bại, FPO v4 thành công (cùng cảnh, cùng seed)", "",
         f"{len(df)} episode của eval cuối (task 0, 8 × 50 init × seed 3000/4000/5000) mà SmolVLA gốc thất bại còn "
         "FPO v4 thành công. Mỗi video replay lại episode gốc (cùng init, seed; kết quả khớp bản gốc ở cả 84 video) (`_meta/replay/replay_check_*.csv`).", "",
         "Mở `sidebyside/`: trái là SmolVLA gốc, phải là FPO v4, cùng bước thời gian; camera agent, phóng to ×2. "
         "Episode kết thúc trước thì giữ khung cuối. Video đầy đủ cả camera cổ tay: `_meta/replay/*.mp4`.", "",
         "Script: `report/scripts/make_sidebyside_v4wins.py`; config replay: `_meta/configs/videos.yaml`.", ""]
    if notes:
        L += ["Replay không khớp từng bước với bản gốc (cùng kết quả, khác độ dài; khả năng do tính toán GPU không tất định):", ""]
        L += [f"- {n}" for n in notes] + [""]
    for t, g in df.groupby("task"):
        L += [f"## Task {t} ({len(g)} episode)", "", "| video | init | seed | số bước v4 | số bước gốc |", "|---|---|---|---|---|"]
        L += [f"| `{r.video}` | {r.init} | {r.seed} | {r.v4_steps} | {r.official_steps} |" for r in g.itertuples()]
        L.append("")
    (V / "index.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
