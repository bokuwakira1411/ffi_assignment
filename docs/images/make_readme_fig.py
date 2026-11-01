"""Hình cho readme.md: official và FPO v4 trên cùng episode (camera agent, 4 khung đều nhau mỗi episode).

Nguồn: keyframe của replay chính xác ~/ffi26/results/videos/3_v4_paired/keyframes/t{task}_init{k}_s{seed}_{arm}.png
và keyframes/index.csv (kết quả, số bước), sinh bởi report/scripts/make_keyframes.py. Viền xanh = thành công, đỏ = thất bại.
Episode = 2 episode của hình keyframe trong báo cáo mà v4 thắng official (chọn để minh hoạ).

    python docs/images/make_readme_fig.py   -> docs/images/official_vs_v4.png
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

K = Path.home() / "ffi26/results/videos/3_v4_paired/keyframes"
OUT = Path(__file__).resolve().parent / "official_vs_v4.png"
TASK = {0: "task 0: bỏ hộp súp và sốt cà chua vào giỏ", 8: "task 8: đặt cả hai ấm moka lên bếp"}
EPISODES = [(0, 4, 5000), (8, 0, 3000)]
ARMS = [("official", "official"), ("v4", "FPO v4 (ours)")]
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"


def main():
    idx = {(int(r["task"]), int(r["init"]), int(r["seed"]), r["arm"]): r for r in csv.DictReader(open(K / "index.csv"))}
    W, IMG_W, LEFT = 8.0, 5.6, 1.25                      # inches; strip is 4:1
    strip_h, gap_arm, gap_ep, title_h, foot_h = IMG_W / 4, 0.06, 0.25, 0.28, 0.3
    H = foot_h + len(EPISODES) * (title_h + 2 * strip_h + gap_arm) + (len(EPISODES) - 1) * gap_ep + 0.05
    fig = plt.figure(figsize=(W, H), dpi=150)
    fig.patch.set_facecolor(SURF)
    y = H - 0.05
    for e, (t, k, s) in enumerate(EPISODES):
        y -= title_h
        fig.text(LEFT / W, (y + 0.07) / H, f"{TASK[t]}  (init {k}, seed {s})", fontsize=9, color=INK)
        for a, (arm, label) in enumerate(ARMS):
            y -= strip_h + (gap_arm if a else 0)
            ax = fig.add_axes((LEFT / W, y / H, IMG_W / W, strip_h / H))
            r = idx[(t, k, s, arm)]
            ok = r["success"] == "True"
            ax.imshow(mpimg.imread(K / f"t{t}_init{k}_s{s}_{arm}.png"))
            ax.axis("off")
            fig.text((LEFT - 0.08) / W, (y + strip_h / 2) / H, label, fontsize=9, color=INK, ha="right", va="center")
            fig.text((LEFT + IMG_W + 0.08) / W, (y + strip_h / 2) / H,
                     ("thành công" if ok else "thất bại") + f"\n{r['frames']} bước", fontsize=8, color=INK2, va="center")
        y -= gap_ep
    fig.text(0.01, 0.08 / H, "Replay chính xác cùng (task, init, seed). Mỗi dải: 4 khung đều nhau của camera agent. "
             "Viền xanh = thành công, đỏ = thất bại; thất bại = hết 520 bước.", fontsize=6.8, color=INK2)
    fig.savefig(OUT, facecolor=SURF)


if __name__ == "__main__":
    main()
