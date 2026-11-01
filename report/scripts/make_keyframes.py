"""Keyframe strips and side-by-side videos from the exact-replay videos of results/videos/3_v4_paired (CPU only).

Input: ~/ffi26/results/videos/3_v4_paired/{arm}/{arm}_t{task}_init{k}_s{seed}_{success|fail}.mp4, written by
improve/scripts/phase3_eval_video.py (frame = 60 px text header + agent view 256x256 | wrist view 256x256, 20 fps).
Replays are only used if _meta/replay_check_{arm}.csv says match = True for that episode (else the script stops).
Outputs (~/ffi26/results/videos/3_v4_paired/):
  keyframes/{t}_{init}_s{seed}_{arm}.png   4 evenly spaced agent-view frames (0, T/3, 2T/3, T-1), upright as in the
                                          video, border green = success, red = fail
  sidebyside/t{t}_init{k}_s{seed}.mp4     official | v3 | v4 (with headers), synchronised by step; a finished episode
                                          holds its last frame
  keyframes/index.csv                     episode, arm, outcome, length, frame indices
and report/figures/fig_keyframes.pdf (3 episodes x 3 arms, chosen below).

    conda activate lerobot && python report/scripts/make_keyframes.py
"""

from pathlib import Path

import imageio.v3 as iio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

V = Path.home() / "ffi26/results/videos/3_v4_paired"
FIG = Path(__file__).resolve().parents[1] / "figures"
ARMS = ["official", "v3", "v4"]
LAB = {"official": "official", "v3": "FPO v3", "v4": "FPO v4"}
HEADER, W = 60, 256
EPISODES = [(0, 4, 5000), (0, 16, 5000), (0, 9, 4000), (0, 2, 5000), (8, 0, 3000), (8, 9, 3000), (8, 6, 3000), (8, 2, 3000)]
IN_PDF = [(0, 4, 5000), (8, 0, 3000), (8, 6, 3000)]
GREEN, RED = np.array([49, 163, 84]), np.array([222, 45, 38])


def video(arm, t, k, s):
    f = list((V / arm).glob(f"{arm}_t{t}_init{k}_s{s}_*.mp4"))
    assert len(f) == 1, (arm, t, k, s, f)
    return f[0], f[0].stem.endswith("success")


def border(img, col, w=6):
    out = img.copy()
    out[:w], out[-w:], out[:, :w], out[:, -w:] = col, col, col, col
    return out


def main():
    (V / "keyframes").mkdir(exist_ok=True)
    (V / "sidebyside").mkdir(exist_ok=True)
    checks = pd.concat([pd.read_csv(V / "_meta" / f"replay_check_{a}.csv") for a in ARMS])
    rows, strips = [], {}
    for t, k, s in EPISODES:
        clips = {}
        for arm in ARMS:
            c = checks[(checks.arm == arm) & (checks.seed == s) & (checks.task == t) & (checks.init == k)]
            assert len(c) == 1 and bool(c.match.iloc[0]), f"replay of {arm} t{t} init{k} s{s} does not match the original"
            path, ok = video(arm, t, k, s)
            fr = iio.imread(path, index=None)
            clips[arm] = fr
            T = len(fr)
            idx = [0, T // 3, 2 * T // 3, T - 1]
            # the replay videos already show the agent view upright (phase3_eval_video.py); no rotation here
            agent = [fr[i, HEADER:, :W] for i in idx]
            strip = np.concatenate([border(a, GREEN if ok else RED) for a in agent], axis=1)
            iio.imwrite(V / "keyframes" / f"t{t}_init{k}_s{s}_{arm}.png", strip)
            strips[(t, k, s, arm)] = (strip, ok, T)
            rows.append({"task": t, "init": k, "seed": s, "arm": arm, "success": ok, "frames": T, "frame_idx": idx,
                         "video": path.name})
        # side by side, synchronised by step
        T = max(len(c) for c in clips.values())
        sbs = [np.concatenate([c[min(i, len(c) - 1)] for c in clips.values()], axis=1) for i in range(T)]
        iio.imwrite(V / "sidebyside" / f"t{t}_init{k}_s{s}.mp4", np.stack(sbs), fps=20, codec="libx264")
        print("ok", t, k, s, {a: len(c) for a, c in clips.items()})
    pd.DataFrame(rows).to_csv(V / "keyframes" / "index.csv", index=False)

    fig, axes = plt.subplots(len(ARMS), len(IN_PDF), figsize=(7.2, 2.5))
    for j, (t, k, s) in enumerate(IN_PDF):
        for i, arm in enumerate(ARMS):
            strip, ok, T = strips[(t, k, s, arm)]
            ax = axes[i, j]
            ax.imshow(strip)
            ax.set_xticks([]), ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if j == 0:
                ax.set_ylabel(LAB[arm], fontsize=7, rotation=0, ha="right", va="center")
            ax.text(strip.shape[1] - 8, 20, f"{'thành công' if ok else 'thất bại'}, {T} bước", color="white",
                    fontsize=5.5, ha="right", va="top", bbox=dict(facecolor="black", alpha=0.55, pad=1, lw=0))
        axes[0, j].set_title(f"task {t}, init {k}, seed {s}", fontsize=7)
    plt.subplots_adjust(wspace=0.04, hspace=0.06, left=0.09, right=0.995, top=0.9, bottom=0.01)
    fig.savefig(FIG / "fig_keyframes.pdf", dpi=220)
    print("figure", FIG / "fig_keyframes.pdf")


if __name__ == "__main__":
    main()
