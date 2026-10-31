"""Round 2, step 0: run the reward_r2 failure-event detectors and reward on logged baseline
episodes (offline, no GPU), and save frames around events for visual checks.

    python scripts/r2_check_detectors.py --dirs outputs/baseline/logged outputs/eval_baseline \
        --out ~/ffi26/results/round2/detector_check
Outputs: events.csv (one row per event), episodes.csv (per episode: success, return, counts,
segments), frames/<src>_task{t}_ep{k}_<type>_<step>.png (only for episodes with an .mp4).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"), str(Path(__file__).resolve().parent)]

import av  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from failure_labels import label_episode  # noqa: E402
from phase1_logging import load_episode  # noqa: E402
from reward_r2 import RewardR2Config, detect_events, milestones, returns_to_go, step_rewards_r2  # noqa: E402


def frames_at(mp4: Path, steps: list[int]) -> dict[int, np.ndarray]:
    want = set(steps)
    out = {}
    with av.open(str(mp4)) as c:
        for i, f in enumerate(c.decode(c.streams.video[0])):
            if i in want:
                out[i] = f.to_ndarray(format="rgb24")
            if len(out) == len(want):
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_frames", type=int, default=40)
    args = ap.parse_args()
    out = Path(args.out).expanduser()
    (out / "frames").mkdir(parents=True, exist_ok=True)
    cfg = RewardR2Config()
    ev_rows, ep_rows = [], []
    n_img = 0
    for d in args.dirs:
        d = Path(d).expanduser()
        src = d.parent.name + "/" + d.name if d.name == "logged" else d.name
        for npz in sorted(d.glob("task*/ep*.npz")):
            log, actions, meta = load_episode(npz)
            t = int(npz.parent.name[4:])
            k = int(npz.stem[2:])
            succ = bool(meta["success"])
            ev = detect_events(log, actions, succ)
            r = step_rewards_r2(log, actions, succ, cfg, ev)
            G = returns_to_go(r, cfg.gamma)
            lab = label_episode(log, actions, succ)
            ms = milestones(log, len(actions))
            for e in ev:
                ev_rows.append({"src": src, "task": t, "ep": k, "success": succ, **e})
            cnt = pd.Series([e["type"] for e in ev]).value_counts().to_dict()
            ep_rows.append({"src": src, "task": t, "ep": k, "success": succ, "length": len(actions),
                            "return": G[0], "phi_T": float(log["phi"][-1]), "primary_label": lab["primary"],
                            "milestones": " ".join(map(str, ms)),
                            **{f"n_{x}": cnt.get(x, 0) for x in ("drop", "slip", "collision", "timeout")}})
            mp4 = npz.with_suffix(".mp4")
            show = [e for e in ev if e["type"] in ("drop", "slip", "collision")]
            if mp4.exists() and show and n_img < args.max_frames:
                # video frame i = observation before action i; show before / at / after the event
                steps = sorted({max(e["step"] + o, 0) for e in show for o in (-10, 0, 15)})
                fr = frames_at(mp4, steps)
                for e in show:
                    tiles = [fr[s] for s in (max(e["step"] - 10, 0), e["step"], e["step"] + 15) if s in fr]
                    if tiles:
                        Image.fromarray(np.concatenate(tiles, axis=1)).save(
                            out / "frames" / f"{src.replace('/', '_')}_task{t}_ep{k:03d}_{e['type']}_{e['step']}.png")
                        n_img += 1
    ev_df, ep_df = pd.DataFrame(ev_rows), pd.DataFrame(ep_rows)
    ev_df.to_csv(out / "events.csv", index=False)
    ep_df.to_csv(out / "episodes.csv", index=False)
    json.dump({"reward_config": cfg.to_dict(), "dirs": args.dirs}, open(out / "config.json", "w"), indent=1)
    print(ep_df.groupby(["task", "success"])[["return", "n_drop", "n_slip", "n_collision", "n_timeout"]]
          .agg(["mean", "min", "max"]).round(3).to_string())
    print("\nevents by type/task/success:\n", ev_df.groupby(["type", "task", "success"]).size().to_string())
    print("\nfailures: primary label vs events\n",
          ep_df[~ep_df.success].groupby(["task", "primary_label"])[["n_drop", "n_slip", "n_collision"]].sum().to_string())
    print(f"\nframes saved: {n_img}")


if __name__ == "__main__":
    main()
