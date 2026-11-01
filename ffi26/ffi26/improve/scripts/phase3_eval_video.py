"""Phase 3 with videos: replay selected eval episodes exactly and save annotated mp4s.

Separate from phase3_eval.py on purpose (that file is called by the running fpo_v2 loop).
Same episode mapping as phase3_eval.py: episode k of (model, seed, task) uses init state k and env
seed seed+k; the torch RNG is reset with set_seed(seed) once per (model, task) and then consumed batch
by batch (batch_size envs). So the policy noise of episode k depends on every earlier batch: to replay
episode k exactly, all batches 0..k//batch_size are re-run. Frames are recorded only in batches that
hold a selected episode; every replayed episode is saved as .npz/.meta.json and compared with the
original run (replay_check.csv) to test determinism.

    python scripts/phase3_eval_video.py --config ~/ffi26/results/videos/configs/videos.yaml [--only official,v1t3] [--blind]
--blind: the header shows only the step and Phi (no arm / task / init / seed, no SUCCESS / FAIL, no reward_r2 events),
and episodes_{arm}.csv gets empty primary / events columns; for manual labelling without hints.
Outputs (<output_dir>):
  {arm}_t{task}_init{k}_s{seed}_{success|fail}.mp4   agent view | wrist view, 20 fps, text header
  raw/{arm}/s{seed}/task{t}/ep{k:03d}.npz|.meta.json all replayed episodes
  replay_check_{arm}.csv                             replay vs original success / length
  episodes_{arm}.csv                                 selected episodes: outcome, labels, events
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from lerobot.utils.io_utils import write_video  # noqa: E402
from lerobot.utils.random_utils import set_seed  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from failure_labels import label_episode  # noqa: E402
from phase1_logging import save_episode  # noqa: E402
from reward_r2 import detect_events  # noqa: E402
from rollout import load_policy, make_vec_env, run_batch  # noqa: E402

FPS = 20
HEADER = 60  # px
EVENT_SHOW = 40  # frames an event stays highlighted (2 s)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"


def annotate(frames: dict, log: dict, events: list[dict], *, arm: str, task: int, init: int, seed: int,
             success: bool, blind: bool = False) -> list[np.ndarray]:
    """Frame i = observation before action i; Phi row i is the same state (log has T+1 rows)."""
    font = ImageFont.truetype(FONT, 12) if Path(FONT).exists() else ImageFont.load_default()
    phi = np.asarray(log["phi"])
    succ = np.asarray(log["success"], bool)
    T = len(frames["image"])
    out = []
    for i in range(T):
        im = np.concatenate([frames["image"][i], frames["image2"][i]], axis=1)
        canvas = Image.new("RGB", (im.shape[1], im.shape[0] + HEADER), (0, 0, 0))
        canvas.paste(Image.fromarray(im), (0, HEADER))
        d = ImageDraw.Draw(canvas)
        if blind:
            d.text((4, 2), f"step {i:3d}/{T}  Phi {phi[i]:.3f}", fill=(255, 255, 255), font=font)
            d.text((4, HEADER + 2), "agent view", fill=(255, 255, 0), font=font)
            d.text((im.shape[1] // 2 + 4, HEADER + 2), "wrist view", fill=(255, 255, 0), font=font)
            out.append(np.asarray(canvas))
            continue
        d.text((4, 2), f"{arm} | task {task} | init {init} | eval seed {seed} (env seed {seed + init})",
               fill=(255, 255, 255), font=font)
        state = "SUCCESS" if succ[: i + 1].any() else ("FAIL (timeout)" if i == T - 1 and not success else "")
        d.text((4, 18), f"step {i:3d}/{T}  Phi {phi[i]:.3f}  {state}", fill=(255, 255, 255), font=font)
        past = [e for e in events if e["step"] <= i]
        recent = [e for e in past if i - e["step"] < EVENT_SHOW]
        if recent:
            e = recent[-1]
            txt, col = f"EVENT @{e['step']}: {e['type']} {e['obj']}", (255, 80, 80)
        else:
            txt, col = f"events so far: {len(past)}", (180, 180, 180)
        d.text((4, 34), txt, fill=col, font=font)
        d.text((4, HEADER + 2), "agent view", fill=(255, 255, 0), font=font)
        d.text((im.shape[1] // 2 + 4, HEADER + 2), "wrist view", fill=(255, 255, 0), font=font)
        out.append(np.asarray(canvas))
    return out


def original(cfg: dict, arm: str, seed: int, task: int, k: int) -> dict | None:
    root = (cfg.get("originals") or {}).get(arm, {}).get(seed)
    if root is None:
        return None
    p = Path(str(root).format(task=task)).expanduser() / f"ep{k:03d}.meta.json"
    return json.load(open(p)) if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--only", default=None, help="comma-separated arm names to run")
    ap.add_argument("--blind", action="store_true", help="header: step and Phi only; no labels / events in episodes csv")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    out = Path(cfg["output_dir"]).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    bs = cfg["batch_size"]
    only = set(args.only.split(",")) if args.only else None
    groups: dict[tuple[int, int], set[int]] = {}
    for e in cfg["episodes"]:
        groups.setdefault((e["seed"], e["task"]), set()).add(e["init"])
    for arm in cfg["arms"]:
        name = arm["name"]
        if only and name not in only:
            continue
        chk_path, ep_path = out / f"replay_check_{name}.csv", out / f"episodes_{name}.csv"
        if chk_path.exists():
            print(f"[skip] {name}: {chk_path} exists (delete it to re-run)")
            continue
        path = arm["path"]
        policy = load_policy(str(Path(path).expanduser()) if path.startswith(("~", "/")) else path,
                             cfg["n_action_steps"], expert_fp32=arm.get("expert_fp32", False))
        chk, sel = [], []
        for (seed, task), wanted in sorted(groups.items()):
            rdir = out / "raw" / name / f"s{seed}" / f"task{task}"
            rdir.mkdir(parents=True, exist_ok=True)
            set_seed(seed)
            t0 = time.time()
            for b0 in range(0, max(wanted) + 1, bs):
                ks = list(range(b0, min(b0 + bs, cfg["n_episodes"])))
                rec_frames = any(k in wanted for k in ks)
                vec = make_vec_env(task, len(ks), log_sim_state=False)
                task_text = vec.call("task_description")[0]
                res = run_batch(vec, *policy, task_text=task_text, seeds=[seed + k for k in ks], init_ids=ks,
                                task_suffix=arm.get("task_suffix", ""), record_frames=rec_frames)
                vec.close()
                for k, rec in zip(ks, res):
                    o = original(cfg, name, seed, task, k)
                    chk.append({"arm": name, "seed": seed, "task": task, "init": k, "selected": k in wanted,
                                "success_replay": rec["success"], "length_replay": rec["length"],
                                "success_orig": "" if o is None else o["success"],
                                "length_orig": "" if o is None else o["length"],
                                "match": "" if o is None else (o["success"] == rec["success"]
                                                               and o["length"] == rec["length"])})
                    if k in wanted:
                        ev = detect_events(rec["log"], rec["actions"], rec["success"])
                        lab = label_episode(rec["log"], rec["actions"], rec["success"])
                        fn = f"{name}_t{task}_init{k}_s{seed}_{'success' if rec['success'] else 'fail'}.mp4"
                        write_video(str(out / fn), annotate(rec["frames"], rec["log"], ev, arm=arm["label"],
                                                            task=task, init=k, seed=seed,
                                                            success=rec["success"], blind=args.blind), fps=FPS)
                        sel.append({"arm": name, "seed": seed, "task": task, "init": k, "file": fn,
                                    "success": rec["success"], "length": rec["length"],
                                    "primary": "" if args.blind else lab["primary"],
                                    "subtask_completion": "" if args.blind else lab["subtask_completion"],
                                    "events": "" if args.blind else "; ".join(f"{e['type']}@{e['step']} {e['obj']}".strip()
                                                                             for e in ev)})
                    save_episode(rdir / f"ep{k:03d}", rec, video=False)
            g = [r for r in chk if r["seed"] == seed and r["task"] == task]
            print(f"[ok] {name} seed {seed} task {task}: {sum(r['success_replay'] for r in g)}/{len(g)} replayed"
                  f" succeeded, {time.time() - t0:.0f}s ({(time.time() - t0) / len(g):.1f}s/ep)", flush=True)
        for p, rows in ((chk_path, chk), (ep_path, sel)):
            with open(p, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
        bad = [r for r in chk if r["match"] is False]
        print(f"[check] {name}: {len(bad)} / {sum(r['match'] != '' for r in chk)} replayed episodes differ from the original"
              + "".join(f"\n  seed {r['seed']} task {r['task']} init {r['init']}: orig {r['success_orig']}/{r['length_orig']}"
                        f" replay {r['success_replay']}/{r['length_replay']}" for r in bad), flush=True)
        del policy
        torch.cuda.empty_cache()
    print("[done]")


if __name__ == "__main__":
    main()
