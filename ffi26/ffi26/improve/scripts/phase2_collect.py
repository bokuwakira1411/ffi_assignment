"""Phase 2 (A.1) step 1-3: collect G rollouts per group, compute group-relative advantages,
label them, and write a LeRobot-format dataset of executed actions.

A group = (task, start state). One batch of G sub-envs = one group, so every rollout in a group
starts from the same init state (or the same saved mid-episode sim state when reset-to-failure
is on) and differs only by env seed and policy sampling noise.

    python scripts/phase2_collect.py --config configs/phase2_r1.yaml
Outputs (cfg.output_dir): groups.csv, rollouts.csv, logs/task{t}_g{gid}_r{j}.npz (+ .meta.json),
and the dataset at cfg.dataset_dir.
"""

import argparse
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
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.utils.random_utils import set_seed  # noqa: E402

from failure_labels import label_episode  # noqa: E402
from phase1_logging import load_episode, save_episode  # noqa: E402
from reward import RewardConfig, episode_return  # noqa: E402
from rollout import load_policy, make_vec_env, run_batch  # noqa: E402

POS, NEG = " Advantage: positive", " Advantage: negative"
FEATURES = {  # same as lerobot/libero meta/info.json (video, 256x256x3; state 8; action 7)
    "observation.images.image": {"dtype": "video", "shape": [256, 256, 3], "names": ["height", "width", "channel"]},
    "observation.images.image2": {"dtype": "video", "shape": [256, 256, 3], "names": ["height", "width", "channel"]},
    "observation.state": {"dtype": "float32", "shape": [8], "names": ["state"]},
    "action": {"dtype": "float32", "shape": [7], "names": ["actions"]},
}


def failure_start_state(logged_dir: Path, task_id: int, init_id: int, offset: int):
    """Sim state right after the first goal predicate (False at reset) becomes True in the
    Phase 1 episode with this init state. Returns (state, step) or (None, 0)."""
    p = logged_dir / f"task{task_id}" / f"ep{init_id:03d}.npz"
    if not p.exists():
        return None, 0
    log, _, _ = load_episode(p)
    preds = log["preds"].astype(bool)
    new = ~preds[0]
    hit = np.where((preds[:, new]).any(axis=1))[0]
    if len(hit) == 0 or "sim_state" not in log:
        return None, 0
    t = min(int(hit[0]) + offset, len(log["sim_state"]) - 1)
    if log["success"][t]:
        return None, 0
    return log["sim_state"][t], t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    out = Path(cfg["output_dir"]).expanduser()
    (out / "logs").mkdir(parents=True, exist_ok=True)
    rcfg = RewardConfig(**cfg["reward"])
    set_seed(cfg["seed"])
    pol = cfg["policy"]
    policy = load_policy(str(Path(pol).expanduser()) if pol.startswith(("~", "/")) else pol, cfg["n_action_steps"])
    G = cfg["group_size"]
    ds = LeRobotDataset.create(
        repo_id=cfg["dataset_repo_id"], fps=10, features=FEATURES,
        root=Path(cfg["dataset_dir"]).expanduser(), robot_type="panda", use_videos=True,
    )
    group_rows, roll_rows = [], []
    gid = 0
    t_all = time.time()
    for tcfg in cfg["tasks"]:
        task_id = tcfg["task_id"]
        vec = make_vec_env(task_id, G)
        task_text = vec.call("task_description")[0]
        for init_id in tcfg["init_ids"]:
            sim_state, t0_step = None, 0
            if tcfg.get("reset_to_failure", False):
                sim_state, t0_step = failure_start_state(
                    Path(cfg["phase1_logged_dir"]).expanduser(), task_id, init_id, cfg.get("failure_offset", 0))
            seeds = [cfg["seed"] + 1000 * gid + j for j in range(G)]
            t0 = time.time()
            res = run_batch(
                vec, *policy, task_text=task_text, seeds=seeds, init_ids=[init_id] * G,
                sim_states=[sim_state] * G if sim_state is not None else None,
                max_steps=520 - t0_step, noise_scale=cfg.get("noise_scale", 1.0), record_frames=True,
                task_suffix=cfg.get("collect_task_suffix", ""),
            )
            dt = time.time() - t0
            returns = np.array([episode_return(r["log"], rcfg) for r in res])
            spread = float(returns.max() - returns.min())
            rejected = spread < cfg.get("min_return_spread", 1e-6)
            adv = (returns - returns.mean()) / (returns.std() + 1e-6)
            n_succ = int(sum(r["success"] for r in res))
            group_rows.append({
                "group": gid, "task_id": task_id, "init_id": init_id, "reset_to_failure": sim_state is not None,
                "start_step": t0_step, "n_success": n_succ, "G": G, "return_mean": returns.mean(),
                "return_std": returns.std(), "return_spread": spread, "rejected": rejected, "seconds": dt,
            })
            for j, r in enumerate(res):
                lab = label_episode(r["log"], r["actions"], r["success"])
                positive = bool(adv[j] > cfg.get("adv_threshold", 0.0))
                name = f"task{task_id}_g{gid:02d}_r{j}"
                rec = {**r, "frames": r["frames"]}
                save_episode(out / "logs" / name, {k: v for k, v in rec.items() if k != "frames"}, video=False)
                in_ds = (not rejected) and r["length"] > 0
                roll_rows.append({
                    "name": name, "group": gid, "task_id": task_id, "init_id": init_id, "seed": r["seed"],
                    "success": r["success"], "length": r["length"], "return": returns[j], "advantage": adv[j],
                    "label": "positive" if positive else "negative", "in_dataset": in_ds,
                    "ds_episode": ds.meta.total_episodes if in_ds else -1,
                    "primary": lab["primary"], "phi_T": lab["phi_T"],
                })
                if in_ds:
                    task_str = task_text + (POS if positive else NEG)
                    fr = r["frames"]
                    for k in range(r["length"]):
                        ds.add_frame({
                            "observation.images.image": fr["image"][k],
                            "observation.images.image2": fr["image2"][k],
                            "observation.state": fr["state"][k],
                            "action": r["actions"][k],
                            "task": task_str,
                        })
                    ds.save_episode()
            print(f"[group {gid}] task {task_id} init {init_id} rtf={sim_state is not None} start={t0_step} "
                  f"succ={n_succ}/{G} spread={spread:.3f} rejected={rejected} {dt:.0f}s", flush=True)
            gid += 1
        vec.close()
    ds.finalize()
    pd.DataFrame(group_rows).to_csv(out / "groups.csv", index=False)
    pd.DataFrame(roll_rows).to_csv(out / "rollouts.csv", index=False)
    json.dump({"config": cfg, "seconds": time.time() - t_all}, open(out / "collect_info.json", "w"), indent=1)
    print(f"[done] {time.time()-t_all:.0f}s, dataset episodes={ds.meta.total_episodes} frames={ds.meta.total_frames}")


if __name__ == "__main__":
    main()
