"""Round-2 study: collect G rollouts per group with the CURRENT checkpoint, compute reward_r2
returns, per-segment group-relative advantages and labels, and write a LeRobot dataset.

Differences from phase2_collect.py (round-1 tool, kept unchanged):
  * reward / labels from src/reward_r2.py (shaping + failure costs, per-SEGMENT labels);
  * the dataset stores the PLAIN task text; labels live in a per-frame side table
    (labels.parquet: ds_episode, frame_index, label, adv, ...). The training wrapper
    (scripts/train_r2.py) appends the suffix per sample, so CFG dropout / rebalancing / round
    weights / AWR can change without rewriting videos;
  * groups can also start from saved sim states (reset-to-failure, cfg.rtf_groups);
  * full sim states are kept in every rollout log (for reset-to-failure in the next round).

    python scripts/r2_collect.py --config ~/ffi26/results/round2/configs/round1.yaml
Outputs (cfg.output_dir): groups.csv, rollouts.csv, segments.csv, labels.parquet, events.csv,
logs/<name>.npz|.meta.json, collect_info.json; dataset at cfg.dataset_dir.
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
import torch  # noqa: E402
import yaml  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.utils.random_utils import set_seed  # noqa: E402

from failure_labels import label_episode  # noqa: E402
from phase1_logging import load_episode, save_episode  # noqa: E402
from phase2_collect import FEATURES  # noqa: E402
from reward_r2 import RewardR2Config, group_segment_labels  # noqa: E402
from rollout import MAX_STEPS, load_policy, make_vec_env, run_batch  # noqa: E402


def expand(p: str) -> str:
    return str(Path(p).expanduser()) if p.startswith(("~", "/")) else p


def group_specs(cfg: dict) -> list[dict]:
    """Init-state groups first, then reset-to-failure groups. RTF spec:
    {task_id, from_log: <path to .npz with sim_state>, step: <row>, init_id, note}."""
    specs = []
    for t in cfg["tasks"]:
        for k in t["init_ids"]:
            specs.append({"task_id": t["task_id"], "init_id": int(k), "kind": "init"})
    for g in cfg.get("rtf_groups") or []:
        specs.append({**g, "kind": "rtf"})
    return specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max_groups", type=int, default=None, help="smoke test: stop after n groups")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    out = Path(cfg["output_dir"]).expanduser()
    (out / "logs").mkdir(parents=True, exist_ok=True)
    rcfg = RewardR2Config(**cfg["reward"])
    set_seed(cfg["seed"])
    policy = load_policy(expand(cfg["policy"]), cfg["n_action_steps"])
    G = cfg["group_size"]
    rnd = int(cfg["round"])
    ds = LeRobotDataset.create(repo_id=cfg["dataset_repo_id"], fps=10, features=FEATURES,
                               root=Path(cfg["dataset_dir"]).expanduser(), robot_type="panda", use_videos=True)
    specs = group_specs(cfg)[: args.max_groups]
    group_rows, roll_rows, seg_rows, lab_rows, ev_rows = [], [], [], [], []
    t_all = time.time()
    t_roll = t_enc = 0.0
    vec, vec_task = None, None
    for gid, sp in enumerate(specs):
        task_id = sp["task_id"]
        if vec_task != task_id:
            if vec is not None:
                vec.close()
            vec, vec_task = make_vec_env(task_id, G), task_id
            task_text = vec.call("task_description")[0]
        sim_state, start_step = None, 0
        if sp["kind"] == "rtf":
            log0, _, _ = load_episode(Path(sp["from_log"]).expanduser())
            start_step = int(sp["step"])
            sim_state = log0["sim_state"][start_step]
        seeds = [cfg["seed"] + 1000 * gid + j for j in range(G)]
        t0 = time.time()
        res = run_batch(vec, *policy, task_text=task_text, seeds=seeds, init_ids=[sp["init_id"]] * G,
                        sim_states=[sim_state] * G if sim_state is not None else None,
                        max_steps=MAX_STEPS - start_step, noise_scale=cfg.get("noise_scale", 1.0),
                        record_frames=True, task_suffix=cfg.get("collect_task_suffix", ""))
        t_roll += time.time() - t0
        lab = group_segment_labels(res, rcfg, cfg.get("min_return_spread", 0.1), cfg.get("adv_deadzone", 0.02))
        n_succ = int(sum(r["success"] for r in res))
        group_rows.append({"round": rnd, "group": gid, "task_id": task_id, "kind": sp["kind"], "init_id": sp["init_id"],
                           "start_step": start_step, "from_log": sp.get("from_log", ""), "n_success": n_succ, "G": G,
                           "return_mean": lab["return_mean"], "return_spread": lab["spread"],
                           "all_success": lab["all_success"], "rejected": lab["rejected"],
                           "rollout_s": time.time() - t0})
        t1 = time.time()
        for j, (r, lr) in enumerate(zip(res, lab["rollouts"])):
            name = f"r{rnd}_task{task_id}_g{gid:02d}_j{j}"
            save_episode(out / "logs" / name, r, video=False)
            fl = label_episode(r["log"], r["actions"], r["success"])
            in_ds = (not lab["rejected"]) and r["length"] > 0
            ds_ep = ds.meta.total_episodes if in_ds else -1
            cnt = pd.Series([e["type"] for e in lr["events"]], dtype=object).value_counts().to_dict()
            roll_rows.append({"round": rnd, "name": name, "group": gid, "task_id": task_id, "kind": sp["kind"],
                              "init_id": sp["init_id"], "start_step": start_step, "seed": r["seed"],
                              "success": r["success"], "length": r["length"], "return": lr["return"],
                              "in_dataset": in_ds, "ds_episode": ds_ep, "primary": fl["primary"],
                              "phi_T": fl["phi_T"], "subtask_completion": fl["subtask_completion"],
                              **{f"n_{x}": cnt.get(x, 0) for x in ("drop", "slip", "collision", "timeout")}})
            for e in lr["events"]:
                ev_rows.append({"round": rnd, "name": name, "task_id": task_id, **e})
            for s in lr["segments"]:
                seg_rows.append({"round": rnd, "name": name, "group": gid, "task_id": task_id, "ds_episode": ds_ep, **s})
            if not in_ds:
                continue
            fr = r["frames"]
            for s in lr["segments"]:
                for k in range(s["start"], min(s["end"], r["length"])):
                    lab_rows.append((ds_ep, k, rnd, task_id, gid, s["stage"], s["label"], s["adv"]))
            for k in range(r["length"]):
                ds.add_frame({"observation.images.image": fr["image"][k], "observation.images.image2": fr["image2"][k],
                              "observation.state": fr["state"][k], "action": r["actions"][k], "task": task_text})
            ds.save_episode()
        t_enc += time.time() - t1
        print(f"[group {gid}] task {task_id} {sp['kind']} init {sp['init_id']} start={start_step} succ={n_succ}/{G} "
              f"spread={lab['spread']:.3f} rejected={lab['rejected']} roll {group_rows[-1]['rollout_s']:.0f}s "
              f"write {time.time()-t1:.0f}s", flush=True)
    vec.close()
    ds.finalize()
    pd.DataFrame(group_rows).to_csv(out / "groups.csv", index=False)
    pd.DataFrame(roll_rows).to_csv(out / "rollouts.csv", index=False)
    pd.DataFrame(seg_rows).to_csv(out / "segments.csv", index=False)
    pd.DataFrame(ev_rows).to_csv(out / "events.csv", index=False)
    pd.DataFrame(lab_rows, columns=["ds_episode", "frame_index", "round", "task_id", "group", "stage", "label", "adv"]
                 ).to_parquet(out / "labels.parquet")
    info = {"config": cfg, "seconds_total": time.time() - t_all, "seconds_rollout": t_roll, "seconds_write": t_enc,
            "torch_max_mem_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
            "dataset_episodes": ds.meta.total_episodes, "dataset_frames": ds.meta.total_frames}
    json.dump(info, open(out / "collect_info.json", "w"), indent=1)
    print(f"[done] {json.dumps({k: v for k, v in info.items() if k != 'config'})}")


if __name__ == "__main__":
    main()
