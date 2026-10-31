"""Phase 2: build the single training dataset = target-task demos (labelled positive) + rollouts.

lerobot 0.6.2 has MultiLeRobotDataset disabled, so demos and rollouts are merged into one
LeRobot dataset:
  1. demos: target-task episodes of lerobot/libero with " Advantage: positive" appended,
     written by scripts/phase2_demos_subset.py (delete_episodes fails on this hub copy);
  2. merge with the rollout dataset written by phase2_collect.py (merge_datasets);
  3. overwrite observation.state / action stats with the full lerobot/libero stats, which are
     the stats in the checkpoint's normalizer (max abs diff ~4e-8). lerobot-train replaces the
     normalizer stats with dataset.meta.stats when fine-tuning (lerobot_train.py:678-704), so
     without this the merged subset would change the normalization.

    python scripts/phase2_build_dataset.py --config configs/phase2_r1.yaml
"""

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
import yaml
from lerobot.datasets.dataset_tools import merge_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset

POS = " Advantage: positive"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    bcfg = cfg["build"]
    src_root = Path(bcfg["demo_root"]).expanduser()
    demos_root = Path(bcfg["demos_out"]).expanduser()
    merged_root = Path(bcfg["merged_out"]).expanduser()
    roll_root = Path(cfg["dataset_dir"]).expanduser()

    if not demos_root.exists():
        raise SystemExit(f"{demos_root} missing: run scripts/phase2_demos_subset.py first")
    demos = LeRobotDataset("local/ffi26_demos_targets", root=demos_root)
    roll_specs = bcfg.get("rollout_datasets") or [{"repo_id": cfg["dataset_repo_id"], "root": str(roll_root)}]
    rolls = [LeRobotDataset(r["repo_id"], root=Path(r["root"]).expanduser()) for r in roll_specs]
    n_roll_eps = sum(r.meta.total_episodes for r in rolls)
    n_roll_frames = sum(r.meta.total_frames for r in rolls)
    print(f"demos: {demos.meta.total_episodes} eps {demos.meta.total_frames} frames | "
          f"rollouts ({len(rolls)} datasets): {n_roll_eps} eps {n_roll_frames} frames", flush=True)

    if merged_root.exists():
        shutil.rmtree(merged_root)
    merged = merge_datasets([demos, *rolls], output_repo_id=bcfg["merged_repo_id"], output_dir=merged_root)

    stats_path = merged_root / "meta/stats.json"
    stats = json.load(open(stats_path))
    full = json.load(open(src_root / "meta/stats.json"))
    for k in ("observation.state", "action"):
        stats[k] = full[k]
    json.dump(stats, open(stats_path, "w"), indent=1)

    merged = LeRobotDataset(bcfg["merged_repo_id"], root=merged_root)
    t = pd.read_parquet(merged_root / "meta/tasks.parquet")
    info = {
        "demo_episodes": demos.meta.total_episodes, "demo_frames": demos.meta.total_frames,
        "rollout_datasets": [r["root"] for r in roll_specs],
        "rollout_episodes": n_roll_eps, "rollout_frames": n_roll_frames,
        "merged_episodes": merged.meta.total_episodes, "merged_frames": merged.meta.total_frames,
        "rollout_frame_fraction": n_roll_frames / merged.meta.total_frames,
        "tasks": t.index.tolist(),
        "stats_source": "state/action stats copied from full lerobot/libero meta/stats.json",
    }
    json.dump(info, open(merged_root / "BUILD_INFO.json", "w"), indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
