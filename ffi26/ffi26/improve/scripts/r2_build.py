"""Round-2 study: build the training dataset of round R and its per-frame side table.

  1. merge_datasets([libero_10 demos (plain text), rollouts round 1..R]) -> cfg.build.train_out;
     state/action stats overwritten with the full lerobot/libero stats (= the checkpoint's
     normalizer stats; same reason as phase2_build_dataset.py).
  2. side table r2_frames.parquet, one row per frame, keyed by the merged global `index`:
       src (demo/rollout), task_id (libero_10 id), round, label (positive/negative/none),
       label_code (1/2/0), adv, w_a1, w_awr.
     Demo labels: DEMO_POS_FRAC of the TARGET-task demo episodes (seeded) -> positive, all
     others none. Rollout labels: from r2_collect.py labels.parquet (computed within each round).
  3. demo share ~50% of frames: all target-task demos + a stratified random subset of the other
     8 tasks' demos until demo frames >= rollout frames -> train_episodes.json (--dataset.episodes).
  4. weights. w_a1 = round decay (0.5^(R-round) for rollouts, 1 for demos) x pos/neg rebalancing:
     if the weighted pos share among labelled rollout frames is outside [0.3, 0.7], the majority
     label is down-weighted to exactly 70/30 (equivalent in expectation to subsampling it).
     w_awr = round decay x clip(exp(A / beta), 0, AWR_CLIP), beta = std(A) over rollout frames; demos 1.
  5. label stats (frames and segments; by source, task, round) -> stdout and <stats_dir>/label_stats.*

    python scripts/r2_build.py --config ~/ffi26/results/round2/configs/round1.yaml
"""

import argparse
import glob
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from lerobot.datasets.dataset_tools import merge_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset

DEMO_POS_FRAC = 0.2
AWR_CLIP = 5.0
CODES = {"none": 0, "positive": 1, "negative": 2}


def read_data(root: Path, cols) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob(str(root / "data/*/*.parquet")))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    b = cfg["build"]
    R = int(cfg["round"])
    seed = int(b.get("seed", 1000))
    rng = np.random.default_rng(seed)
    texts = b["demos_task_texts"]
    tid_of = {t: i for i, t in enumerate(texts)}
    targets = {tid_of[t] for t in b["target_task_texts"]}
    out = Path(b["train_out"]).expanduser()
    stats_dir = Path(b["stats_dir"]).expanduser()
    stats_dir.mkdir(parents=True, exist_ok=True)

    demos = LeRobotDataset(b["demos_repo_id"], root=Path(b["demos_out"]).expanduser())
    rounds = b["rollout_rounds"]  # [{round, dataset_dir, repo_id, output_dir}]
    rolls = [LeRobotDataset(r["repo_id"], root=Path(r["dataset_dir"]).expanduser()) for r in rounds]
    if out.exists():
        shutil.rmtree(out)
    merge_datasets([demos, *rolls], output_repo_id=b["train_repo_id"], output_dir=out)
    sp = out / "meta/stats.json"
    st = json.load(open(sp))
    full = json.load(open(Path(b["demo_root"]).expanduser() / "meta/stats.json"))
    for k in ("observation.state", "action"):
        st[k] = full[k]
    json.dump(st, open(sp, "w"), indent=1)

    # merged episode -> (source, round, local episode); merge keeps dataset order (checked via lengths)
    src_eps = []
    for ds, tag, rnd in [(demos, "demo", 0), *[(d, "rollout", int(r["round"])) for d, r in zip(rolls, rounds)]]:
        lens = read_data(Path(ds.root), ["episode_index"]).groupby("episode_index").size().sort_index()
        src_eps += [(tag, rnd, int(e), int(n)) for e, n in lens.items()]
    data = read_data(out, ["index", "episode_index", "frame_index", "task_index"])
    tasks = pd.read_parquet(out / "meta/tasks.parquet")
    task_text = {int(v): k for k, v in tasks["task_index"].items()}
    mlens = data.groupby("episode_index").size().sort_index()
    assert len(mlens) == len(src_eps) and all(int(n) == s[3] for n, s in zip(mlens.values, src_eps)), \
        "merged episode order/lengths differ from the source datasets"
    emap = pd.DataFrame(src_eps, columns=["src", "round", "local_ep", "length"])
    emap["episode_index"] = mlens.index.values
    data = data.merge(emap[["episode_index", "src", "round", "local_ep"]], on="episode_index")
    data["task_id"] = data.task_index.map(lambda i: tid_of[task_text[int(i)]])

    # rollout labels
    lab = []
    for r in rounds:
        l = pd.read_parquet(Path(r["output_dir"]).expanduser() / "labels.parquet")
        l = l.rename(columns={"ds_episode": "local_ep"})[["local_ep", "frame_index", "round", "label", "adv", "group", "stage"]]
        lab.append(l)
    lab = pd.concat(lab)
    data = data.merge(lab, on=["local_ep", "frame_index", "round"], how="left")
    roll = data.src == "rollout"
    assert data.loc[roll, "label"].notna().all(), "rollout frames without a label"
    # demo labels: fixed per episode
    demo_eps = emap[emap.src == "demo"].episode_index.values
    ep_task = data.groupby("episode_index").task_id.first()
    tgt_demo_eps = [e for e in demo_eps if ep_task[e] in targets]
    pos_eps = set(rng.choice(tgt_demo_eps, size=int(round(DEMO_POS_FRAC * len(tgt_demo_eps))), replace=False).tolist())
    data.loc[~roll, "label"] = np.where(data.loc[~roll, "episode_index"].isin(pos_eps), "positive", "none")
    data.loc[~roll, "adv"] = 0.0
    data["label_code"] = data.label.map(CODES).astype(np.int8)

    # demo selection for ~50% demo frames
    n_roll = int(roll.sum())
    chosen = list(tgt_demo_eps)
    n_demo = int(emap.set_index("episode_index").loc[chosen, "length"].sum())
    others = {t: list(rng.permutation([e for e in demo_eps if ep_task[e] == t])) for t in range(10) if t not in targets}
    while n_demo < n_roll and any(others.values()):
        for t in sorted(others):
            if others[t] and n_demo < n_roll:
                e = int(others[t].pop())
                chosen.append(e)
                n_demo += int(emap.set_index("episode_index").loc[e, "length"])
    roll_eps = emap[emap.src == "rollout"].episode_index.tolist()
    train_eps = sorted(int(e) for e in chosen + roll_eps)
    json.dump(train_eps, open(out / "train_episodes.json", "w"))
    data["in_train"] = data.episode_index.isin(train_eps)

    # weights
    decay = np.where(roll, 0.5 ** (R - data["round"].fillna(R)), 1.0)
    tr = data.in_train & roll
    pos, neg = (data.label == "positive") & tr, (data.label == "negative") & tr
    wp, wn = decay[pos].sum(), decay[neg].sum()
    share = wp / max(wp + wn, 1e-9)
    w_a1 = decay.copy()
    rebalance = "none"
    if share > 0.7:
        w_a1[pos] *= 0.7 * wn / (0.3 * wp)
        rebalance = f"positive down-weighted x{0.7 * wn / (0.3 * wp):.3f}"
    elif share < 0.3:
        w_a1[neg] *= 0.7 * wp / (0.3 * wn)
        rebalance = f"negative down-weighted x{0.7 * wp / (0.3 * wn):.3f}"
    data["w_a1"] = w_a1
    beta = float(data.loc[tr, "adv"].std())
    w_awr = np.ones(len(data))
    w_awr[roll] = np.clip(np.exp(data.loc[roll, "adv"].to_numpy() / beta), 0, AWR_CLIP) * decay[roll]
    data["w_awr"] = w_awr
    data.to_parquet(out / "r2_frames.parquet")

    # stats
    t = data[data.in_train]
    fr = t.groupby(["src", "task_id", "round", "label"]).size().unstack("label", fill_value=0)
    seg = (t[t.src == "rollout"].drop_duplicates(["round", "local_ep", "stage"])
           .groupby(["task_id", "round", "label"]).size().unstack("label", fill_value=0))
    by_src = t.groupby(["src", "label"]).size().unstack("label", fill_value=0)
    info = {
        "round": R, "train_episodes": len(train_eps), "train_frames": int(len(t)),
        "demo_frames": int((t.src == "demo").sum()), "rollout_frames": int((t.src == "rollout").sum()),
        "demo_frame_share": float((t.src == "demo").mean()),
        "rollout_pos_share_weighted_before": float(share), "rebalance": rebalance, "awr_beta": beta,
        "demo_pos_episodes": sorted(int(e) for e in pos_eps), "seed": seed,
    }
    json.dump(info, open(out / "BUILD_INFO.json", "w"), indent=1)
    json.dump(info, open(stats_dir / "build_info.json", "w"), indent=1)
    fr.to_csv(stats_dir / "label_stats_frames.csv")
    seg.to_csv(stats_dir / "label_stats_segments.csv")
    print(json.dumps(info, indent=1))
    print("\nframes by source x label:\n", by_src.to_string())
    print("\nframes by source / task / round x label:\n", fr.to_string())
    print("\nrollout segments by task / round x label:\n", seg.to_string())
    print(f"\nNote: during A1 training {cfg.get('cfg_drop', 0.3):.0%} of rollout samples additionally lose the suffix (CFG dropout).")


if __name__ == "__main__":
    main()
