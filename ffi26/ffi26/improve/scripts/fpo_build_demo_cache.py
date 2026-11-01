"""FPO v3: cache LIBERO-10 demo samples in the rollout-decision format for the demo-replay BC term.

Source: data/r2/demos_libero10 (379 episodes of lerobot/libero, all 10 LIBERO-10 tasks; the training data of
lerobot/smolvla_libero). Every sample goes through exactly what lerobot-train does with a dataset batch
(lerobot_train._preprocess_dataset_batch: uint8 -> /255, rename, preprocessor), using the preprocessor of the
OFFICIAL checkpoint (load_policy, same object as collection / eval): rename image -> camera1, image2 -> camera2,
newline, tokenizer (max_length 48, padding max_length), normaliser (state / action MEAN_STD with the checkpoint's
stats, images IDENTITY). Action chunks = 50 consecutive actions (delta_timestamps i / fps, i < chunk_size, as
lerobot-train with SmolVLA's action_delta_indices); action_is_pad -> n_exec = number of valid steps.
Demo task_index is NOT the LIBERO-10 env task id: tasks are mapped by language string to the env task id.

Split per task by EPISODE: `probe_episodes` episodes per task -> probe file (L_CFM log on the untrained tasks),
the rest -> train file; `per_task` samples per task drawn uniformly over frames.

    python scripts/fpo_build_demo_cache.py --config ~/ffi26/results/round2/configs/fpo_v3.yaml
Outputs (<demo.cache_dir>): demo_train.npz, demo_probe.npz (keys as decisions/g*.npz: KEEP_KEYS, chunk, n_exec,
task_id, plus episode, frame), demo_cache_info.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

from fpo_capture import IMG_KEYS, KEEP_KEYS, to_uint8  # noqa: E402
from rollout import load_policy  # noqa: E402


def env_task_names() -> list[str]:
    from libero.libero import benchmark

    s = benchmark.get_benchmark_dict()["libero_10"]()
    return [s.get_task(i).language for i in range(s.n_tasks)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    dc = cfg["demo"]
    out = Path(dc["cache_dir"]).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    policy, pre, _, _, _ = load_policy(cfg["init_policy"], cfg["n_action_steps"])
    chunk = policy.config.chunk_size
    root = Path(dc["dataset_dir"]).expanduser()
    meta = LeRobotDataset(dc["repo_id"], root=root).meta
    fps = meta.fps
    ds = LeRobotDataset(dc["repo_id"], root=root, delta_timestamps={"action": [i / fps for i in range(chunk)]})

    names = env_task_names()
    tasks = pd.read_parquet(root / "meta" / "tasks.parquet")
    demo2env = {}
    for text, row in tasks.iterrows():
        assert text in names, f"demo task {text!r} not a LIBERO-10 env task"
        demo2env[int(row.task_index)] = names.index(text)
    assert sorted(demo2env.values()) == list(range(10)), demo2env

    ep = meta.episodes.to_pandas() if hasattr(meta.episodes, "to_pandas") else pd.DataFrame(meta.episodes)
    hf = ds.hf_dataset
    ep_idx = np.asarray(hf["episode_index"])
    tk_idx = np.asarray(hf["task_index"])
    rng = np.random.default_rng(dc["seed"])
    picks = {"train": [], "probe": []}
    split = {}
    for dt, et in sorted(demo2env.items(), key=lambda x: x[1]):
        eps = np.unique(ep_idx[tk_idx == dt])
        perm = rng.permutation(eps)
        probe_eps, train_eps = np.sort(perm[: dc["probe_episodes"]]), np.sort(perm[dc["probe_episodes"]:])
        split[et] = {"demo_task_index": dt, "n_episodes": len(eps), "probe_episodes": probe_eps.tolist()}
        for name, E, n in (("train", train_eps, dc["per_task"]), ("probe", probe_eps, dc["probe_per_task"])):
            rows = np.flatnonzero(np.isin(ep_idx, E))
            sel = np.sort(rng.choice(rows, min(n, len(rows)), replace=False))
            picks[name] += [(int(i), et) for i in sel]
    print(f"[demo] fps {fps}, chunk {chunk}, demo->env task {demo2env}; train {len(picks['train'])}, "
          f"probe {len(picks['probe'])}", flush=True)

    for name, P in picks.items():
        rec = {k: [] for k in (*KEEP_KEYS, "chunk", "n_exec", "task_id", "episode", "frame")}
        for s in range(0, len(P), 32):
            items = [ds[i] for i, _ in P[s:s + 32]]
            batch = {k: torch.stack([it[k] for it in items]) for k in
                     ("observation.images.image", "observation.images.image2", "observation.state", "action",
                      "action_is_pad")}
            batch["task"] = [it["task"] for it in items]
            for k in ("observation.images.image", "observation.images.image2"):
                if batch[k].dtype == torch.uint8:
                    batch[k] = batch[k].float() / 255.0
            b = pre(batch)
            for k in KEEP_KEYS:
                v = b[k]
                rec[k].append(to_uint8(v) if k in IMG_KEYS else v.cpu().numpy())
            rec["chunk"].append(b["action"].float().cpu().numpy())
            pad = b["action_is_pad"].cpu().numpy()
            valid = (~pad).sum(1)
            assert all((~pad[j, :valid[j]]).all() for j in range(len(pad))), "padding is not a tail"
            rec["n_exec"].append(valid.astype(np.int64))
            rec["task_id"].append(np.array([t for _, t in P[s:s + 32]], np.int64))
            rec["episode"].append(np.array([int(it["episode_index"]) for it in items], np.int64))
            rec["frame"].append(np.array([int(it["frame_index"]) for it in items], np.int64))
            if s % 640 == 0:
                print(f"[demo] {name} {s}/{len(P)} {time.time() - t0:.0f}s", flush=True)
        arr = {k: np.concatenate(v) for k, v in rec.items()}
        arr["observation.language.attention_mask"] = arr["observation.language.attention_mask"].astype(bool)
        np.savez(out / f"demo_{name}.npz", **arr)
        print(f"[demo] wrote {out / f'demo_{name}.npz'}: " + ", ".join(f"{k} {v.shape}" for k, v in arr.items()),
              flush=True)
    info = {"dataset_dir": str(root), "repo_id": dc["repo_id"], "fps": fps, "chunk_size": chunk,
            "preprocessor_of": cfg["init_policy"], "demo_task_index_to_env_task": demo2env, "split": split,
            "n_train": len(picks["train"]), "n_probe": len(picks["probe"]), "seed": dc["seed"],
            "seconds": time.time() - t0}
    json.dump(info, open(out / "demo_cache_info.json", "w"), indent=1)
    print(f"[done] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
