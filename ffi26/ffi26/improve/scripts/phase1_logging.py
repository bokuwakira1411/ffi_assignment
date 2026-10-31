"""Phase 1: logged baseline rollouts on target tasks (predicates, physics, sim states).

Episode k of a task uses init state k and env seed seed+k, the same mapping as
lerobot-eval with --seed=seed (lerobot_eval.py:549, envs/libero.py:173-175,345).
Outputs per episode: <out>/task{t}/ep{k:03d}.npz (+ .mp4 agentview video), and
<out>/episodes.csv with one row per episode.

    python scripts/phase1_logging.py --config configs/phase1.yaml
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

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from lerobot.utils.io_utils import write_video  # noqa: E402
from lerobot.utils.random_utils import set_seed  # noqa: E402

from rollout import load_policy, make_vec_env, run_batch  # noqa: E402


def save_episode(path: Path, rec: dict, video: bool):
    log = dict(rec["log"])
    meta = log.pop("meta")
    names = log.pop("obj_names")
    np.savez_compressed(
        path.with_suffix(".npz"),
        actions=rec["actions"],
        **{k: v for k, v in log.items()},
    )
    with open(path.with_suffix(".meta.json"), "w") as f:
        json.dump(
            {**meta, "obj_names": names, "success": rec["success"], "length": rec["length"],
             "seed": rec["seed"], "init_id": rec["init_id"], "task_string": rec["task"]},
            f,
        )
    if video and "frames" in rec:
        write_video(str(path.with_suffix(".mp4")), rec["frames"]["image"], fps=20)


def load_episode(npz_path: Path) -> tuple[dict, np.ndarray, dict]:
    d = dict(np.load(npz_path, allow_pickle=False))
    meta = json.load(open(str(npz_path).replace(".npz", ".meta.json")))
    actions = d.pop("actions")
    log = {**d, "meta": meta, "obj_names": meta["obj_names"]}
    return log, actions, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    out = Path(cfg["output_dir"]).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    set_seed(cfg["seed"])
    policy = load_policy(cfg["policy"], cfg["n_action_steps"])
    bs = cfg["batch_size"]
    t_all = time.time()
    for task_id in cfg["tasks"]:
        tdir = out / f"task{task_id}"
        tdir.mkdir(exist_ok=True)
        eps = list(range(cfg["n_episodes"]))
        for b0 in range(0, len(eps), bs):
            ks = eps[b0 : b0 + bs]
            if all((tdir / f"ep{k:03d}.npz").exists() for k in ks):
                print(f"[skip] task {task_id} eps {ks}")
                continue
            vec = make_vec_env(task_id, len(ks))
            task_text = vec.call("task_description")[0]
            t0 = time.time()
            res = run_batch(
                vec, *policy, task_text=task_text,
                seeds=[cfg["seed"] + k for k in ks], init_ids=ks,
                record_frames=cfg.get("save_video", True),
            )
            vec.close()
            dt = time.time() - t0
            for k, rec in zip(ks, res):
                save_episode(tdir / f"ep{k:03d}", rec, cfg.get("save_video", True))
            print(f"[ok] task {task_id} eps {ks[0]}-{ks[-1]} succ={sum(r['success'] for r in res)}/{len(ks)} "
                  f"{dt:.0f}s ({dt/len(ks):.1f}s/ep)", flush=True)
    print(f"[done] total {time.time()-t_all:.0f}s")


if __name__ == "__main__":
    main()
