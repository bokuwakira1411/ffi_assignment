"""Phase 3: evaluate several policies on the SAME init states / env seeds with predicate logging.

Episode k of a task uses init state k and env seed seed+k for every model (as in Phase 1 and
lerobot-eval --seed=seed). Policy sampling noise comes from the global torch RNG, reset with
set_seed(seed) at the start of each (model, task), so models see the same RNG stream too.

    python scripts/phase3_eval.py --config configs/phase3_r1.yaml
Outputs: <output_dir>/<model>/task{t}/ep{k}.npz|.meta.json
"""

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
import yaml  # noqa: E402
from lerobot.utils.random_utils import set_seed  # noqa: E402

from phase1_logging import save_episode  # noqa: E402
from rollout import load_policy, make_vec_env, run_batch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--only", default=None, help="comma-separated model names to run")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    out = Path(cfg["output_dir"]).expanduser()
    bs = cfg["batch_size"]
    only = set(args.only.split(",")) if args.only else None
    for m in cfg["models"]:
        if only and m["name"] not in only:
            continue
        policy = load_policy(str(Path(m["path"]).expanduser()) if m["path"].startswith(("~", "/")) else m["path"],
                             cfg["n_action_steps"], adv_label=m.get("adv_label"))
        for job in m["jobs"]:
            for task_id in job["tasks"]:
                tdir = out / m["name"] / f"task{task_id}"
                tdir.mkdir(parents=True, exist_ok=True)
                eps = list(range(job["n_episodes"]))
                if all((tdir / f"ep{k:03d}.npz").exists() for k in eps):
                    print(f"[skip] {m['name']} task {task_id}")
                    continue
                set_seed(cfg["seed"])
                t0 = time.time()
                n_succ = 0
                for b0 in range(0, len(eps), bs):
                    ks = eps[b0 : b0 + bs]
                    vec = make_vec_env(task_id, len(ks), log_sim_state=False)
                    task_text = vec.call("task_description")[0]
                    res = run_batch(vec, *policy, task_text=task_text, seeds=[cfg["seed"] + k for k in ks],
                                    init_ids=ks, task_suffix=m.get("task_suffix", ""))
                    vec.close()
                    for k, rec in zip(ks, res):
                        save_episode(tdir / f"ep{k:03d}", rec, video=False)
                    n_succ += sum(r["success"] for r in res)
                dt = time.time() - t0
                print(f"[ok] {m['name']} task {task_id}: {n_succ}/{len(eps)} {dt:.0f}s ({dt/len(eps):.1f}s/ep)",
                      flush=True)
        del policy
        torch.cuda.empty_cache()
    print("[done]")


if __name__ == "__main__":
    main()
