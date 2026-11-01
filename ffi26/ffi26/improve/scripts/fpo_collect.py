"""FPO iteration k, step 1-2: on-policy collection with the CURRENT checkpoint + advantages.

Same groups / reward / segment advantages as the round-2 study (r2_collect.py, reward_r2.py), but
no LeRobot dataset and no video: every chunk decision is captured (src/fpo_capture.py) and saved
with its advantage, one file per group:

  <out>/decisions/g{gid:02d}.npz : model inputs (uint8 images, normalised state, tokens, mask),
      chunk (D, 50, 7, normalised), env, step, n_exec, noise_seed, adv, label_code, group, task_id
  <out>/logs/<name>.npz|.meta.json (predicate logs, no sim state), groups.csv, rollouts.csv,
  segments.csv, collect_info.json

Advantage of a decision = advantage of the segment that contains its first step. Groups rejected by
the spread rule (< min_return_spread, not all-success) are saved with adv = 0 and flag rejected=1.

    python scripts/fpo_collect.py --config ~/ffi26/results/round2/configs/fpo.yaml --iter 0 \
        --policy lerobot/smolvla_libero --out outputs/fpo/iter0 [--max_groups 2]
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

from failure_labels import label_episode  # noqa: E402
from fpo_capture import ChunkCapture  # noqa: E402
from phase1_logging import save_episode  # noqa: E402
from reward_r2 import RewardR2Config, group_segment_labels  # noqa: E402
from rollout import load_policy, make_vec_env, run_batch  # noqa: E402

CODES = {"none": 0, "positive": 1, "negative": 2}


def expand(p: str) -> str:
    return str(Path(p).expanduser()) if p.startswith(("~", "/")) else p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--iter", type=int, required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_groups", type=int, default=None, help="smoke test: stop after n groups")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    c = cfg["collect"]
    out = Path(args.out).expanduser()
    (out / "logs").mkdir(parents=True, exist_ok=True)
    (out / "decisions").mkdir(parents=True, exist_ok=True)
    rcfg = RewardR2Config(**cfg["reward"])
    G = c["group_size"]
    seed0 = c["seed"] + c["seed_iter_stride"] * args.iter  # env seed of rollout j in group g = seed0 + 1000 g + j
    torch.manual_seed(seed0)
    policy = load_policy(expand(args.policy), cfg["n_action_steps"], expert_fp32=cfg.get("expert_fp32", False))
    specs = [(t["task_id"], int(k)) for t in c["tasks"] for k in t["init_ids"]][: args.max_groups]

    group_rows, roll_rows, seg_rows = [], [], []
    t_all = time.time()
    t_roll = t_save = 0.0
    vec, vec_task = None, None
    for gid, (task_id, init_id) in enumerate(specs):
        if vec_task != task_id:
            if vec is not None:
                vec.close()
            vec, vec_task = make_vec_env(task_id, G, log_sim_state=False), task_id
            task_text = vec.call("task_description")[0]
        seeds = [seed0 + 1000 * gid + j for j in range(G)]
        cap = ChunkCapture(policy[0], base_seed=seed0 * 100 + 1000 * gid)
        t0 = time.time()
        res = run_batch(vec, *policy, task_text=task_text, seeds=seeds, init_ids=[init_id] * G)
        cap.detach()
        t_roll += time.time() - t0
        lab = group_segment_labels(res, rcfg, c["min_return_spread"], c["adv_deadzone"])
        dec = cap.decisions([r["length"] for r in res])
        adv = np.zeros(len(dec["env"]), np.float32)
        code = np.zeros(len(dec["env"]), np.int8)
        for i, (j, s) in enumerate(zip(dec["env"], dec["step"])):
            seg = [x for x in lab["rollouts"][j]["segments"] if x["start"] <= s < x["end"]]
            if seg and not lab["rejected"]:
                adv[i], code[i] = seg[0]["adv"], CODES[seg[0]["label"]]
        t1 = time.time()
        np.savez(out / "decisions" / f"g{gid:02d}.npz", **dec, adv=adv, label_code=code,
                 group=np.full(len(adv), gid), task_id=np.full(len(adv), task_id),
                 rejected=np.full(len(adv), int(lab["rejected"])))
        n_succ = int(sum(r["success"] for r in res))
        group_rows.append({"iter": args.iter, "group": gid, "task_id": task_id, "init_id": init_id, "n_success": n_succ,
                           "G": G, "return_mean": lab["return_mean"], "return_spread": lab["spread"],
                           "all_success": lab["all_success"], "rejected": lab["rejected"],
                           "n_decisions": len(adv), "rollout_s": t1 - t0})
        for j, (r, lr) in enumerate(zip(res, lab["rollouts"])):
            name = f"it{args.iter}_task{task_id}_g{gid:02d}_j{j}"
            save_episode(out / "logs" / name, r, video=False)
            fl = label_episode(r["log"], r["actions"], r["success"])
            cnt = pd.Series([e["type"] for e in lr["events"]], dtype=object).value_counts().to_dict()
            roll_rows.append({"iter": args.iter, "name": name, "group": gid, "task_id": task_id, "init_id": init_id,
                              "seed": r["seed"], "success": r["success"], "length": r["length"],
                              "return": lr["return"], "primary": fl["primary"],
                              **{f"n_{x}": cnt.get(x, 0) for x in ("drop", "slip", "collision", "timeout")}})
            for s in lr["segments"]:
                seg_rows.append({"iter": args.iter, "name": name, "group": gid, "task_id": task_id, **s})
        t_save += time.time() - t1
        print(f"[group {gid}] task {task_id} init {init_id} succ={n_succ}/{G} spread={lab['spread']:.3f} "
              f"rejected={lab['rejected']} decisions={len(adv)} roll {t1 - t0:.0f}s save {time.time() - t1:.0f}s",
              flush=True)
    vec.close()
    pd.DataFrame(group_rows).to_csv(out / "groups.csv", index=False)
    pd.DataFrame(roll_rows).to_csv(out / "rollouts.csv", index=False)
    pd.DataFrame(seg_rows).to_csv(out / "segments.csv", index=False)
    info = {"iter": args.iter, "policy": args.policy, "seed0": seed0, "seconds_total": time.time() - t_all,
            "seconds_rollout": t_roll, "seconds_save": t_save,
            "torch_max_mem_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
            "n_rollouts": len(roll_rows), "n_success": int(sum(r["success"] for r in roll_rows)),
            "n_decisions": int(sum(g["n_decisions"] for g in group_rows))}
    json.dump(info, open(out / "collect_info.json", "w"), indent=1)
    print(f"[done] {json.dumps(info)}")


if __name__ == "__main__":
    main()
