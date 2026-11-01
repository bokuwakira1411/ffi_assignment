"""FPO v3 smoke test (required before the run): is the cached demo data in the format theta0 saw in training / eval?

Checks, theta0 = official checkpoint loaded exactly as in fpo_train_v2 (fp32, trainable tensors from file):
  1. language tokens of the demo samples of env tasks 0 / 8 == tokens stored in the rollout decisions (v2 iter0);
  2. normalised state: per-dim mean / std of demos vs rollouts (tasks 0, 8);
  3. pixels: mean |demo - rollout| of the first frames (task 0, 8), demo as stored vs rotated 180 deg;
  4. L_CFM(theta0), fp32, with the SAME (tau, eps) per sample index for every variant (paired):
       rollout successful decisions (tasks 0, 8, executed steps n_exec <= 10)  -- reference
       demo tasks 0 / 8 and all 10 tasks, first min(10, valid) steps and the full valid chunk
       controls that must be clearly WORSE if the format is right: demo images rotated 180 deg, camera1 <-> camera2
       swapped, language tokens of another task (rolled by 1 task).
Writes <cache_dir>/demo_check.json and demo_vs_rollout.png (first frames, camera1 / camera2).

    python scripts/fpo_v3_demo_check.py --config ~/ffi26/results/round2/configs/fpo_v3.yaml --data outputs/fpo_v2/iter0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402

from fpo_train import make_batch  # noqa: E402
from fpo_train_v2 import build_policy_fp32, cfm, load_success_decisions  # noqa: E402

C1, C2 = "observation.images.camera1", "observation.images.camera2"
TOK, MASK = "observation.language.tokens", "observation.language.attention_mask"


def sub(d, idx):
    return {k: v[idx] for k, v in d.items()}


@torch.no_grad()
def loss_on(policy, d, n_exec, tau, eps, dev, mb=32):
    out = []
    for s in range(0, len(n_exec), mb):
        j = np.arange(s, min(s + mb, len(n_exec)))
        dd = dict(d)
        dd["n_exec"] = n_exec
        b = make_batch(dd, j, dev, policy.config.chunk_size)
        out.append(cfm(policy, b, eps[j].to(dev), tau[j].to(dev)).cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=128, help="samples per set")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    cdir = Path(cfg["demo"]["cache_dir"]).expanduser()
    demo = dict(np.load(cdir / "demo_train.npz"))
    succ = load_success_decisions(Path(args.data).expanduser())
    dev = "cuda"
    rep: dict = {}

    # 1. tokens
    for t in (0, 8):
        dt = np.unique(demo[TOK][demo["task_id"] == t], axis=0)
        rt = np.unique(succ[t][TOK], axis=0)
        rep[f"tokens_task{t}"] = {"n_unique_demo": len(dt), "n_unique_rollout": len(rt),
                                  "identical": bool(len(dt) == 1 and len(rt) == 1 and (dt == rt).all()
                                                    and (np.unique(demo[MASK][demo["task_id"] == t], axis=0)
                                                         == np.unique(succ[t][MASK], axis=0)).all())}

    # 2. state
    for t in (0, 8):
        ds, rs = demo["observation.state"][demo["task_id"] == t], succ[t]["observation.state"]
        rep[f"state_task{t}"] = {"demo_mean": ds.mean(0).round(3).tolist(), "rollout_mean": rs.mean(0).round(3).tolist(),
                                 "demo_std": ds.std(0).round(3).tolist(), "rollout_std": rs.std(0).round(3).tolist()}

    # 3. pixels: early frames (demo frame < 10, rollout step 0)
    tiles = []
    for t in (0, 8):
        dm = (demo["task_id"] == t) & (demo["frame"] < 10)
        rm = succ[t]["step"] == 0
        r = {}
        for cam in (C1, C2):
            dimg = demo[cam][dm].astype(np.float32).mean(0)
            rimg = succ[t][cam][rm].astype(np.float32).mean(0)
            r[cam] = {"as_stored": float(np.abs(dimg - rimg).mean()),
                      "rot180": float(np.abs(dimg[:, ::-1, ::-1] - rimg).mean())}
            if dm.any() and rm.any():
                tiles.append(np.concatenate([demo[cam][dm][0], succ[t][cam][rm][0]], axis=2).transpose(1, 2, 0))
        rep[f"pixel_mad_task{t}"] = {**r, "n_demo": int(dm.sum()), "n_rollout": int(rm.sum())}
    if tiles:
        Image.fromarray(np.concatenate(tiles, axis=0)).save(cdir / "demo_vs_rollout.png")

    # 4. losses
    policy, _, _ = build_policy_fp32(cfg["init_policy"], cfg["n_action_steps"])
    rng = np.random.default_rng(0)
    N = args.n
    torch.manual_seed(1234)
    tau = policy.model.sample_time(N, "cpu").float()
    eps = torch.randn(N, policy.config.chunk_size, policy.config.max_action_dim)
    L = {}
    for t in (0, 8):
        j = np.sort(rng.choice(len(succ[t]["adv"]), N, replace=False))
        d = sub(succ[t], j)
        L[f"rollout_success_task{t}_nexec"] = loss_on(policy, d, d["n_exec"], tau, eps, dev)
    sets = {"task0": demo["task_id"] == 0, "task8": demo["task_id"] == 8, "all10": np.ones(len(demo["task_id"]), bool)}
    for name, m in sets.items():
        j = np.sort(rng.choice(np.flatnonzero(m), N, replace=False))
        d = sub(demo, j)
        n10 = np.minimum(10, d["n_exec"])
        L[f"demo_{name}_first10"] = loss_on(policy, d, n10, tau, eps, dev)
        L[f"demo_{name}_full"] = loss_on(policy, d, d["n_exec"], tau, eps, dev)
        if name == "all10":
            rot = dict(d)
            rot[C1], rot[C2] = d[C1][:, :, ::-1, ::-1].copy(), d[C2][:, :, ::-1, ::-1].copy()
            L["ctrl_all10_rot180_first10"] = loss_on(policy, rot, n10, tau, eps, dev)
            sw = dict(d)
            sw[C1], sw[C2] = d[C2], d[C1]
            L["ctrl_all10_cam_swap_first10"] = loss_on(policy, sw, n10, tau, eps, dev)
            wt = dict(d)
            other = {t: demo[TOK][demo["task_id"] == (t + 1) % 10][0] for t in range(10)}
            otherm = {t: demo[MASK][demo["task_id"] == (t + 1) % 10][0] for t in range(10)}
            wt[TOK] = np.stack([other[t] for t in d["task_id"]])
            wt[MASK] = np.stack([otherm[t] for t in d["task_id"]])
            L["ctrl_all10_wrong_task_first10"] = loss_on(policy, wt, n10, tau, eps, dev)
    rep["loss"] = {k: {"mean": float(v.mean()), "median": float(np.median(v)), "sem": float(v.std() / np.sqrt(len(v)))}
                   for k, v in L.items()}
    base = L["demo_all10_first10"]
    rep["paired_ctrl_minus_demo_first10"] = {k: {"mean_diff": float((L[k] - base).mean()),
                                                 "frac_ctrl_higher": float((L[k] > base).mean())}
                                             for k in L if k.startswith("ctrl_")}
    json.dump(rep, open(cdir / "demo_check.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
