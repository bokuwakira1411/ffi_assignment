"""Task-0 diagnosis, part B (no training, no optimizer step): do the task-0 and task-8 FPO terms of v3 pull the
trainable weights in opposite directions? Gradients at the START of v3 training iteration k (= what step 1 of that
iteration saw), k = 0..3:
  policy theta_k = the policy that collected iteration k (outputs/fpo_v3/iter{k}/collect_info.json "policy"):
      k = 0 official lerobot/smolvla_libero, k >= 1 outputs/fpo_v3/iter{k-1}/train/pretrained_model;
  data   = outputs/fpo_v3/iter{k} decisions (load_decisions: rejected / zero-advantage rows dropped, as v3 training);
  A^     = v3 per-task normalisation fpo_adv.normalise(d, 1, "mean") = clip((A - mean_t) / std_t, -5, 5);
  c      = ratio_scale_used of that iteration (train_info.json); (tau, eps) = the stored training pairs
           draw_pairs32(theta_k, n_t, n_pairs, fpo.seed + t), pair 0 for every decision.
At the start of an iteration l_old = l_new, so r = 1 and the gradient of the v3 PPO term of one task's decisions,
-(1 / B) sum min(r A^, clip(r) A^), is (c / B) sum A^ grad l_new (clip inactive at r = 1). Here
g_t = (c / 2) * mean_i A^_i grad l_new_i (= one optimizer step's task-t share, B / 2 decisions, at any n).
g_demo = lambda_demo * mean grad L_CFM over demo_batch demo samples (stratified over the 10 tasks, as v3).
BC / demo are NOT in g_0 / g_8. Same build as training: build_policy_fp32 (fp32, VLM frozen, expert trainable),
train mode, torch seed fixed before every backward.
Per repeat r: the rollouts (group, env) of each task are split at random into halves a / b, n_dec decisions are drawn
from each half -> g0_a, g0_b, g8_a, g8_b; g0 = (g0_a + g0_b) / 2. Controls: cos(g0_a, g0_b), cos(g8_a, g8_b);
cross-task at the same sample size: cos(g0_a, g8_a), cos(g0_b, g8_b), cos(g0_a, g8_b), cos(g0_b, g8_a).

    CUDA_VISIBLE_DEVICES="" nice -n 19 python scripts/task0_diag_grad.py --iters 0 1 2 3 --reps 5   (from ~/ffi26/improve)
Output: outputs/task0_diag/grad_cos.csv (one row per iteration x repeat; resumable).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from fpo_adv import normalise  # noqa: E402
from fpo_train import FREEZE, expand, load_decisions, make_batch  # noqa: E402
from fpo_train_v2 import cfm, draw_pairs32  # noqa: E402
from rollout import checkpoint_file, load_policy  # noqa: E402

torch.set_float32_matmul_precision("highest")
CFG = Path("~/ffi26/results/round2/configs/fpo_v3.yaml").expanduser()
V3 = ROOT / "outputs/fpo_v3"
OUT = ROOT / "outputs/task0_diag"


def build(path: str, nas: int, dev: str):
    """= fpo_train_v2.build_policy_fp32 with a device argument."""
    policy = load_policy(expand(path), nas, device=dev, cfg_overrides=FREEZE)[0]
    policy.float()
    sd = load_file(str(checkpoint_file(expand(path))))
    with torch.no_grad():
        for n, p in policy.named_parameters():
            if p.requires_grad and n in sd:
                p.copy_(sd[n].float().to(p.device))
    policy.train()
    return policy


def grad_vec(policy, params, d, idx, w, tau, eps, scale, mb, dev, seed) -> torch.Tensor:
    """flattened grad of scale * mean_i w_i * l(idx_i; tau_i, eps_i) (w = A^ or 1)."""
    for p in params:
        p.grad = None
    torch.manual_seed(seed)
    for s in range(0, len(idx), mb):
        j = idx[s:s + mb]
        b = make_batch(d, j, dev, policy.config.chunk_size)
        l = cfm(policy, b, eps[s:s + mb].to(dev), tau[s:s + mb].to(dev))
        (scale * (torch.as_tensor(w[s:s + mb], device=dev) * l).sum() / len(idx)).backward()
    return torch.cat([p.grad.detach().flatten().float().cpu() for p in params])


def cos(a, b) -> float:
    return float(torch.dot(a, b) / (a.norm() * b.norm()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--n_dec", type=int, default=32, help="decisions per half per task")
    ap.add_argument("--mb", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default=str(OUT / "grad_cos.csv"))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    cfg = yaml.safe_load(open(CFG))
    f, dev = cfg["fpo"], args.device
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        with open(out) as fh:
            done = {(int(r["iter"]), int(r["rep"])) for r in csv.DictReader(fh)}
    demo = dict(np.load(Path(cfg["demo"]["cache_dir"]).expanduser() / "demo_train.npz"))
    demo_tasks = np.unique(demo["task_id"])
    demo_by_task = {int(t): np.flatnonzero(demo["task_id"] == t) for t in demo_tasks}
    fh, wr = None, None
    for k in args.iters:
        if all((k, r) in done for r in range(args.reps)):
            continue
        ddir = V3 / f"iter{k}"
        pol_path = json.load(open(ddir / "collect_info.json"))["policy"]
        tinfo = json.load(open(ddir / "train" / "train_info.json"))
        assert expand(tinfo["policy"]) == expand(pol_path), (tinfo["policy"], pol_path)
        c = float(tinfo["ratio_scale_used"])
        t0 = time.time()
        policy = build(pol_path, cfg["n_action_steps"], dev)
        params = [p for p in policy.parameters() if p.requires_grad]
        data = load_decisions(ddir)
        assert {int(t): len(d["adv"]) for t, d in data.items()} == {int(t): v for t, v in tinfo["decisions"].items()}
        ahat = {t: normalise(d, np.ones(len(d["adv"])), "mean")[0] for t, d in data.items()}
        pairs = {t: draw_pairs32(policy, len(d["adv"]), f["n_pairs"], f["seed"] + t) for t, d in data.items()}
        print(f"[iter{k}] policy {pol_path}, c {c:.3f}, decisions { {t: len(d['adv']) for t, d in data.items()} }, "
              f"{sum(p.numel() for p in params)} trainable, load {time.time() - t0:.0f}s", flush=True)
        for r in range(args.reps):
            if (k, r) in done:
                continue
            ts = time.time()
            rng = np.random.default_rng(1000 + r)
            g, rec = {}, {"iter": k, "rep": r, "policy": pol_path, "c": c, "n_dec_half": args.n_dec}
            for t in (0, 8):
                d = data[t]
                key = d["group"].astype(np.int64) * 1000 + d["env"]
                ro = rng.permutation(np.unique(key))
                halves = {"a": ro[: len(ro) // 2], "b": ro[len(ro) // 2:]}
                for h, keys in halves.items():
                    pool = np.flatnonzero(np.isin(key, keys))
                    idx = np.sort(rng.choice(pool, args.n_dec, replace=False))
                    tau, eps = pairs[t][0][idx, 0], pairs[t][1][idx, 0]
                    g[f"{t}{h}"] = grad_vec(policy, params, d, idx, ahat[t][idx], tau, eps, c / 2, args.mb, dev,
                                            seed=50000 + 100 * r + t)
                    rec[f"n_rollouts_{t}{h}"] = len(keys)
                    rec[f"ahat_mean_{t}{h}"] = float(ahat[t][idx].mean())
            # demo replay term, stratified as v3 (demo_batch // 10 per task + remainder on random tasks)
            nd, nt = f["demo_batch"], len(demo_tasks)
            cnt = {int(t): nd // nt for t in demo_tasks}
            for t in rng.choice(demo_tasks, nd % nt, replace=False):
                cnt[int(t)] += 1
            jd = np.concatenate([rng.choice(demo_by_task[t], n, replace=False) for t, n in cnt.items() if n > 0])
            torch.manual_seed(9000 + r)
            taud = policy.model.sample_time(len(jd), "cpu").float()
            epsd = torch.randn(len(jd), policy.config.chunk_size, policy.config.max_action_dim)
            g["demo"] = grad_vec(policy, params, demo, jd, np.ones(len(jd), np.float32), taud, epsd,
                                 f["lambda_demo"], args.mb, dev, seed=60000 + r)
            g["0"], g["8"] = (g["0a"] + g["0b"]) / 2, (g["8a"] + g["8b"]) / 2
            rec.update({
                "cos_0a_0b": cos(g["0a"], g["0b"]), "cos_8a_8b": cos(g["8a"], g["8b"]),
                "cos_0a_8a": cos(g["0a"], g["8a"]), "cos_0b_8b": cos(g["0b"], g["8b"]),
                "cos_0a_8b": cos(g["0a"], g["8b"]), "cos_0b_8a": cos(g["0b"], g["8a"]),
                "cos_0_8": cos(g["0"], g["8"]), "cos_0_demo": cos(g["0"], g["demo"]),
                "cos_8_demo": cos(g["8"], g["demo"]),
                **{f"norm_{n}": float(g[n].norm()) for n in ("0", "8", "0a", "0b", "8a", "8b", "demo")},
                "seconds": round(time.time() - ts, 1)})
            if wr is None:
                new = not out.exists()
                fh = open(out, "a", newline="")
                wr = csv.DictWriter(fh, fieldnames=list(rec))
                if new:
                    wr.writeheader()
            wr.writerow(rec)
            fh.flush()
            print(json.dumps({k2: (round(v, 4) if isinstance(v, float) else v) for k2, v in rec.items()}), flush=True)
            del g
        del policy, data, pairs
    if fh:
        fh.close()


if __name__ == "__main__":
    main()
