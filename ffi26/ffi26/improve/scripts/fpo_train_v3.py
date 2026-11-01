"""FPO v3, iteration k, training step (collection unchanged: scripts/fpo_collect.py). = fpo_train_v2.py + 3 changes
(plan in results/round2/configs/fpo_v3.yaml); everything not listed is the v2 code path (fp32, stored (tau, eps)
pairs for l_old / l_new, fp32 master weights across iterations, warmup, clip, early stop, KL proxy, epoch cap):
  (1) demo replay: + lambda_demo * L_CFM on demo_batch LIBERO-10 demo samples per step (cache from
      scripts/fpo_build_demo_cache.py, official preprocessor), stratified over the 10 tasks (demo_batch // 10 per
      task + the remainder on tasks drawn without replacement), full valid 50-step chunk (lerobot-train objective),
      fresh (tau, eps). Demo-probe L_CFM (held-out demo episodes, fixed (tau, eps)) per task at start / end.
  (2) per-task advantage normalisation: A^ = clip((A - mean_t) / std_t, -5, 5) per task t of this iteration;
      every optimizer step = batch_size / 2 decisions of each task (separate shuffled streams per task; one epoch =
      sum_t n_t // batch_size steps as in v2), rollout-BC bc_batch / 2 per task.
  (3) pairs_per_step: P stored (tau, eps) pairs per decision per step (distinct, of n_pairs), ratio
      r = exp(clamp(c * mean_p (l_old_p - l_new_p), -5, 5)) (v2: P = 1 random pair).
Outputs: <out>/pretrained_model, train_log.csv, kl_log.csv, demo_probe.csv, train_info.json, one row in
<out_root>/iter_summary.csv.

    python scripts/fpo_train_v3.py --config ~/ffi26/results/round2/configs/fpo_v3.yaml \
        --policy <theta_k> --data outputs/fpo_v3/iter0 --out outputs/fpo_v3/iter0/train
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from fpo_train import expand, load_decisions, make_batch, save_policy  # noqa: E402
from fpo_train_v2 import build_policy_fp32, cfm, draw_pairs32, load_success_decisions, losses_fixed  # noqa: E402

torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class Stream:
    """endless shuffled index stream over n items (reshuffled at every pass)."""

    def __init__(self, n: int, rng):
        self.n, self.rng, self.buf = n, rng, np.zeros(0, np.int64)

    def take(self, k: int) -> np.ndarray:
        out = []
        while k > 0:
            if len(self.buf) == 0:
                self.buf = self.rng.permutation(self.n)
            out.append(self.buf[:k])
            k -= len(out[-1])
            self.buf = self.buf[len(out[-1]):]
        return np.concatenate(out)


def demo_probe(policy, probe, tau, eps, mb, dev) -> dict[int, float]:
    l = losses_fixed(policy, probe, np.arange(len(tau)), tau, eps, mb, dev).numpy()
    return {int(t): float(l[probe["task_id"] == t].mean()) for t in np.unique(probe["task_id"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_steps", type=int, default=None, help="smoke test: stop after n steps")
    ap.add_argument("--no_probe", action="store_true", help="smoke test: skip KL / demo probes and the old-loss "
                                                            "pass beyond the decisions used (timing runs)")
    ap.add_argument("--max_decisions", type=int, default=None, help="smoke test: at most n decisions per task")
    ap.add_argument("--set", nargs="*", default=[], help="override fpo.<key>=<value>")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    f = cfg["fpo"]
    for kv in args.set:
        k, v = kv.split("=", 1)
        f[k] = yaml.safe_load(v)
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data).expanduser()
    dev, B, MB, P = "cuda", f["batch_size"], f["micro_batch"], int(f.get("pairs_per_step", 1))
    assert B % 2 == 0 and (B // 2) % MB == 0 and 1 <= P <= f["n_pairs"]
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()

    data = load_decisions(data_dir)
    bcdata = load_success_decisions(data_dir)
    tasks = sorted(data)
    assert len(tasks) == 2, tasks
    if args.no_probe:  # timing smoke test: only the first decisions of each task (old-loss pass is per iteration)
        data = {t: {k: v[: B * (args.max_steps or 1)] for k, v in d.items()} for t, d in data.items()}
    if args.max_decisions:
        data = {t: {k: v[: args.max_decisions] for k, v in d.items()} for t, d in data.items()}
    adv_stats = {t: (float(d["adv"].mean()), float(d["adv"].std())) for t, d in data.items()}
    adv_hat = {t: np.clip((d["adv"] - adv_stats[t][0]) / adv_stats[t][1], -5, 5).astype(np.float32)
               for t, d in data.items()}
    cdir = Path(cfg["demo"]["cache_dir"]).expanduser()
    demo = dict(np.load(cdir / "demo_train.npz"))
    probe = dict(np.load(cdir / "demo_probe.npz"))
    demo_tasks = np.unique(demo["task_id"])
    demo_by_task = {int(t): np.flatnonzero(demo["task_id"] == t) for t in demo_tasks}

    # KL probe (fixed decisions + (tau, eps)) and l_theta0 on it, theta0 fp32 (= v2); demo probe (fixed (tau, eps))
    p0, _, _ = build_policy_fp32(cfg["init_policy"], cfg["n_action_steps"])
    torch.manual_seed(cfg["demo"]["probe_seed"])
    ptau = p0.model.sample_time(len(probe["task_id"]), "cpu").float()
    peps = torch.randn(len(probe["task_id"]), p0.config.chunk_size, p0.config.max_action_dim)
    if not args.no_probe:
        prng = np.random.default_rng(f["seed"] + 17)
        kprobe = {}
        for t, d in data.items():
            idx = np.sort(prng.choice(len(d["adv"]), min(f["kl_probe_per_task"], len(d["adv"])), replace=False))
            torch.manual_seed(f["seed"] + 31 + t)
            tau = p0.model.sample_time(len(idx), "cpu").float()
            eps = torch.randn(len(idx), p0.config.chunk_size, p0.config.max_action_dim)
            kprobe[t] = (idx, tau, eps)
        l0 = {t: losses_fixed(p0, data[t], *kprobe[t], MB, dev) for t in data}
        c0 = 1.0 / float(torch.cat(list(l0.values())).mean())
        dp_theta0 = demo_probe(p0, probe, ptau, peps, MB, dev)
    del p0
    torch.cuda.empty_cache()

    policy, orig_dtype, n_restored = build_policy_fp32(args.policy, cfg["n_action_steps"])
    params = [p for p in policy.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)

    def kl_proxy():
        if args.no_probe:
            return {"kl_proxy": float("nan"), "absmean_dl_theta0": float("nan")}
        lt = torch.cat([losses_fixed(policy, data[t], *kprobe[t], MB, dev) for t in data])
        lr0 = c0 * (torch.cat([l0[t] for t in data]) - lt).double()
        return {"kl_proxy": float(0.5 * (lr0 ** 2).mean()), "absmean_dl_theta0": float((lr0 / c0).abs().mean())}

    untrained = [int(t) for t in np.unique(probe["task_id"]) if t not in tasks]
    ph = open(out / "demo_probe.csv", "w", newline="")
    pw = csv.DictWriter(ph, fieldnames=["when", "step", "untrained_mean", *[f"task{t}" for t in np.unique(probe["task_id"])]])
    pw.writeheader()

    def log_probe(when, step):
        if args.no_probe:
            return float("nan")
        d = dp_theta0 if when == "theta0" else demo_probe(policy, probe, ptau, peps, MB, dev)
        um = float(np.mean([d[t] for t in untrained]))
        pw.writerow({"when": when, "step": step, "untrained_mean": um, **{f"task{t}": v for t, v in d.items()}})
        ph.flush()
        return um

    # old losses, fp32, stored (tau, eps) (= v2)
    pairs = {t: draw_pairs32(policy, len(d["adv"]), f["n_pairs"], f["seed"] + t) for t, d in data.items()}
    t_old = time.time()
    lold = {}
    for t, d in data.items():
        tau, eps = pairs[t]
        n = len(d["adv"])
        L = torch.zeros(n, f["n_pairs"])
        for p in range(f["n_pairs"]):
            L[:, p] = losses_fixed(policy, d, np.arange(n), tau[:, p], eps[:, p], MB, dev)
        lold[t] = L
    t_old = time.time() - t_old
    mean_lold = float(torch.cat([v.flatten() for v in lold.values()]).mean())
    c = 1.0 / mean_lold if f["ratio_scale"] == "auto" else float(f["ratio_scale"])
    steps_per_epoch = sum(len(d["adv"]) // B for d in data.values())
    max_steps = f["epochs"] * steps_per_epoch
    warm = max(1, math.ceil(f["warmup_samples"] / B))
    kl_start = kl_proxy()
    log_probe("theta0", 0)
    dp_start = log_probe("start", 0)
    print(f"[fpo_v3] decisions { {t: len(d['adv']) for t, d in data.items()} }, success decisions "
          f"{ {t: len(d['adv']) for t, d in bcdata.items()} }, adv mean/std per task {adv_stats}; B {B} "
          f"({B // 2}/task, micro {MB}), pairs/step {P}, steps/epoch {steps_per_epoch}, max epochs {f['epochs']}, "
          f"warmup {warm}; demo {len(demo['task_id'])} samples, lambda_demo {f['lambda_demo']} x {f['demo_batch']}; "
          f"old-loss pass {t_old:.0f}s, mean l_old {mean_lold:.5f}, c {c:.3f}; fp32 restored {n_restored}; KL start "
          f"{kl_start}; demo-probe untrained start {dp_start:.5f}", flush=True)

    opt = torch.optim.AdamW(params, lr=f["lr"], betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm))
    rng = np.random.default_rng(f["seed"])
    streams = {t: Stream(len(data[t]["adv"]), rng) for t in tasks}
    bc_streams = {t: Stream(len(bcdata[t]["adv"]), rng) for t in tasks if t in bcdata and len(bcdata[t]["adv"])}
    demo_streams = {t: Stream(len(ix), rng) for t, ix in demo_by_task.items()}
    fh = open(out / "train_log.csv", "w", newline="")
    cols = ["step", "epoch", "loss_ppo", "loss_bc", "loss_demo", "ratio_mean", "ratio_min", "ratio_max", "std_logr",
            "clip_frac", "clip_frac_t0", "clip_frac_t8", "dl_absmean", "l_new", "grad_norm", "lr", "time_s",
            "step_s", "gpu_mem_gb"]
    wr = csv.DictWriter(fh, fieldnames=cols)
    wr.writeheader()
    kh = open(out / "kl_log.csv", "w", newline="")
    kw = csv.DictWriter(kh, fieldnames=["step", "kl_proxy", "absmean_dl_theta0"])
    kw.writeheader()
    kw.writerow({"step": 0, **kl_start})
    step, clips, stop_reason = 0, [], "epochs"
    step_times = []

    while step < max_steps:
        ts = time.time()
        opt.zero_grad(set_to_none=True)
        rs, dls, lns, cft, lp = [], [], [], {}, 0.0
        for t in tasks:  # (2) B/2 decisions of each task
            d, (tau, eps), idx = data[t], pairs[t], streams[t].take(B // 2)
            # (3) P distinct pairs per decision
            pp = np.stack([rng.permutation(f["n_pairs"])[:P] for _ in idx])
            rt = []
            for s in range(0, B // 2, MB):
                j, pj = idx[s:s + MB], pp[s:s + MB]
                b = make_batch(d, j, dev, policy.config.chunk_size)
                dl = 0.0
                for q in range(P):
                    lnew = cfm(policy, b, eps[j, pj[:, q]].to(dev), tau[j, pj[:, q]].to(dev))
                    dl = dl + (lold[t][j, pj[:, q]].to(dev) - lnew) / P
                    lns.append(lnew.detach())
                r = torch.exp((c * dl).clamp(-5, 5))
                A = torch.from_numpy(adv_hat[t][j]).to(dev)
                surr = torch.minimum(r * A, r.clamp(1 - f["clip"], 1 + f["clip"]) * A)
                loss = -surr.sum() / B
                loss.backward()
                lp += float(loss)
                rs.append(r.detach()); dls.append(dl.detach()); rt.append(r.detach())
            cft[t] = float(((torch.cat(rt) - 1).abs() > f["clip"]).float().mean())
        # rollout BC (= v2 term, bc_batch / 2 per task)
        lb = 0.0
        if f["lambda_bc"] > 0 and bc_streams:
            nb = f["bc_batch"]
            for t, st in bc_streams.items():
                jb = st.take(nb // len(bc_streams))
                for s in range(0, len(jb), MB):
                    jj = jb[s:s + MB]
                    b = make_batch(bcdata[t], jj, dev, policy.config.chunk_size)
                    taub = policy.model.sample_time(len(jj), dev).float()
                    epsb = torch.randn(len(jj), policy.config.chunk_size, policy.config.max_action_dim, device=dev)
                    lbc = cfm(policy, b, epsb, taub).sum() / nb
                    (f["lambda_bc"] * lbc).backward()
                    lb += float(lbc)
        # (1) demo replay, stratified over the 10 tasks
        ld = 0.0
        if f["lambda_demo"] > 0:
            nd = f["demo_batch"]
            nt = len(demo_tasks)
            cnt = {int(t): nd // nt for t in demo_tasks}
            for t in rng.choice(demo_tasks, nd % nt, replace=False):
                cnt[int(t)] += 1
            jd = np.concatenate([demo_by_task[t][demo_streams[t].take(k)] for t, k in cnt.items() if k > 0])
            for s in range(0, nd, MB):
                jj = jd[s:s + MB]
                b = make_batch(demo, jj, dev, policy.config.chunk_size)
                taud = policy.model.sample_time(len(jj), dev).float()
                epsd = torch.randn(len(jj), policy.config.chunk_size, policy.config.max_action_dim, device=dev)
                ldm = cfm(policy, b, epsd, taud).sum() / nd
                (f["lambda_demo"] * ldm).backward()
                ld += float(ldm)
        gn = torch.nn.utils.clip_grad_norm_(params, f["grad_clip"])
        opt.step()
        sched.step()
        step += 1
        torch.cuda.synchronize()
        step_times.append(time.time() - ts)
        r = torch.cat(rs); dl = torch.cat(dls)
        logr = (c * dl).clamp(-5, 5)
        cf = float(((r - 1).abs() > f["clip"]).float().mean())
        clips.append(cf)
        ep_used = step / steps_per_epoch
        wr.writerow({"step": step, "epoch": int((step - 1) // steps_per_epoch), "loss_ppo": lp, "loss_bc": lb,
                     "loss_demo": ld, "ratio_mean": float(r.mean()), "ratio_min": float(r.min()),
                     "ratio_max": float(r.max()), "std_logr": float(logr.std()), "clip_frac": cf,
                     "clip_frac_t0": cft.get(tasks[0]), "clip_frac_t8": cft.get(tasks[1]),
                     "dl_absmean": float(dl.abs().mean()), "l_new": float(torch.cat(lns).mean()),
                     "grad_norm": float(gn), "lr": sched.get_last_lr()[0], "time_s": round(time.time() - t0, 1),
                     "step_s": round(step_times[-1], 2),
                     "gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3)})
        if step % f["kl_every"] == 0 and not args.no_probe:
            fh.flush()
            k = kl_proxy()
            kw.writerow({"step": step, **k}); kh.flush()
            print(f"[fpo_v3] step {step} ep {ep_used:.2f} loss {lp:+.4f} bc {lb:.4f} demo {ld:.4f} ratio "
                  f"[{float(r.min()):.3f},{float(r.max()):.3f}] std_logr {float(logr.std()):.4f} clip_frac(last10) "
                  f"{np.mean(clips[-10:]):.3f} gn {float(gn):.2f} KL {k['kl_proxy']:.2e} {time.time() - t0:.0f}s",
                  flush=True)
        if args.max_steps is not None and step >= args.max_steps:
            stop_reason = "max_steps"
            break
        w = f["early_stop_window"]
        if len(clips) >= w and np.mean(clips[-w:]) > f["early_stop_clip_frac"]:
            print(f"[fpo_v3] early stop at step {step}: clip_frac over last {w} steps {np.mean(clips[-w:]):.3f}",
                  flush=True)
            stop_reason = "early_stop"
            break
    ep_used = step / steps_per_epoch
    fh.close()
    kl_end = kl_proxy()
    kw.writerow({"step": step, **kl_end}); kh.close()
    dp_end = log_probe("end", step)
    ph.close()
    log = pd.read_csv(out / "train_log.csv")
    info = {"policy": args.policy, "data": str(data_dir), "decisions": {int(t): len(d["adv"]) for t, d in data.items()},
            "success_decisions": {int(t): len(d["adv"]) for t, d in bcdata.items()},
            "adv_mean_std_per_task": {int(t): v for t, v in adv_stats.items()}, "batch_size": B, "micro_batch": MB,
            "pairs_per_step": P, "steps": step, "steps_per_epoch": steps_per_epoch, "epochs_used": ep_used,
            "stop_reason": stop_reason, "warmup_steps": warm, "ratio_scale_used": c, "mean_l_old": mean_lold,
            "kl_start": kl_start, "kl_end": kl_end, "demo_probe_untrained_start": dp_start,
            "demo_probe_untrained_end": dp_end, "clip_frac_mean": float(log.clip_frac.mean()),
            "clip_frac_last10": float(log.clip_frac.tail(10).mean()),
            "step_s_mean_after_first": float(np.mean(step_times[1:])) if len(step_times) > 1 else float("nan"),
            "fp32_trainable_restored": n_restored, "seconds_total": time.time() - t0, "seconds_old_loss": t_old,
            "trainable_params": n_train, "torch_max_mem_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
            "fpo": f}
    if args.max_steps is None:
        ro = pd.read_csv(data_dir / "rollouts.csv")
        cs = {int(t): f"{int(g.success.sum())}/{len(g)}" for t, g in ro.groupby("task_id")}
        info["collect_success_per_task"] = cs
        info["collect_success"] = f"{int(ro.success.sum())}/{len(ro)}"
    json.dump(info, open(out / "train_info.json", "w"), indent=1)
    with torch.no_grad():  # frozen tensors back to their original dtype: collection / eval load as in v1 / v2
        for n, p in policy.named_parameters():
            if not p.requires_grad and orig_dtype[n] != torch.float32:
                p.data = p.data.to(orig_dtype[n])
    save_policy(policy, expand(args.policy), out / "pretrained_model")
    if args.max_steps is None:
        summ = Path(expand(cfg["out_root"])) / "iter_summary.csv"
        new = not summ.exists()
        with open(summ, "a", newline="") as fs:
            w = csv.writer(fs)
            if new:
                w.writerow(["iter_dir", "batch_size", "pairs_per_step", "steps", "epochs_used", "stop_reason",
                            "clip_frac_mean", "clip_frac_last10", "kl_start", "kl_end", "demo_probe_untrained_start",
                            "demo_probe_untrained_end", "collect_success", "collect_task0", "collect_task8",
                            "seconds_train"])
            w.writerow([data_dir.name, B, P, step, round(ep_used, 3), stop_reason, round(info["clip_frac_mean"], 4),
                        round(info["clip_frac_last10"], 4), f"{kl_start['kl_proxy']:.3e}", f"{kl_end['kl_proxy']:.3e}",
                        f"{dp_start:.5f}", f"{dp_end:.5f}", info["collect_success"], cs.get(0, ""), cs.get(8, ""),
                        round(info["seconds_total"])])
    print(f"[done] {json.dumps({k: v for k, v in info.items() if k != 'fpo'})}")


if __name__ == "__main__":
    main()
