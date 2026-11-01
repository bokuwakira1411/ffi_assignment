"""FPO v2, iteration k, training step (collection is unchanged: scripts/fpo_collect.py).

Changes vs fpo_train.py (v1), plan in results/round2/configs/fpo_v2.yaml:
  * whole model in fp32 (no autocast, TF32 off) for l_old, l_new, BC loss and the KL probe; the same stored
    (tau, eps) pairs (kept in fp32) for l_old and l_new; clip_frac / early stop on these fp32 ratios;
  * fp32 master weights that persist across iterations: trainable tensors are restored from the checkpoint FILE
    (load_policy casts 129 expert tensors to bf16 modules; the saved fp32 values would otherwise be rounded at
    every iteration boundary). Frozen tensors go back to their original dtype before saving, so collection / eval
    load the checkpoint exactly as in v1;
  * gradient accumulation: one optimizer step = batch_size decisions of ONE task in micro-batches of micro_batch;
  * anchor: + lambda_bc * CFM loss on decisions of the iteration's successful rollouts (bc_batch per step, fresh
    tau / eps), and a KL proxy vs theta0 on a fixed probe set: 0.5 * mean((c0 * (l_theta0 - l_theta))^2) with the
    same (tau, eps), c0 = 1 / mean(l_theta0) on the probe set;
  * at most `epochs` epochs; warmup in samples (warmup_samples), lr, clip, early stop as v1.
Outputs: <out>/pretrained_model, train_log.csv (per step), kl_log.csv, train_info.json, and one row appended to
<out_root>/iter_summary.csv (batch size, steps, epochs used, clip_frac, KL proxy, collection success).

    python scripts/fpo_train_v2.py --config ~/ffi26/results/round2/configs/fpo_v2.yaml \
        --policy <theta_k> --data outputs/fpo_v2/iter0 --out outputs/fpo_v2/iter0/train
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from fpo_train import FREEZE, expand, load_decisions, make_batch, save_policy  # noqa: E402
from rollout import checkpoint_file, load_policy  # noqa: E402

torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def build_policy_fp32(path: str, n_action_steps: int):
    """load_policy (as collection / eval) -> every tensor fp32; trainable tensors restored exactly from the file.
    Returns policy and {name: original dtype} for restoring frozen tensors before saving."""
    policy = load_policy(expand(path), n_action_steps, cfg_overrides=FREEZE)[0]
    orig = {n: p.dtype for n, p in policy.named_parameters()}
    policy.float()
    sd = load_file(str(checkpoint_file(expand(path))))
    n_restored = 0
    with torch.no_grad():
        for n, p in policy.named_parameters():
            if p.requires_grad and n in sd:
                p.copy_(sd[n].float().to(p.device))
                n_restored += int(sd[n].dtype == torch.float32)
    policy.train()
    return policy, orig, n_restored


def cfm(policy, batch, noise, tau):
    """per-sample CFM loss (B,) on executed steps, fp32 (no autocast)."""
    loss, _ = policy.forward(dict(batch), noise=noise, time=tau, reduction="none")
    return loss.float()


def draw_pairs32(policy, n_dec: int, n_pairs: int, seed: int):
    cfg = policy.config
    torch.manual_seed(seed)
    tau = torch.stack([policy.model.sample_time(n_dec, "cpu") for _ in range(n_pairs)], 1)
    eps = torch.randn(n_dec, n_pairs, cfg.chunk_size, cfg.max_action_dim)
    return tau, eps


@torch.no_grad()
def losses_fixed(policy, d, idx, tau, eps, mb, dev):
    """l on decisions idx with the given per-decision tau (n,) and eps (n, chunk, dim), micro-batches of mb."""
    out = torch.zeros(len(idx))
    for s in range(0, len(idx), mb):
        j = slice(s, s + mb)
        b = make_batch(d, idx[j], dev, policy.config.chunk_size)
        out[j] = cfm(policy, b, eps[j].to(dev), tau[j].to(dev)).cpu()
    return out


def load_success_decisions(data_dir: Path) -> dict[int, dict[str, np.ndarray]]:
    """decisions of successful rollouts (any group, incl. rejected / zero advantage), split by task."""
    ro = pd.read_csv(data_dir / "rollouts.csv")
    ok = {(int(r.group), int(re.search(r"_j(\d+)$", r.name).group(1))) for r in ro.itertuples() if r.success}
    per: dict[int, list] = {}
    for f in sorted(data_dir.glob("decisions/g*.npz")):
        d = dict(np.load(f))
        m = np.array([(int(g), int(j)) in ok for g, j in zip(d["group"], d["env"])])
        if m.any():
            per.setdefault(int(d["task_id"][0]), []).append({k: v[m] for k, v in d.items()})
    return {t: {k: np.concatenate([x[k] for x in L]) for k in L[0]} for t, L in per.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_steps", type=int, default=None, help="dry run: stop after n steps")
    ap.add_argument("--max_decisions", type=int, default=None, help="dry run: use at most n decisions per task")
    ap.add_argument("--grad_parts", action="store_true", help="dry run: log PPO and unit-BC grad norms separately")
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
    dev, B, MB = "cuda", f["batch_size"], f["micro_batch"]
    assert B % MB == 0
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()

    data = load_decisions(data_dir)
    bcdata = load_success_decisions(data_dir)
    if args.max_decisions:
        data = {t: {k: v[: args.max_decisions] for k, v in d.items()} for t, d in data.items()}
    adv_all = np.concatenate([d["adv"] for d in data.values()])
    adv_std = float(adv_all.std())

    # KL probe (fixed decisions + (tau, eps)) and l_theta0 on it, with theta0 in fp32
    p0, _, _ = build_policy_fp32(cfg["init_policy"], cfg["n_action_steps"])
    prng = np.random.default_rng(f["seed"] + 17)
    probe = {}
    for t, d in data.items():
        idx = np.sort(prng.choice(len(d["adv"]), min(f["kl_probe_per_task"], len(d["adv"])), replace=False))
        torch.manual_seed(f["seed"] + 31 + t)
        tau = p0.model.sample_time(len(idx), "cpu").float()
        eps = torch.randn(len(idx), p0.config.chunk_size, p0.config.max_action_dim)
        probe[t] = (idx, tau, eps)
    l0 = {t: losses_fixed(p0, data[t], *probe[t], MB, dev) for t in data}
    del p0
    torch.cuda.empty_cache()
    c0 = 1.0 / float(torch.cat(list(l0.values())).mean())

    policy, orig_dtype, n_restored = build_policy_fp32(args.policy, cfg["n_action_steps"])
    params = [p for p in policy.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)

    def kl_proxy():
        lt = torch.cat([losses_fixed(policy, data[t], *probe[t], MB, dev) for t in data])
        lr0 = c0 * (torch.cat([l0[t] for t in data]) - lt).double()
        return {"kl_proxy": float(0.5 * (lr0 ** 2).mean()), "absmean_dl_theta0": float((lr0 / c0).abs().mean())}

    # old losses, fp32, stored (tau, eps)
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
    warm = max(1, math.ceil(f["warmup_samples"] / B))
    kl_start = kl_proxy()
    print(f"[fpo_v2] decisions { {t: len(d['adv']) for t, d in data.items()} }, success decisions "
          f"{ {t: len(d['adv']) for t, d in bcdata.items()} }; B {B} (micro {MB}), steps/epoch {steps_per_epoch}, "
          f"max epochs {f['epochs']}, warmup {warm} steps; old-loss pass {t_old:.0f}s, mean l_old {mean_lold:.5f}, "
          f"c {c:.3f}; fp32 trainable restored from file {n_restored}; KL proxy vs theta0 at start {kl_start}",
          flush=True)

    opt = torch.optim.AdamW(params, lr=f["lr"], betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm))
    rng = np.random.default_rng(f["seed"])
    fh = open(out / "train_log.csv", "w", newline="")
    cols = ["step", "epoch", "task", "loss_ppo", "loss_bc", "ratio_mean", "ratio_min", "ratio_max", "std_logr",
            "clip_frac", "dl_absmean", "l_new", "grad_norm", "gn_ppo", "gn_bc_unit", "lr", "time_s", "gpu_mem_gb"]
    wr = csv.DictWriter(fh, fieldnames=cols)
    wr.writeheader()
    kh = open(out / "kl_log.csv", "w", newline="")
    kw = csv.DictWriter(kh, fieldnames=["step", "kl_proxy", "absmean_dl_theta0"])
    kw.writeheader()
    kw.writerow({"step": 0, **kl_start})
    step, stop, clips, ep_used = 0, False, [], 0.0
    stop_reason = "epochs"
    nmb = B // MB

    def flat_grad():
        return torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros_like(p).flatten() for p in params])

    for ep in range(f["epochs"]):
        batches = []
        for t, d in data.items():
            perm = rng.permutation(len(d["adv"]))
            batches += [(t, perm[i:i + B]) for i in range(0, len(perm) - B + 1, B)]
        rng.shuffle(batches)
        for bi, (t, idx) in enumerate(batches):
            d = data[t]
            tau, eps = pairs[t]
            p = rng.integers(0, f["n_pairs"], len(idx))
            opt.zero_grad(set_to_none=True)
            rs, dls, lns, lp = [], [], [], 0.0
            for s in range(0, B, MB):
                j, pj = idx[s:s + MB], p[s:s + MB]
                b = make_batch(d, j, dev, policy.config.chunk_size)
                lnew = cfm(policy, b, eps[j, pj].to(dev), tau[j, pj].to(dev))
                dl = lold[t][j, pj].to(dev) - lnew
                r = torch.exp((c * dl).clamp(-5, 5))
                A = torch.from_numpy(np.clip(d["adv"][j] / adv_std, -5, 5)).to(dev).float()
                surr = torch.minimum(r * A, r.clamp(1 - f["clip"], 1 + f["clip"]) * A)
                loss = -surr.sum() / B
                loss.backward()
                lp += float(loss)
                rs.append(r.detach()); dls.append(dl.detach()); lns.append(lnew.detach())
            gn_ppo = gn_bc = float("nan")
            if args.grad_parts:
                g_ppo = flat_grad().clone()
                gn_ppo = float(g_ppo.norm())
            lb = 0.0
            bd = bcdata.get(t)
            if f["lambda_bc"] > 0 and bd is not None and len(bd["adv"]) > 0:
                nb = min(f["bc_batch"], len(bd["adv"]))
                jb = rng.choice(len(bd["adv"]), nb, replace=False)
                for s in range(0, nb, MB):
                    jj = jb[s:s + MB]
                    b = make_batch(bd, jj, dev, policy.config.chunk_size)
                    taub = policy.model.sample_time(len(jj), dev).float()
                    epsb = torch.randn(len(jj), policy.config.chunk_size, policy.config.max_action_dim, device=dev)
                    lbc = cfm(policy, b, epsb, taub).sum() / nb
                    (f["lambda_bc"] * lbc).backward()
                    lb += float(lbc)
                if args.grad_parts:
                    gn_bc = float((flat_grad() - g_ppo).norm()) / f["lambda_bc"]
                    del g_ppo
            gn = torch.nn.utils.clip_grad_norm_(params, f["grad_clip"])
            opt.step()
            sched.step()
            step += 1
            r = torch.cat(rs); dl = torch.cat(dls)
            logr = (c * dl).clamp(-5, 5)
            cf = float(((r - 1).abs() > f["clip"]).float().mean())
            clips.append(cf)
            wr.writerow({"step": step, "epoch": ep, "task": t, "loss_ppo": lp, "loss_bc": lb,
                         "ratio_mean": float(r.mean()), "ratio_min": float(r.min()), "ratio_max": float(r.max()),
                         "std_logr": float(logr.std()), "clip_frac": cf, "dl_absmean": float(dl.abs().mean()),
                         "l_new": float(torch.cat(lns).mean()), "grad_norm": float(gn), "gn_ppo": gn_ppo,
                         "gn_bc_unit": gn_bc, "lr": sched.get_last_lr()[0], "time_s": round(time.time() - t0, 1),
                         "gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3)})
            ep_used = ep + (bi + 1) / len(batches)
            if step % f["kl_every"] == 0:
                fh.flush()
                k = kl_proxy()
                kw.writerow({"step": step, **k}); kh.flush()
                print(f"[fpo_v2] step {step} ep {ep_used:.2f} loss {lp:+.4f} bc {lb:.4f} ratio [{float(r.min()):.3f},"
                      f"{float(r.max()):.3f}] std_logr {float(logr.std()):.4f} clip_frac(last10) "
                      f"{np.mean(clips[-10:]):.3f} gn {float(gn):.2f} KL {k['kl_proxy']:.2e} "
                      f"{time.time() - t0:.0f}s", flush=True)
            if args.max_steps is not None and step >= args.max_steps:
                stop, stop_reason = True, "max_steps"
                break
            w = f["early_stop_window"]
            if len(clips) >= w and np.mean(clips[-w:]) > f["early_stop_clip_frac"]:
                print(f"[fpo_v2] early stop at step {step}: clip_frac over last {w} steps {np.mean(clips[-w:]):.3f}",
                      flush=True)
                stop, stop_reason = True, "early_stop"
                break
        if stop:
            break
    fh.close()
    kl_end = kl_proxy()
    kw.writerow({"step": step, **kl_end}); kh.close()
    with torch.no_grad():  # frozen tensors back to their original dtype: collection / eval load as in v1
        for n, p in policy.named_parameters():
            if not p.requires_grad and orig_dtype[n] != torch.float32:
                p.data = p.data.to(orig_dtype[n])
    save_policy(policy, expand(args.policy), out / "pretrained_model")
    log = pd.read_csv(out / "train_log.csv")
    ci = json.load(open(data_dir / "collect_info.json"))
    info = {"policy": args.policy, "data": str(data_dir), "decisions": int(len(adv_all)),
            "success_decisions": int(sum(len(d["adv"]) for d in bcdata.values())), "adv_std": adv_std,
            "batch_size": B, "micro_batch": MB, "steps": step, "steps_per_epoch": steps_per_epoch,
            "epochs_used": ep_used, "stop_reason": stop_reason, "warmup_steps": warm, "ratio_scale_used": c,
            "mean_l_old": mean_lold, "c0_probe": c0, "kl_start": kl_start, "kl_end": kl_end,
            "clip_frac_mean": float(log.clip_frac.mean()), "clip_frac_last10": float(log.clip_frac.tail(10).mean()),
            "step1_ratio": {k: float(log[k].iloc[0]) for k in ("ratio_mean", "ratio_min", "ratio_max", "std_logr", "clip_frac")},
            "collect_success": f"{ci['n_success']}/{ci['n_rollouts']}", "fp32_trainable_restored": n_restored,
            "seconds_total": time.time() - t0, "seconds_old_loss": t_old, "trainable_params": n_train,
            "torch_max_mem_allocated_gb": torch.cuda.max_memory_allocated() / 1e9, "fpo": f}
    json.dump(info, open(out / "train_info.json", "w"), indent=1)
    if args.max_steps is None:
        summ = Path(expand(cfg["out_root"])) / "iter_summary.csv"
        new = not summ.exists()
        with open(summ, "a", newline="") as fs:
            w = csv.writer(fs)
            if new:
                w.writerow(["iter_dir", "batch_size", "steps", "epochs_used", "stop_reason", "clip_frac_mean",
                            "clip_frac_last10", "kl_start", "kl_end", "collect_success", "seconds_train"])
            w.writerow([data_dir.name, B, step, round(ep_used, 3), stop_reason, round(info["clip_frac_mean"], 4),
                        round(info["clip_frac_last10"], 4), f"{kl_start['kl_proxy']:.3e}", f"{kl_end['kl_proxy']:.3e}",
                        info["collect_success"], round(info["seconds_total"])])
    print(f"[done] {json.dumps({k: v for k, v in info.items() if k != 'fpo'})}")


if __name__ == "__main__":
    main()
