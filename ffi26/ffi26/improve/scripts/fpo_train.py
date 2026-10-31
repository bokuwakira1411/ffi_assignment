"""FPO iteration k, steps 3-4: old CFM losses with fixed (tau, eps) pairs, then a PPO-clip update.

For every decision d (model inputs o_d, sampled chunk a_d, executed steps n_exec_d, advantage A_d)
and pair p = (tau_dp, eps_dp), with the CFM loss restricted to the n_exec executed steps
(action_is_pad on the rest):
    l_old[d,p] = l(theta_k; o_d, a_d, tau_dp, eps_dp)          (no grad, before the update)
    r[d,p]     = exp(clamp(c * (l_old - l_new), -5, 5))            (per-pair ratio)
    c = ratio_scale, or 1 / mean(l_old) over the iteration with ratio_scale "auto": with c = 1 the ratio
    of a good action can never exceed exp(l_old) ~ exp(0.035) (never clipped), while a bad action's loss
    can grow without bound (verification V4); with c = 1/mean(l_old), clip 0.2 ~ a 20% loss change.
    L          = - mean_d min(r A^, clip(r, 1-eps_clip, 1+eps_clip) A^),  A^ = clip(A / std(A), -5, 5)
Each training step uses B decisions of ONE task (same language-token length, no padding) and one
random pair per decision. Only the action expert + projections train (vision encoder and VLM frozen,
same flags as lerobot-train in round 1). Old and new losses use the same path: train mode, bf16
autocast. Saves <out>/pretrained_model (policy + processors copied from the input checkpoint) and
<out>/train_log.csv, <out>/train_info.json.

    python scripts/fpo_train.py --config ~/ffi26/results/round2/configs/fpo.yaml \
        --policy lerobot/smolvla_libero --data outputs/fpo/iter0 --out outputs/fpo/iter0/train
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from fpo_capture import KEEP_KEYS  # noqa: E402
from rollout import load_policy  # noqa: E402

FREEZE = {"freeze_vision_encoder": True, "train_expert_only": True, "train_state_proj": True}


def expand(p: str) -> str:
    return str(Path(p).expanduser()) if p.startswith(("~", "/")) else p


def load_decisions(data_dir: Path, keep_zero_adv: bool = False) -> dict[int, dict[str, np.ndarray]]:
    """All decisions of an iteration, split by task (token lengths differ between tasks).
    Rows of rejected groups / zero advantage are dropped unless keep_zero_adv."""
    per: dict[int, list[dict]] = {}
    for f in sorted(glob.glob(str(data_dir / "decisions" / "g*.npz"))):
        d = dict(np.load(f))
        m = np.ones(len(d["adv"]), bool) if keep_zero_adv else (d["rejected"] == 0) & (d["adv"] != 0)
        if m.any():
            per.setdefault(int(d["task_id"][0]), []).append({k: v[m] for k, v in d.items()})
    return {t: {k: np.concatenate([x[k] for x in L]) for k in L[0]} for t, L in per.items()}


def make_batch(d: dict[str, np.ndarray], idx: np.ndarray, device, chunk_size: int) -> dict:
    b = {k: torch.from_numpy(d[k][idx]).to(device) for k in KEEP_KEYS}
    b["observation.language.attention_mask"] = b["observation.language.attention_mask"].bool()
    b["action"] = torch.from_numpy(d["chunk"][idx]).to(device).float()
    n_exec = torch.from_numpy(d["n_exec"][idx]).to(device)
    b["action_is_pad"] = torch.arange(chunk_size, device=device)[None, :] >= n_exec[:, None]
    return b


def cfm_loss(policy, batch, noise, tau, bf16=True) -> torch.Tensor:
    """Per-sample CFM loss (B,) on the executed steps, same call as lerobot-train (reduction='none')."""
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
        loss, _ = policy.forward(dict(batch), noise=noise, time=tau, reduction="none")
    return loss.float()


def draw_pairs(policy, n_dec: int, n_pairs: int, seed: int):
    """tau (D,N) float32, eps (D,N,chunk,max_action_dim) float16 — drawn once, reused for old and new."""
    cfg = policy.config
    torch.manual_seed(seed)
    tau = torch.stack([policy.model.sample_time(n_dec, "cpu") for _ in range(n_pairs)], 1)
    eps = torch.randn(n_dec, n_pairs, cfg.chunk_size, cfg.max_action_dim).half()
    return tau, eps


@torch.no_grad()
def old_losses(policy, data, pairs, bs, device) -> dict[int, torch.Tensor]:
    out = {}
    for t, d in data.items():
        tau, eps = pairs[t]
        n, N = tau.shape
        L = torch.zeros(n, N)
        for p in range(N):
            for s in range(0, n, bs):
                idx = np.arange(s, min(s + bs, n))
                b = make_batch(d, idx, device, policy.config.chunk_size)
                L[idx, p] = cfm_loss(policy, b, eps[idx, p].float().to(device), tau[idx, p].to(device)).cpu()
        out[t] = L
    return out


def save_policy(policy, src: str, out: Path):
    """policy weights/config via save_pretrained + processor files copied from the source checkpoint."""
    out.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(out)
    if Path(src).exists():
        sdir = Path(src)
    else:
        from huggingface_hub import snapshot_download

        sdir = Path(snapshot_download(src, local_files_only=True))
    for f in sdir.iterdir():
        if "processor" in f.name:  # HF cache files are read-only: copy content only, replace any old copy
            (out / f.name).unlink(missing_ok=True)
            shutil.copyfile(f, out / f.name)


def build_policy(path: str, n_action_steps: int):
    """Trainable tensors are upcast to fp32 (master weights). The checkpoint stores 129 of the 155
    trainable tensors (action expert) in bf16; AdamW steps of ~lr = 1e-5 are below bf16 resolution and
    are rounded away (round-1 lerobot-train runs: only ~5.5% of expert elements ever changed)."""
    policy, pre, post, env_pre, env_post = load_policy(expand(path), n_action_steps, cfg_overrides=FREEZE)
    for p in policy.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    policy.train()
    return policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_steps", type=int, default=None, help="smoke test: stop after n steps")
    ap.add_argument("--set", nargs="*", default=[], help="override fpo.<key>=<value> (e.g. lr=2e-6)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    f = cfg["fpo"]
    for kv in args.set:
        k, v = kv.split("=", 1)
        try:
            f[k] = float(v) if any(ch in v for ch in ".e") else int(v)
        except ValueError:
            f[k] = yaml.safe_load(v)
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    policy = build_policy(args.policy, cfg["n_action_steps"])
    params = [p for p in policy.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    n_all = sum(p.numel() for p in policy.parameters())

    data = load_decisions(Path(args.data).expanduser())
    adv_all = np.concatenate([d["adv"] for d in data.values()])
    adv_std = float(adv_all.std())
    pairs = {t: draw_pairs(policy, len(d["adv"]), f["n_pairs"], f["seed"] + t) for t, d in data.items()}
    t_old = time.time()
    lold = old_losses(policy, data, pairs, f["batch_size"], dev)
    t_old = time.time() - t_old
    n_dec = len(adv_all)
    mean_lold = float(np.mean(np.concatenate([v.flatten().numpy() for v in lold.values()])))
    c = 1.0 / mean_lold if f["ratio_scale"] == "auto" else float(f["ratio_scale"])
    steps_per_epoch = sum(len(d["adv"]) // f["batch_size"] for d in data.values())
    total = steps_per_epoch * f["epochs"] if args.max_steps is None else args.max_steps
    print(f"[fpo_train] decisions {n_dec} ({ {t: len(d['adv']) for t, d in data.items()} }), adv std {adv_std:.4f}, "
          f"pos {(adv_all > 0).mean():.2f}; trainable {n_train}/{n_all}; steps/epoch {steps_per_epoch}, "
          f"total {total}; old-loss pass {t_old:.0f}s (mean l_old {np.mean([v.mean() for v in lold.values()]):.4f})",
          flush=True)

    opt = torch.optim.AdamW(params, lr=f["lr"], betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, f["warmup"])))
    rng = np.random.default_rng(f["seed"])
    fh = open(out / "train_log.csv", "w", newline="")
    wr = csv.writer(fh)
    wr.writerow(["step", "epoch", "task", "loss", "ratio_mean", "ratio_min", "ratio_max", "clip_frac",
                 "dl_mean", "dl_absmean", "l_new", "adv_mean", "grad_norm", "lr", "time_s", "gpu_mem_gb"])
    step, stop = 0, False
    for ep in range(f["epochs"]):
        batches = []
        for t, d in data.items():
            perm = rng.permutation(len(d["adv"]))
            batches += [(t, perm[i:i + f["batch_size"]]) for i in range(0, len(perm) - f["batch_size"] + 1, f["batch_size"])]
        rng.shuffle(batches)
        clips = []
        for t, idx in batches:
            d = data[t]
            p = rng.integers(0, f["n_pairs"], len(idx))
            tau, eps = pairs[t]
            b = make_batch(d, idx, dev, policy.config.chunk_size)
            lnew = cfm_loss(policy, b, eps[idx, p].float().to(dev), tau[idx, p].to(dev))
            dl = lold[t][idx, p].to(dev) - lnew
            r = torch.exp((c * dl).clamp(-5, 5))
            A = torch.from_numpy(np.clip(d["adv"][idx] / adv_std, -5, 5)).to(dev).float()
            surr = torch.minimum(r * A, r.clamp(1 - f["clip"], 1 + f["clip"]) * A)
            loss = -surr.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(params, f["grad_clip"])
            opt.step()
            sched.step()
            step += 1
            cf = float(((r - 1).abs() > f["clip"]).float().mean())
            clips.append(cf)
            wr.writerow([step, ep, t, float(loss), float(r.mean()), float(r.min()), float(r.max()), cf,
                         float(dl.mean()), float(dl.abs().mean()), float(lnew.mean()), float(A.mean()), float(gn),
                         sched.get_last_lr()[0], round(time.time() - t0, 1),
                         round(torch.cuda.max_memory_allocated() / 1024**3, 3)])
            if step % 50 == 0:
                fh.flush()
                print(f"[fpo_train] step {step}/{total} loss {float(loss):+.4f} ratio {float(r.mean()):.4f} "
                      f"[{float(r.min()):.3f},{float(r.max()):.3f}] clip_frac(last50) {np.mean(clips[-50:]):.3f} "
                      f"|dl| {float(dl.abs().mean()):.5f} gn {float(gn):.3f}", flush=True)
            if args.max_steps is not None and step >= args.max_steps:
                stop = True
                break
            w = f.get("early_stop_window", 10)
            if len(clips) >= w and np.mean(clips[-w:]) > f["early_stop_clip_frac"]:
                print(f"[fpo_train] early stop at step {step}: clip_frac over last {w} steps {np.mean(clips[-w:]):.3f}", flush=True)
                stop = True
                break
        if stop:
            break
    fh.close()
    save_policy(policy, expand(args.policy), out / "pretrained_model")
    info = {"policy": args.policy, "data": args.data, "decisions": n_dec, "adv_std": adv_std, "steps": step,
            "steps_per_epoch": steps_per_epoch, "ratio_scale_used": c, "mean_l_old": mean_lold, "seconds_total": time.time() - t0, "seconds_old_loss": t_old,
            "trainable_params": n_train, "torch_max_mem_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
            "fpo": f}
    json.dump(info, open(out / "train_info.json", "w"), indent=1)
    print(f"[done] {json.dumps(info)}")


if __name__ == "__main__":
    main()
