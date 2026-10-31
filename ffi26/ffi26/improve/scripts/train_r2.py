"""Round-2 training wrapper around lerobot-train (LeRobot 0.6.2). The installed package is not
modified; three functions are patched from outside before lerobot_train.main() runs:

  1. lerobot_train._preprocess_dataset_batch: before the preprocessor, look up every sample's
     global frame index (batch["index"]) in the side table written by r2_build.py and
       - mode a1 : append " Advantage: positive|negative" to batch["task"] for labelled frames;
                   for ROLLOUT frames the suffix is dropped with prob --cfg_drop (re-drawn every
                   batch = classifier-free-guidance dropout). Demo frames carry their fixed label.
       - mode a1emb: same labels and CFG dropout as a1, but NO text suffix: the label code goes to a
                   learned embedding added to the action-expert tokens (src/adv_embed.py), attached
                   right after make_policy so the optimizer includes it (fallback for round 1,
                   where the text suffix gave positive - negative = -6 points).
       - mode awr: no suffix at all (plain advantage-weighted BC on the same frames).
     and stash the per-sample weights for (2).
  2. sample_weighting.make_sample_weighter -> returns a weighter that hands back those weights
     (lerobot-train then uses the per-sample loss, sum(w*l)/sum(w); lerobot_train.py:253-269).
     Weights: a1 = side["w_a1"] (round decay x0.5 per older round, pos/neg rebalancing);
              awr = side["w_awr"] (clip(exp(A/beta)) x round decay; demos 1).
  3. lerobot_train.update_policy -> after each step append step, loss, grad_norm, lr, and the
     batch label mix to our own CSV (--loss_csv).

    python scripts/train_r2.py --side <train_root>/r2_frames.parquet --mode a1 \
        --episodes_file <train_root>/train_episodes.json --loss_csv <out>/loss.csv -- <lerobot-train args>
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SUFFIX = {1: " Advantage: positive", 2: " Advantage: negative"}  # label codes: 0 none, 1 pos, 2 neg


RENAME_MAP = '{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'


def args_from_config(cfg_path: str, mode: str, out_override: str | None) -> tuple[dict, list[str]]:
    """Wrapper args + lerobot-train args from a round config (same flags as phase2_train.sh)."""
    import yaml

    cfg = yaml.safe_load(open(cfg_path))
    b, t = cfg["build"], cfg["train"]
    root = Path(b["train_out"]).expanduser()
    out = Path(out_override or t["out"]).expanduser()
    pol = t["init_policy"]
    pol = str(Path(pol).expanduser()) if pol.startswith(("~", "/")) else pol
    wrap = {"side": str(root / "r2_frames.parquet"), "episodes_file": str(root / "train_episodes.json"),
            "loss_csv": str(Path(b["stats_dir"]).expanduser() / f"loss_{mode}.csv"), "cfg_drop": cfg.get("cfg_drop", 0.3)}
    rest = [f"--policy.path={pol}", "--policy.push_to_hub=false", "--policy.freeze_vision_encoder=true",
            "--policy.train_expert_only=true", "--policy.train_state_proj=true",
            f"--policy.optimizer_lr={t['lr']}", f"--policy.scheduler_warmup_steps={t['warmup']}",
            f"--policy.scheduler_decay_steps={t['steps']}", f"--policy.scheduler_decay_lr={t['decay_lr']}",
            f"--dataset.repo_id={b['train_repo_id']}", f"--dataset.root={root}", f"--rename_map={RENAME_MAP}",
            f"--batch_size={t['batch_size']}", f"--steps={t['steps']}", f"--save_freq={t.get('save_freq', 1000)}",
            "--log_freq=50", "--num_workers=8", "--seed=1000", "--accelerator.mixed_precision=bf16",
            "--wandb.enable=false", f"--output_dir={out}"]
    return wrap, rest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="round config; fills every other argument")
    ap.add_argument("--out", default=None, help="with --config: override train.out (e.g. for AWR)")
    ap.add_argument("--side")
    ap.add_argument("--mode", choices=["a1", "a1emb", "awr"], required=True)
    ap.add_argument("--episodes_file")
    ap.add_argument("--loss_csv")
    ap.add_argument("--cfg_drop", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--emb_lr", type=float, default=1e-3,
                    help="a1emb: peak lr of the label embedding (own param group, same warmup/cosine schedule). "
                         "Action-expert tokens have L2 norm ~43; at the base lr 1e-5 a zero-init table could reach ~0.5.")
    ap.add_argument("--dry_run", action="store_true", help="print sizes/epochs and the lerobot-train args, then exit")
    args, rest = ap.parse_known_args()
    if rest and rest[0] == "--":
        rest = rest[1:]
    if args.config:
        wrap, rest_cfg = args_from_config(args.config, args.mode, args.out)
        for k, v in wrap.items():
            setattr(args, k, v)
        rest = rest_cfg + rest

    side = pd.read_parquet(args.side)
    n_index = int(side["index"].max()) + 1
    label = np.zeros(n_index, np.int8)
    is_roll = np.zeros(n_index, bool)
    weight = np.zeros(n_index, np.float32)
    label[side["index"]] = side["label_code"].to_numpy()
    is_roll[side["index"]] = side["src"].eq("rollout").to_numpy()
    wkey = "w_awr" if args.mode == "awr" else "w_a1"
    weight[side["index"]] = side[wkey].to_numpy()
    episodes = json.load(open(args.episodes_file))
    used = side[side.episode_index.isin(episodes)]

    def arg(name, default=None):
        for a in reversed(rest):  # last occurrence wins, as in lerobot-train's parser
            if a.startswith(f"--{name}="):
                return a.split("=", 1)[1]
        return default

    steps, bs = int(arg("steps")), int(arg("batch_size"))
    n = len(used)
    print(f"[train_r2] mode={args.mode} episodes={len(episodes)} frames={n} "
          f"(demo {int((used.src == 'demo').sum())}, rollout {int((used.src == 'rollout').sum())}) "
          f"steps={steps} batch={bs} -> {steps * bs / n:.2f} epochs", flush=True)
    for src, g in used.groupby("src"):
        print(f"[train_r2]   {src}: labels {g.label.value_counts().to_dict()} weight mean {g[wkey].mean():.3f}",
              flush=True)
    if args.dry_run:
        print("[train_r2] lerobot-train", " ".join(rest))
        return

    import lerobot.scripts.lerobot_train as lt
    import lerobot.utils.sample_weighting as sw

    rng = np.random.default_rng(args.seed)
    state = {"w": None, "mix": (0, 0, 0)}

    orig_pre = lt._preprocess_dataset_batch

    def pre(batch, camera_keys, rename_map, preprocessor):
        idx = batch["index"].cpu().numpy().astype(np.int64)
        lab = label[idx].copy()
        if args.mode in ("a1", "a1emb"):
            drop = is_roll[idx] & (rng.random(len(idx)) < args.cfg_drop)
            lab[drop] = 0
            if args.mode == "a1":
                batch["task"] = [t + SUFFIX.get(int(c), "") for t, c in zip(batch["task"], lab)]
            else:
                state["policy"].model._adv_label = torch.as_tensor(lab, dtype=torch.long)
        else:
            lab[:] = 0
        state["mix"] = tuple(int((lab == c).sum()) for c in (1, 2, 0))
        state["w"] = torch.as_tensor(weight[idx], dtype=torch.float32)
        return orig_pre(batch, camera_keys, rename_map, preprocessor)

    class SideWeighter(sw.SampleWeighter):
        def __init__(self, device):
            self.device = device

        def compute_batch_weights(self, batch):
            w = state["w"].to(self.device)
            w = w * (len(w) / w.sum().clamp_min(1e-6))  # normalise to sum = batch size
            return w, {"mean": float(w.mean()), "max": float(w.max())}

        def get_stats(self):
            return {}

    def make_weighter(config, policy, device, **kw):
        return SideWeighter(device)

    orig_update = lt.update_policy
    Path(args.loss_csv).parent.mkdir(parents=True, exist_ok=True)
    fh = open(args.loss_csv, "w", newline="")
    wr = csv.writer(fh)
    wr.writerow(["step", "loss", "grad_norm", "lr", "n_pos", "n_neg", "n_none", "time_s", "gpu_mem_gb"])
    t0 = time.time()
    step = {"n": 0}

    def update(train_metrics, *a, **kw):
        out = orig_update(train_metrics, *a, **kw)
        step["n"] += 1
        m = train_metrics.metrics
        wr.writerow([step["n"], m["loss"].val, m["grad_norm"].val if "grad_norm" in m else "", m["lr"].val,
                     *state["mix"], round(time.time() - t0, 2),
                     round(torch.cuda.max_memory_allocated() / 1024**3, 3) if torch.cuda.is_available() else ""])
        if step["n"] % 50 == 0:
            fh.flush()
        return out

    lt._preprocess_dataset_batch = pre
    sw.make_sample_weighter = make_weighter
    if args.mode == "a1emb":
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from adv_embed import attach

        orig_make_policy = lt.make_policy
        init_from = arg("policy.path")

        def make_policy(*a, **kw):
            policy = orig_make_policy(*a, **kw)
            emb = attach(policy, init_from=init_from if init_from and Path(init_from).exists() else None)
            state["policy"] = policy
            emb_ids = {id(p) for p in emb.parameters()}
            rest_params = [p for p in policy.parameters() if id(p) not in emb_ids]
            policy.get_optim_params = lambda: [{"params": rest_params},
                                               {"params": list(emb.parameters()), "lr": args.emb_lr}]
            n = sum(p.numel() for p in policy.parameters() if p.requires_grad)
            print(f"[train_r2] label embedding attached; learnable params now {n}", flush=True)
            return policy

        lt.make_policy = make_policy
    lt.update_policy = update
    sys.argv = ["lerobot-train", *rest, f"--dataset.episodes={json.dumps(episodes)}",
                "--sample_weighting.type=uniform"]
    try:
        lt.main()
    finally:
        fh.close()


if __name__ == "__main__":
    main()
