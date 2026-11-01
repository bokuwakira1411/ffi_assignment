"""Demo: run one or more methods on one LIBERO-10 episode and save annotated videos (GPU; ~1–3 min per method).

Uses the selected checkpoint and the original inference settings of each method from demo/methods.yaml, and the existing
replay script improve/scripts/phase3_eval_video.py (no new inference code). Episode k of an eval seed uses init state k
and env seed seed + k, with the policy RNG reset by set_seed(seed); batches 0..k//8 are re-run, so if the method was
evaluated on that (seed, task) the replay must reproduce the original episode (written to replay_check_<method>.csv,
column `match`). Needs: the checkpoint directory, the HF cache (~/.cache/huggingface: lerobot/smolvla_libero,
HuggingFaceTB/SmolVLM2-500M-Video-Instruct), LIBERO assets (~/.cache/libero), conda env `lerobot`. It does NOT need
decision caches (iter*/decisions) or datasets.

    conda activate lerobot && cd ~/ffi26
    python demo/run_demo.py --methods official,fpo_v4 --task 0 --init 4 --seed 5000
    python demo/run_demo.py --list
Output: demo/out/t{task}_init{k}_s{seed}/{method}_t{task}_init{k}_s{seed}_{success|fail}.mp4, replay_check_*.csv,
episodes_*.csv, config.yaml.
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import yaml

H = Path.home() / "ffi26"
REG = H / "demo/methods.yaml"


def resolve_original(spec, task):
    """first candidate template whose directory exists for this task (template keeps the {task} placeholder)"""
    for c in (spec if isinstance(spec, list) else [spec]):
        if (H / c.format(task=task)).exists():
            return str(H / c)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="official,fpo_v4")
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--init", type=int, default=4)
    ap.add_argument("--seed", type=int, default=5000, help="eval seed (3000/4000/5000: tasks 0, 8; 1000: all tasks)")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    reg = yaml.safe_load(open(REG))["methods"]
    if args.list:
        for k, v in reg.items():
            print(f"{k:15s} {v['label']}  [{v['path']}]  seeds with originals: {sorted(v['originals'])}")
        return
    names = args.methods.split(",")
    bad = [n for n in names if n not in reg]
    if bad:
        sys.exit(f"unknown method(s) {bad}; see --list")
    out = H / f"demo/out/t{args.task}_init{args.init}_s{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    arms, originals = [], {}
    for n in names:
        m = reg[n]
        path = m["path"] if m["path"].startswith("lerobot/") else str(H / m["path"])
        arm = {"name": n, "label": m["label"], "path": path, "expert_fp32": bool(m["expert_fp32"])}
        for k in ("task_suffix", "adv_label"):
            if k in m:
                arm[k] = m[k]
        arms.append(arm)
        o = resolve_original(m["originals"].get(args.seed), args.task) if args.seed in m["originals"] else None
        if o:
            originals[n] = {args.seed: o}
        else:
            print(f"[note] {n}: no original eval episode for seed {args.seed}, task {args.task}; the video is a new rollout "
                  f"(not a replay check)")
    # same number of episodes per (model, task) as the original evals, so batch composition (and RNG use) is identical:
    # tasks 0, 8: 50 episodes; other tasks: 20 (eval configs in methods.yaml / eval_config)
    n_ep = 50 if args.task in (0, 8) else 20
    if args.init >= n_ep:
        sys.exit(f"init {args.init} >= {n_ep} episodes evaluated for task {args.task}")
    cfg = {"seed_note": "episode k uses init state k and env seed seed+k", "n_action_steps": 10, "batch_size": 8,
           "n_episodes": n_ep, "output_dir": str(out), "arms": arms, "originals": originals,
           "episodes": [{"seed": args.seed, "task": args.task, "init": args.init, "cat": "demo"}]}
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    cmd = [sys.executable, str(H / "improve/scripts/phase3_eval_video.py"), "--config", str(out / "config.yaml")]
    print("[run]", " ".join(cmd), flush=True)
    rc = subprocess.call(cmd, cwd=H / "improve")
    for n in names:
        chk = out / f"replay_check_{n}.csv"
        if chk.exists():
            for d in csv.DictReader(open(chk)):
                if d["init"] == str(args.init):
                    print(f"[result] {n}: success={d['success_replay']} length={d['length_replay']} "
                          f"original={d['success_orig']} match={d['match']}")
    print("[videos]", *sorted(p.name for p in out.glob("*.mp4")), sep="\n  ")
    sys.exit(rc)


if __name__ == "__main__":
    main()
