"""DSRL step 2, small scale: learn a noise policy for frozen SmolVLA (lerobot/smolvla_libero, expert fp32), task 8.

    python scripts/dsrl_train.py --out outputs/dsrl/run1            (resumes from the last finished iteration)
Paper: Wagenmaker et al., "Steering Your Diffusion Policy with Latent Space Reinforcement Learning" (DSRL), CoRL 2025.
Kept: the base policy is frozen and never differentiated; the RL action is the initial noise of the flow, one decision
per replan. Simplified here (see summary.md):
  * on-policy PPO (clip 0.2) on the Gaussian log-prob of the noise, NOT SAC with a learned critic (paper);
  * actor = mean only, fixed sigma = 1, last layer zero-initialised -> iteration 0 is exactly N(0, I) = official;
  * the actor controls the 7 action dims of the noise (50 x 7 = 350 numbers); the 25 padded dims stay N(0, I)
    (they change the executed chunk only a little: scripts/dsrl_step0_policy.py D4);
  * advantage: group-relative segment advantage of src/reward_r2.py with the FPO v2 reward config
    (results/round2/configs/fpo_v2.yaml), as in scripts/fpo_collect.py; rejected groups get 0;
  * task 8 only, 4 iterations x 128 episodes (16 init states of the FPO task-8 list x 8 rollouts).
Eval: task 8 x 50 init x seeds 3000 / 4000 / 5000 (env seed = seed + init), noise ~ N(actor mean, I); episodes saved
in the eval npz format, labelled with failure_labels.label_episode_v2; exact McNemar vs official and FPO v2 theta4
(outputs/failure_labels/labels.csv, same seeds / init states; their policy noise is different draws).
"""

import argparse
import json
import os
import sys
import time
from math import comb
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import yaml  # noqa: E402

from dsrl_rollout import run_batch_noise, seeded_noise, small_image  # noqa: E402
from failure_labels import label_episode_v2  # noqa: E402
from phase1_logging import save_episode  # noqa: E402
from reward_r2 import RewardR2Config, group_segment_labels  # noqa: E402
from rollout import load_policy, make_vec_env  # noqa: E402

TASK = 8
INITS = [26, 27, 33, 39, 43, 10, 13, 14, 16, 22, 30, 38, 40, 41, 44, 46]  # fpo_v2.yaml task 8 init_ids
GROUP = 8
BATCH = 16  # envs per vector env = 2 groups
ITERS = 4
ACT_DIMS = 7
SEED0 = 7_000_000  # iteration k, group g, rollout j: env seed SEED0 + 100000 k + 1000 g + j
PPO = dict(clip=0.2, lr=1e-4, epochs=10, minibatch=256, target_kl=0.03, max_grad_norm=1.0)
EVAL_SEEDS = [3000, 4000, 5000]


class Actor(nn.Module):
    """64x64 agent view + normalised state -> noise mean (chunk_size x 7)."""

    def __init__(self, chunk: int, state_dim: int = 8):
        super().__init__()
        self.chunk = chunk
        self.cnn = nn.Sequential(nn.Conv2d(3, 32, 5, 2), nn.ReLU(), nn.Conv2d(32, 64, 3, 2), nn.ReLU(),
                                 nn.Conv2d(64, 64, 3, 2), nn.ReLU(), nn.Flatten(), nn.Linear(64 * 6 * 6, 256), nn.ReLU())
        self.st = nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(256 + 64, 256), nn.ReLU(), nn.Linear(256, chunk * ACT_DIMS))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, img_u8, state):
        h = torch.cat([self.cnn(img_u8.float() / 255.0), self.st(state.float())], dim=-1)
        return self.head(h).view(-1, self.chunk, ACT_DIMS)


def logp(mean, x):
    """log N(x; mean, I) summed over the noise dims, without the constant."""
    return -0.5 * ((x - mean) ** 2).sum(dim=(-1, -2))


def mcnemar_exact(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2**n)


def make_noise_fn(actor, dev, shp, seed_of, store):
    """seed_of(env index, replan k) -> int. The actor's mean + seeded N(0, I) on the 7 dims, seeded N(0, I) padding."""
    def nf(k, batch, active):
        img = torch.from_numpy(small_image(batch)).to(dev)
        st = batch["observation.state"].float().to(dev)
        with torch.no_grad():
            mu = actor(img, st).cpu()
        full = torch.stack([seeded_noise(seed_of(j, k), shp) for j in range(len(active))])
        ctrl = mu + full[:, :, :ACT_DIMS]
        full[:, :, :ACT_DIMS] = ctrl
        for j in np.where(active)[0]:
            store.setdefault(j, []).append({"k": k, "ctrl": ctrl[j].numpy(), "mu": mu[j].numpy()})
        return full.to(dev)
    return nf


def collect(it, actor, pol, out, rcfg, c):
    dev = pol[0].config.device
    shp = (pol[0].config.chunk_size, pol[0].config.max_action_dim)
    vec = make_vec_env(TASK, BATCH, log_sim_state=False)
    task_text = vec.call("task_description")[0]
    D = {k: [] for k in ("img", "state", "ctrl", "logp_old", "adv", "group", "env", "step")}
    rows = []
    seed_it = SEED0 + 100000 * it
    for b0 in range(0, len(INITS), BATCH // GROUP):
        gids = list(range(b0, b0 + BATCH // GROUP))
        seeds = [seed_it + 1000 * g + j for g in gids for j in range(GROUP)]
        inits = [INITS[g] for g in gids for _ in range(GROUP)]
        store = {}
        nf = make_noise_fn(actor, dev, shp, lambda j, k, seeds=seeds: seeds[j] * 1000 + k, store)
        res = run_batch_noise(vec, *pol, task_text=task_text, seeds=seeds, init_ids=inits, noise_fn=nf)
        for gi, g in enumerate(gids):
            grp = res[gi * GROUP:(gi + 1) * GROUP]
            lab = group_segment_labels(grp, rcfg, c["min_return_spread"], c["adv_deadzone"])
            for jj, (r, lr) in enumerate(zip(grp, lab["rollouts"])):
                j = gi * GROUP + jj
                for d, (s, im, st) in enumerate(zip(r["dec_step"], r["dec_img"], r["dec_state"])):
                    seg = [x for x in lr["segments"] if x["start"] <= s < x["end"]]
                    a = seg[0]["adv"] if seg and not lab["rejected"] else 0.0
                    rec = store[j][d]
                    D["img"].append(im); D["state"].append(st); D["ctrl"].append(rec["ctrl"])
                    D["logp_old"].append(float(-0.5 * ((rec["ctrl"] - rec["mu"]) ** 2).sum())); D["adv"].append(a)
                    D["group"].append(g); D["env"].append(jj); D["step"].append(int(s))
                rows.append({"iter": it, "group": g, "init": INITS[g], "j": jj, "seed": seeds[j], "success": r["success"],
                             "length": r["length"], "return": lr["return"], "rejected": lab["rejected"]})
        print(f"[collect it{it}] groups {gids}: {sum(r['success'] for r in res)}/{len(res)} success", flush=True)
    vec.close()
    D = {k: np.stack(v) if k in ("img", "state", "ctrl") else np.asarray(v) for k, v in D.items()}
    np.savez_compressed(out / f"iter{it}_data.npz", **D)
    pd.DataFrame(rows).to_csv(out / f"iter{it}_rollouts.csv", index=False)
    return D, pd.DataFrame(rows)


def ppo_update(actor, D, dev, log):
    opt = torch.optim.Adam(actor.parameters(), lr=PPO["lr"])
    adv = D["adv"].astype(np.float32)
    std = adv[adv != 0].std() if (adv != 0).any() else 1.0
    adv = adv / max(std, 1e-6)
    T = lambda x: torch.from_numpy(np.asarray(x)).to(dev)  # noqa: E731
    img, st, ctrl, lpo, A = T(D["img"]), T(D["state"]), T(D["ctrl"]), T(D["logp_old"]).float(), T(adv)
    n = len(A)
    stop = "epochs"
    for ep in range(PPO["epochs"]):
        perm = torch.randperm(n, device=dev)
        kls, cfs, ls = [], [], []
        for i in range(0, n, PPO["minibatch"]):
            b = perm[i:i + PPO["minibatch"]]
            lp = logp(actor(img[b], st[b]), ctrl[b])
            lr_ = lp - lpo[b]
            r = lr_.exp()
            loss = -torch.min(r * A[b], r.clamp(1 - PPO["clip"], 1 + PPO["clip"]) * A[b]).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), PPO["max_grad_norm"])
            opt.step()
            with torch.no_grad():
                kls.append(float(((r - 1) - lr_).mean()))
                cfs.append(float(((r - 1).abs() > PPO["clip"]).float().mean()))
                ls.append(float(loss))
        with torch.no_grad():
            mu_abs = float(torch.cat([actor(img[i:i + 512], st[i:i + 512]).abs().mean().view(1)
                                      for i in range(0, n, 512)]).mean())
        log.append({"epoch": ep, "loss": np.mean(ls), "approx_kl": np.mean(kls), "clip_frac": np.mean(cfs),
                    "mean_abs_mu": mu_abs})
        print(f"  [ppo] epoch {ep} loss {np.mean(ls):+.4f} kl {np.mean(kls):.4f} clip {np.mean(cfs):.3f} |mu| {mu_abs:.4f}",
              flush=True)
        if np.mean(kls) > PPO["target_kl"]:
            stop = f"target_kl at epoch {ep}"
            break
    return stop


def evaluate(actor, pol, out):
    dev = pol[0].config.device
    shp = (pol[0].config.chunk_size, pol[0].config.max_action_dim)
    vec = make_vec_env(TASK, BATCH, log_sim_state=False)
    task_text = vec.call("task_description")[0]
    path = lambda s, k: out / f"eval_s{s}" / "dsrl" / f"task{TASK}" / f"ep{k:03d}"  # noqa: E731
    for s in EVAL_SEEDS:
        path(s, 0).parent.mkdir(parents=True, exist_ok=True)
    todo = [(s, k) for s in EVAL_SEEDS for k in range(50) if not path(s, k).with_suffix(".npz").exists()]
    for b0 in range(0, len(todo), BATCH):
        sk = todo[b0:b0 + BATCH]
        full = sk + [sk[-1]] * (BATCH - len(sk))  # pad the last batch; padded envs are discarded
        nf = make_noise_fn(actor, dev, shp, lambda j, k, full=full: 10**9 + full[j][0] * 1000 + full[j][1] * 100 + k, {})
        res = run_batch_noise(vec, *pol, task_text=task_text, seeds=[s + k for s, k in full],
                              init_ids=[k for _, k in full], noise_fn=nf)
        for (s, k), r in zip(sk, res):
            save_episode(path(s, k), r, video=False)
        print(f"[eval] {sk[0]}..{sk[-1]}: {sum(r['success'] for r in res[:len(sk)])}/{len(sk)}", flush=True)
    vec.close()


def report(out, info):
    from phase1_logging import load_episode
    reg = pd.read_csv(ROOT / "outputs/failure_labels/region_centres.csv")
    rxy = {(r.region, r.object): (r.x, r.y) for r in reg[reg.task == TASK].itertuples()}
    rows = []
    for seed in EVAL_SEEDS:
        for npz in sorted((out / f"eval_s{seed}" / "dsrl" / f"task{TASK}").glob("ep*.npz")):
            log, act, meta = load_episode(npz)
            lab = label_episode_v2(log, act, bool(meta["success"]), region_xy=rxy)
            rows.append({"arm": "dsrl", "seed": seed, "task": TASK, "init": int(npz.stem[2:]),
                         "success": bool(meta["success"]), **lab, "npz": str(npz)})
    df = pd.DataFrame(rows)
    df.to_csv(out / "eval_labels.csv", index=False)
    ref = pd.read_csv(ROOT / "outputs/failure_labels/labels.csv", keep_default_na=False)
    ref = ref[(ref.task == TASK) & ref.seed.isin(EVAL_SEEDS)]
    lines = ["# DSRL nhỏ (task 8): kết quả", "", "Sinh bởi `scripts/dsrl_train.py`. Eval: task 8 × 50 init × seed 3000/4000/5000, "
             "noise ~ N(mean của actor, I). official và v2 θ4 lấy từ `outputs/failure_labels/labels.csv` (cùng seed env, "
             "cùng init state; noise của policy là các lần rút khác).", ""]
    lines += ["## Thu thập theo iteration", "", "| iter | thành công | rejected groups | PPO dừng | approx KL cuối | clip frac cuối | |mu| cuối |",
              "|---|---|---|---|---|---|---|"]
    for x in info["iters"]:
        lines.append(f"| {x['iter']} | {x['collect_success']} | {x['rejected_groups']}/16 | {x['ppo_stop']} | "
                     f"{x['last']['approx_kl']:.4f} | {x['last']['clip_frac']:.3f} | {x['last']['mean_abs_mu']:.4f} |")
    lines += ["", "## Eval", "", "| arm | s3000 | s4000 | s5000 | gộp |", "|---|---|---|---|---|"]
    arms = {"dsrl": df, "official": ref[ref.arm == "official"], "fpo_v2_sel": ref[ref.arm == "fpo_v2_sel"]}
    for name, g in arms.items():
        cells = [f"{int(g[g.seed == s].success.sum())}/{int((g.seed == s).sum())}" for s in EVAL_SEEDS]
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {int(g.success.sum())}/{len(g)} = {100 * g.success.mean():.1f}% |")
    lines += ["", "McNemar chính xác, cặp theo (seed, init):", ""]
    a = df.set_index(["seed", "init"]).success
    for name in ("official", "fpo_v2_sel"):
        b = arms[name].set_index(["seed", "init"]).success.reindex(a.index)
        x, y = int((a & ~b).sum()), int((~a & b).sum())
        lines.append(f"- dsrl vs {name}: chỉ dsrl {x}, chỉ {name} {y}, p = {mcnemar_exact(x, y):.3f}")
    f = df[~df.success]
    prim = f.primary.value_counts().to_dict()
    lines += ["", f"Nhãn chính v2 của các thất bại dsrl ({len(f)}): " + ", ".join(f"{k} {v}" for k, v in prim.items()),
              f"Số ấm đã lên bếp trong thất bại (0/1/2): " + "/".join(str(int((f.n_pred_new_ever_true == i).sum())) for i in (0, 1, 2)),
              "", f"Actor: {info['actor_params']:,} tham số. Episode env: thu thập {info['collect_episodes']}, eval {len(df)}.",
              f"Thời gian: {info.get('seconds', {})}"]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs/dsrl/run1"))
    ap.add_argument("--policy", default="lerobot/smolvla_libero")
    ap.add_argument("--skip-eval", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fcfg = yaml.safe_load(open(ROOT.parent / "results/round2/configs/fpo_v2.yaml"))
    rcfg = RewardR2Config(**fcfg["reward"])
    c = {k: fcfg["collect"][k] for k in ("min_return_spread", "adv_deadzone")}
    assert fcfg["collect"]["tasks"][1]["init_ids"] == INITS and fcfg["collect"]["group_size"] == GROUP
    pol = load_policy(a.policy, 10, expert_fp32=True)
    dev = pol[0].config.device
    torch.manual_seed(0)
    actor = Actor(pol[0].config.chunk_size).to(dev)
    n_par = sum(p.numel() for p in actor.parameters())
    info_p = out / "info.json"
    info = json.load(open(info_p)) if info_p.exists() else {"iters": [], "actor_params": n_par, "ppo": PPO,
                                                              "reward": fcfg["reward"], **c, "seconds": {}}
    start = len(info["iters"])
    if start:
        actor.load_state_dict(torch.load(out / f"actor_iter{start}.pt"))
    else:
        torch.save(actor.state_dict(), out / "actor_iter0.pt")
    for it in range(start, ITERS):
        t0 = time.time()
        D, ro = collect(it, actor, pol, out, rcfg, c)
        t1 = time.time()
        log = []
        stop = ppo_update(actor, D, dev, log)
        torch.save(actor.state_dict(), out / f"actor_iter{it + 1}.pt")
        rej = int(ro.groupby("group").rejected.first().sum())
        info["iters"].append({"iter": it, "collect_success": f"{int(ro.success.sum())}/{len(ro)}", "rejected_groups": rej,
                              "n_decisions": int(len(D["adv"])), "n_adv_nonzero": int((D["adv"] != 0).sum()),
                              "ppo_stop": stop, "ppo_log": log, "last": log[-1],
                              "seconds_collect": t1 - t0, "seconds_ppo": time.time() - t1})
        json.dump(info, open(info_p, "w"), indent=1)
        print(f"[iter {it}] collect {ro.success.sum()}/{len(ro)}, rejected groups {rej}, ppo {stop}, "
              f"{t1 - t0:.0f}s + {time.time() - t1:.0f}s", flush=True)
    info["collect_episodes"] = ITERS * len(INITS) * GROUP
    json.dump(info, open(info_p, "w"), indent=1)
    if not a.skip_eval:
        t0 = time.time()
        evaluate(actor, pol, out)
        info["seconds"]["eval"] = time.time() - t0
        json.dump(info, open(info_p, "w"), indent=1)
        report(out, info)


if __name__ == "__main__":
    main()
