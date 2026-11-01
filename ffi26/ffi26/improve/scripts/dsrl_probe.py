"""DSRL step 1: noise probe from stalled states. GPU (policy) + EGL envs.

    python scripts/dsrl_probe.py --out outputs/dsrl/probe [--n-eps 10] [--videos 3]
Episodes (outputs/failure_labels/labels.csv, random with seed 20261101):
  10 official failures, task 8, eval seeds 3000-5000, with an E_contact stall that starts after the first moka pot
     is on the stove; 10 FPO v2 theta4 failures, task 0, with an E_free stall.
Start state: the episode's executed actions are replayed open-loop from reset (exact, scripts/dsrl_step0_env.py)
up to row t0 = floor((stall start - 20) / 10) * 10 (a replan boundary); the sim state there is the start state.
Step budget from t0: 520 - t0 (same end as the original episode). Policy = the episode's arm, expert fp32.
Methods, 16 rollouts each:
  N1, N1.5, N2   every chunk's noise ~ N(0, s^2 I), s = 1, 1.5, 2
  CEM            diagonal-Gaussian CEM over the noise of the next 5 chunks (5 x 50 x 32): 3 rounds x 16 candidates,
                 4 elites, std floor 0.1; a candidate is scored after its 5 chunks (50 steps):
                 score = 10 * (Phi_50 - Phi_0) + 1[a goal object lifted] + 0.5 * 1[EE moving: mean step > 2 mm over
                 steps 20-50]. Then 16 full rollouts: first 5 chunks ~ N(mu, std^2) of the final CEM round, rest N(0, I).
Per full rollout: escaped (no EE stall run, < 2 mm/step for >= 50 steps, starting in the first 100 steps),
Phi up (Phi at the end > Phi at t0), success. Noise of rollout j comes from its own seeded generator (src/dsrl_rollout).
Outputs: episodes.csv, rollouts.csv, cem_rounds.csv, summary.csv, summary.md, videos/*.mp4 (header: step and Phi).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from lerobot.envs.utils import NEW_ROLLOUT_OPTION  # noqa: E402
from lerobot.utils.io_utils import write_video  # noqa: E402

from dsrl_rollout import run_batch_noise, seeded_noise  # noqa: E402
from phase1_logging import load_episode  # noqa: E402
from phase3_eval_video import annotate  # noqa: E402
from reward import lifted_mask, stall_segments  # noqa: E402
from rollout import MAX_STEPS, load_policy, make_vec_env  # noqa: E402

POLICIES = {"official": "lerobot/smolvla_libero",
            "fpo_v2_sel": "/home/student/ffi26/improve/outputs/fpo_v2/iter3/train/pretrained_model"}
G = 16
SEL_SEED = 20261101
CEM_CHUNKS, CEM_ROUNDS, CEM_ELITES, CEM_STD_MIN = 5, 3, 4, 0.1
METHODS = [("N1", 1.0), ("N1.5", 1.5), ("N2", 2.0)]


def segs(s, kind):
    return [(int(a.split("-")[0]), int(a.split("-")[1].split(":")[0])) for a in s.split(";") if a.endswith(kind)]


def pick_episodes(n):
    lab = pd.read_csv(ROOT / "outputs/failure_labels/labels.csv", keep_default_na=False)
    rows = []
    for r in lab[(lab.arm == "official") & (lab.task == 8) & ~lab.success & lab.lab_E_contact].itertuples():
        P = np.load(r.npz)["preds"]
        t1 = min(int(np.argmax(P[:, i])) if P[:, i].any() else 10**9 for i in (0, 1))
        sg = [s for s in segs(r.E_segments, "contact") if s[0] >= t1]
        if sg:
            rows.append(dict(set="t8_official_Econtact", arm=r.arm, seed=r.seed, task=8, init=r.init,
                             stall_start=sg[0][0], stall_end=sg[0][1], npz=r.npz))
    a = pd.DataFrame(rows)
    rows = []
    for r in lab[(lab.arm == "fpo_v2_sel") & (lab.task == 0) & ~lab.success & lab.lab_E_free].itertuples():
        sg = segs(r.E_segments, "free")
        rows.append(dict(set="t0_v2_Efree", arm=r.arm, seed=r.seed, task=0, init=r.init, stall_start=sg[0][0],
                         stall_end=sg[0][1], npz=r.npz))
    b = pd.DataFrame(rows)
    rng = np.random.default_rng(SEL_SEED)
    pick = pd.concat([a.iloc[np.sort(rng.choice(len(a), n, replace=False))],
                      b.iloc[np.sort(rng.choice(len(b), n, replace=False))]]).reset_index(drop=True)
    pick["t0"] = ((pick.stall_start - 20).clip(lower=0) // 10) * 10
    pick.insert(0, "ep_id", range(len(pick)))
    return pick, len(a), len(b)


def start_state(task, init, seed, actions, t0):
    vec = make_vec_env(task, 1, log_sim_state=True)
    vec.set_attr("next_init_state_id", [init])
    vec.set_attr("next_sim_state", [None])
    vec.reset(seed=[seed + init], options={NEW_ROLLOUT_OPTION: True})
    for a in actions[:t0]:
        vec.step(a[None])
    log = vec.call("get_episode_log")[0]
    vec.close()
    return log["sim_state"][t0], log


def metrics(rec):
    log = rec["log"]
    eef = np.asarray(log["eef_pos"])
    phi = np.asarray(log["phi"])
    st = stall_segments(eef, 0.002, 50)
    goal = np.asarray(log["goal_obj_mask"], bool)
    lifted = bool(lifted_mask(log)[:, goal].any())
    return {"success": rec["success"], "length": rec["length"], "phi0": float(phi[0]), "phi_end": float(phi[-1]),
            "phi_up": bool(phi[-1] > phi[0] + 1e-9), "escaped": not any(s < 100 for s, _ in st), "lifted": lifted}


def cem_score(rec):
    log = rec["log"]
    phi = np.asarray(log["phi"])
    eef = np.asarray(log["eef_pos"])
    goal = np.asarray(log["goal_obj_mask"], bool)
    v = np.linalg.norm(np.diff(eef, axis=0), axis=1)
    moving = bool(v[20:50].mean() > 0.002) if len(v) >= 50 else bool(v[20:].mean() > 0.002) if len(v) > 20 else False
    return 10 * (phi[-1] - phi[0]) + float(lifted_mask(log)[:, goal].any()) + 0.5 * moving


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs/dsrl/probe"))
    ap.add_argument("--n-eps", type=int, default=10)
    ap.add_argument("--videos", type=int, default=3)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    eps, n_cand8, n_cand0 = pick_episodes(a.n_eps)
    eps.to_csv(out / "episodes.csv", index=False)
    print(f"candidates: task 8 official E_contact after 1 pot {n_cand8}, task 0 v2 E_free {n_cand0}", flush=True)
    rows, cem_rows = [], []
    done_eps = set()
    if (out / "rollouts.csv").exists():  # resume
        old = pd.read_csv(out / "rollouts.csv")
        rows = old.to_dict("records")
        done_eps = {e for e, g in old.groupby("ep_id") if set(g.method) >= {"N1", "N1.5", "N2", "CEM"}}
        rows = [r for r in rows if r["ep_id"] in done_eps]
        if (out / "cem_rounds.csv").exists():
            cem_rows = [r for r in pd.read_csv(out / "cem_rounds.csv").to_dict("records") if r["ep_id"] in done_eps]
    t_all = time.time()
    for arm in ("official", "fpo_v2_sel"):
        todo = eps[(eps.arm == arm) & ~eps.ep_id.isin(done_eps)]
        if todo.empty:
            continue
        pol = load_policy(POLICIES[arm], 10, expert_fp32=True)
        cfg = pol[0].config
        shp = (cfg.chunk_size, cfg.max_action_dim)
        dev = cfg.device
        for e in todo.itertuples():
            t_ep = time.time()
            log0, actions, meta = load_episode(Path(e.npz))
            st, rlog = start_state(e.task, e.init, e.seed, actions, e.t0)
            dE = float(np.abs(rlog["eef_pos"][: e.t0 + 1] - log0["eef_pos"][: e.t0 + 1]).max())
            budget = MAX_STEPS - e.t0
            vec = make_vec_env(e.task, G, log_sim_state=False)
            task_text = vec.call("task_description")[0]
            base = 10**7 * (e.ep_id + 1)

            def run(noise_fn, steps):
                return run_batch_noise(vec, *pol, task_text=task_text, seeds=[e.seed + e.init] * G,
                                       init_ids=[e.init] * G, sim_states=[st] * G, max_steps=steps, noise_fn=noise_fn)

            for mi, (m, s) in enumerate(METHODS):
                def nf(k, batch, active, mi=mi, s=s):
                    return torch.stack([seeded_noise(base + 10**6 * mi + 1000 * j + k, shp, s) for j in range(G)]).to(dev)
                for j, rec in enumerate(run(nf, budget)):
                    rows.append({"ep_id": e.ep_id, "set": e.set, "method": m, "j": j, **metrics(rec)})
            mu = torch.zeros(CEM_CHUNKS, *shp)
            sd = torch.ones(CEM_CHUNKS, *shp)
            for rd in range(CEM_ROUNDS):
                eps_ = [seeded_noise(base + 10**6 * (10 + rd) + j, (CEM_CHUNKS, *shp)) for j in range(G)]
                cand = [mu + sd * x for x in eps_]

                def nf(k, batch, active, cand=cand, rd=rd):
                    return torch.stack([cand[j][k] if k < CEM_CHUNKS else
                                        seeded_noise(base + 10**6 * (20 + rd) + 1000 * j + k, shp)
                                        for j in range(G)]).to(dev)
                res = run(nf, min(CEM_CHUNKS * 10, budget))
                sc = np.array([cem_score(r) for r in res])
                for j, r in enumerate(res):
                    cem_rows.append({"ep_id": e.ep_id, "round": rd, "j": j, "score": sc[j], **metrics(r)})
                el = torch.stack([cand[j] for j in np.argsort(-sc)[:CEM_ELITES]])
                mu, sd = el.mean(0), el.std(0).clamp(min=CEM_STD_MIN)
            fin = [mu + sd * seeded_noise(base + 10**6 * 30 + j, (CEM_CHUNKS, *shp)) for j in range(G)]

            def nf(k, batch, active, fin=fin):
                return torch.stack([fin[j][k] if k < CEM_CHUNKS else seeded_noise(base + 10**6 * 31 + 1000 * j + k, shp)
                                    for j in range(G)]).to(dev)
            for j, rec in enumerate(run(nf, budget)):
                rows.append({"ep_id": e.ep_id, "set": e.set, "method": "CEM", "j": j, **metrics(rec)})
            torch.save({"mu": mu, "sd": sd}, out / f"cem_ep{e.ep_id:02d}.pt")
            vec.close()
            pd.DataFrame(rows).to_csv(out / "rollouts.csv", index=False)
            pd.DataFrame(cem_rows).to_csv(out / "cem_rounds.csv", index=False)
            r = pd.DataFrame(rows)
            r = r[r.ep_id == e.ep_id].groupby("method")[["escaped", "phi_up", "success"]].sum()
            print(f"[ep {e.ep_id}] {e.set} s{e.seed} init {e.init} t0 {e.t0} (replay max|dEE| to t0 {dE:.2e}) "
                  f"{time.time() - t_ep:.0f}s\n{r.to_string()}", flush=True)
        del pol
        torch.cuda.empty_cache()
    summarize(out, eps)
    if a.videos:
        videos(out, eps, a.videos)
    print(f"[done] {time.time() - t_all:.0f}s", flush=True)


def summarize(out, eps):
    r = pd.read_csv(out / "rollouts.csv")
    s = r.groupby(["set", "method"]).agg(n=("success", "size"), escaped=("escaped", "sum"), phi_up=("phi_up", "sum"),
                                         success=("success", "sum"),
                                         eps_with_success=("success", lambda x: 0)).reset_index()
    s["eps_with_success"] = [r[(r.set == a) & (r.method == m)].groupby("ep_id").success.any().sum()
                             for a, m in zip(s.set, s.method)]
    s.to_csv(out / "summary.csv", index=False)
    per = r.groupby(["ep_id", "method"]).success.sum().unstack().reindex(columns=["N1", "N1.5", "N2", "CEM"])
    lines = ["# Noise probe (DSRL bước 1)", "", "Sinh bởi `scripts/dsrl_probe.py`. Mỗi ô: số rollout / tổng (16 rollout × số episode).", "",
             "| tập | cách | thoát stall | Φ tăng | thành công | số episode có ≥ 1 thành công |", "|---|---|---|---|---|---|"]
    for x in s.itertuples():
        lines.append(f"| {x.set} | {x.method} | {x.escaped}/{x.n} | {x.phi_up}/{x.n} | {x.success}/{x.n} | "
                     f"{x.eps_with_success}/{x.n // 16} |")
    lines += ["", "## Thành công theo episode (/16)", "", "| ep | tập | seed | init | t0 | N1 | N1.5 | N2 | CEM |",
              "|---|---|---|---|---|---|---|---|---|"]
    for e in eps.itertuples():
        if e.ep_id in per.index:
            p = per.loc[e.ep_id]
            lines.append(f"| {e.ep_id} | {e.set} | {e.seed} | {e.init} | {e.t0} | " + " | ".join(
                "—" if pd.isna(p[m]) else str(int(p[m])) for m in ["N1", "N1.5", "N2", "CEM"]) + " |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")


def videos(out, eps, n):
    """Re-run single rollouts (same seeded noise, batch of 1) with frames; outcome is compared with the batched run."""
    r = pd.read_csv(out / "rollouts.csv")
    want = []
    for st in ("t8_official_Econtact", "t0_v2_Efree"):
        x = r[(r.set == st) & r.success]
        if len(x):
            want.append(x.iloc[0])
    x = r[(r.method == "N1") & ~r.success]
    if len(x):
        want.append(x.iloc[0])
    vdir = out / "videos"
    vdir.mkdir(exist_ok=True)
    chk = []
    for w in want[:n]:
        e = eps[eps.ep_id == w.ep_id].iloc[0]
        pol = load_policy(POLICIES[e.arm], 10, expert_fp32=True)
        cfg = pol[0].config
        shp = (cfg.chunk_size, cfg.max_action_dim)
        log0, actions, _ = load_episode(Path(e.npz))
        st, _ = start_state(e.task, e.init, e.seed, actions, e.t0)
        base, j = 10**7 * (e.ep_id + 1), int(w.j)
        if w.method == "CEM":
            c = torch.load(out / f"cem_ep{e.ep_id:02d}.pt")
            fin = c["mu"] + c["sd"] * seeded_noise(base + 10**6 * 30 + j, (CEM_CHUNKS, *shp))

            def nf(k, batch, active):
                return (fin[k] if k < CEM_CHUNKS else seeded_noise(base + 10**6 * 31 + 1000 * j + k, shp))[None].to(cfg.device)
        else:
            mi, s = next((i, s) for i, (m, s) in enumerate(METHODS) if m == w.method)

            def nf(k, batch, active):
                return seeded_noise(base + 10**6 * mi + 1000 * j + k, shp, s)[None].to(cfg.device)
        vec = make_vec_env(e.task, 1, log_sim_state=False)
        rec = run_batch_noise(vec, *pol, task_text=vec.call("task_description")[0], seeds=[e.seed + e.init],
                              init_ids=[e.init], sim_states=[st], max_steps=MAX_STEPS - e.t0, noise_fn=nf,
                              record_frames=True)[0]
        vec.close()
        fn = f"ep{e.ep_id:02d}_{e.set}_{w.method}_j{j}.mp4"
        write_video(str(vdir / fn), annotate(rec["frames"], rec["log"], [], arm="", task=e.task, init=e.init,
                                             seed=e.seed, success=rec["success"], blind=True), fps=20)
        chk.append({"file": fn, "ep_id": int(e.ep_id), "method": w.method, "j": j, "t0": int(e.t0),
                    "success_batched": bool(w.success), "success_replay": rec["success"],
                    "length_batched": int(w.length), "length_replay": rec["length"]})
        del pol
        torch.cuda.empty_cache()
    pd.DataFrame(chk).to_csv(vdir / "videos.csv", index=False)
    print(json.dumps(chk, indent=1), flush=True)


if __name__ == "__main__":
    main()
