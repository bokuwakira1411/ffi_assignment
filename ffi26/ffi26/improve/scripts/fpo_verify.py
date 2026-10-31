"""FPO checks V1-V5 + save/load part of V6 (plan: results/round2/fpo_plan.md §5), on a smoke collection.

  V1 chunk replay: stored inputs + regenerated noise -> sample_actions reproduces the stored chunk;
     post/env_post of the stored chunk reproduces the executed actions in the rollout log.
  V2 deterministic loss: l_old computed twice with the same (tau, eps) is identical.
  V3 ratio = 1 at step 0: l from the old-loss pass vs l recomputed in a different batch composition
     (training batches mix decisions differently) -> numerical noise floor of l_old - l_new.
  V4 gradient sign: A = +1 on half of 32 decisions, -1 on the other half, 20 steps at lr 1e-4:
     l must fall on the + half and rise on the - half.
  V5 frozen weights: vision/VLM tensors unchanged after V4, trainable tensors changed.
  V6a save/load: the V4 policy saved with save_policy and reloaded with load_policy(NAS=10) gives
     the same chunks as the in-memory policy; n_action_steps == 10.

    python scripts/fpo_verify.py --config ~/ffi26/results/round2/configs/fpo.yaml --data outputs/fpo/smoke
Writes <data>/verify.json.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from fpo_capture import ChunkCapture  # noqa: E402
from fpo_train import build_policy, cfm_loss, draw_pairs, load_decisions, make_batch, old_losses, save_policy  # noqa: E402
from phase1_logging import load_episode  # noqa: E402
from rollout import load_policy  # noqa: E402

DEV = "cuda"


@torch.no_grad()
def replay(policy, d, idx, group_size):
    """Re-sample chunks for decisions idx with their original noise rows."""
    b = make_batch(d, idx, DEV, policy.config.chunk_size)
    cap = ChunkCapture.__new__(ChunkCapture)
    cap.policy = policy
    noise = torch.stack([cap.noise(int(s), group_size, DEV)[int(j)] for s, j in zip(d["noise_seed"][idx], d["env"][idx])])
    images, img_masks = policy.prepare_images(b)
    state = policy.prepare_state(b)
    a = policy.model.sample_actions(images, img_masks, b["observation.language.tokens"],
                                    b["observation.language.attention_mask"], state, noise=noise)
    return a[:, :, : d["chunk"].shape[-1]].float().cpu().numpy()


def sha(t: torch.Tensor) -> str:
    return hashlib.sha1(t.detach().float().cpu().numpy().tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    G = cfg["collect"]["group_size"]
    data_dir = Path(args.data).expanduser()
    data = load_decisions(data_dir, keep_zero_adv=True)
    res = {}
    rng = np.random.default_rng(0)

    # ---- V1 (eval-mode policy, exactly as in collection)
    policy, pre, post, env_pre, env_post = load_policy(cfg["init_policy"], cfg["n_action_steps"])
    v1 = {}
    for t, d in data.items():
        idx = np.sort(rng.choice(len(d["env"]), min(16, len(d["env"])), replace=False))
        a = replay(policy, d, idx, G)
        a2 = replay(policy, d, idx, G)
        dw = dict(d)
        dw["noise_seed"] = d["noise_seed"] + 7  # wrong noise, same inputs
        aw = replay(policy, dw, idx, G)
        chunk_err = float(np.abs(a - d["chunk"][idx]).mean())
        rerun_err = float(np.abs(a - a2).mean())
        wrong_err = float(np.abs(aw - d["chunk"][idx]).mean())
        act_err = 0.0
        for i in idx:
            name = f"it*_task{t}_g{int(d['group'][i]):02d}_j{int(d['env'][i])}.npz"
            f = sorted((data_dir / "logs").glob(name))[0]
            _, actions, _ = load_episode(f)
            s, n = int(d["step"][i]), int(d["n_exec"][i])
            ex = []
            for k in range(n):  # postprocessor is applied per executed step in run_batch
                x = post(torch.from_numpy(d["chunk"][i, k:k + 1]).to(DEV))
                ex.append(env_post({"action": x})["action"].cpu().numpy()[0])
            act_err = max(act_err, float(np.abs(np.stack(ex) - actions[s:s + n]).max()))
        v1[t] = {"n": len(idx), "chunk_mean_abs_err": chunk_err, "chunk_max_abs_err": float(np.abs(a - d["chunk"][idx]).max()),
                 "rerun_mean_abs_diff": rerun_err, "wrong_noise_mean_abs_err": wrong_err,
                 "executed_action_max_abs_err": act_err}
    # bf16 inference is not bit-reproducible across batch compositions: pass = replay error at the bf16
    # noise level (< 0.01 mean, chunk std ~0.77) and >= 10x below the error with the wrong noise
    res["V1"] = {"per_task": v1, "pass": all(v["chunk_mean_abs_err"] < 0.01 and 10 * v["chunk_mean_abs_err"] < v["wrong_noise_mean_abs_err"]
                                             and v["executed_action_max_abs_err"] < 1e-4 for v in v1.values())}
    print("V1", json.dumps(res["V1"]), flush=True)
    del policy
    torch.cuda.empty_cache()

    # ---- V2 / V3 (training-mode policy, frozen flags)
    ref = load_policy(cfg["init_policy"], cfg["n_action_steps"])[0]
    bf16_names = {n for n, p in ref.named_parameters() if p.dtype == torch.bfloat16}
    del ref
    policy = build_policy(cfg["init_policy"], cfg["n_action_steps"])
    res["dtypes_trainable_after_upcast"] = sorted({str(p.dtype) for p in policy.parameters() if p.requires_grad})
    t = max(data, key=lambda k: len(data[k]["adv"]))
    d = {k: v[:64] for k, v in data[t].items()}
    pairs = {t: draw_pairs(policy, 64, 2, 123)}
    L1 = old_losses(policy, {t: d}, pairs, 32, DEV)[t]
    L2 = old_losses(policy, {t: d}, pairs, 32, DEV)[t]
    res["V2"] = {"max_abs_diff": float((L1 - L2).abs().max()), "pass": float((L1 - L2).abs().max()) < 1e-6}
    L3 = old_losses(policy, {t: d}, pairs, 8, DEV)[t]  # different batch composition
    perm = rng.permutation(64)[:32]
    tau, eps = pairs[t]
    with torch.no_grad():
        l4 = cfm_loss(policy, make_batch(d, perm, DEV, policy.config.chunk_size), eps[perm, 0].float().to(DEV),
                      tau[perm, 0].to(DEV)).cpu()
    floor = max(float((L1 - L3).abs().mean()), float((L1[perm, 0] - l4).abs().mean()))
    res["V3"] = {"mean_abs_dl_bs8": float((L1 - L3).abs().mean()), "mean_abs_dl_shuffled": float((L1[perm, 0] - l4).abs().mean()),
                 "max_abs_dl": float(max((L1 - L3).abs().max(), (L1[perm, 0] - l4).abs().max())),
                 "mean_l_old": float(L1.mean()), "pass": floor < 1e-3}
    print("V2", json.dumps(res["V2"]), "\nV3", json.dumps(res["V3"]), flush=True)

    # ---- V4 / V5
    frozen = {n: sha(p) for n, p in policy.named_parameters() if not p.requires_grad}
    train0 = {n: sha(p) for n, p in policy.named_parameters() if p.requires_grad}
    params = [p for p in policy.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-5, betas=(0.9, 0.95), weight_decay=0)
    c = 1.0 / float(L1.mean())
    idx = np.arange(32)
    A = torch.tensor([1.0] * 16 + [-1.0] * 16, device=DEV)
    b = make_batch(d, idx, DEV, policy.config.chunk_size)
    nz, tu = eps[idx, 0].float().to(DEV), tau[idx, 0].to(DEV)
    l_before = L1[idx, 0].to(DEV)
    for _ in range(30):
        lnew = cfm_loss(policy, b, nz, tu)
        r = torch.exp((c * (l_before - lnew)).clamp(-5, 5))
        loss = -torch.minimum(r * A, r.clamp(0.8, 1.2) * A).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 10.0)
        opt.step()
    with torch.no_grad():
        l_after = cfm_loss(policy, b, nz, tu)
    dpos = float((l_after[:16] - l_before[:16]).mean())
    dneg = float((l_after[16:] - l_before[16:]).mean())
    res["V4"] = {"setup": "30 steps, lr 1e-5, c = 1/mean(l_old), clip 0.2", "l_before_mean": float(l_before.mean()),
                 "dl_positive_half": dpos, "dl_negative_half": dneg,
                 "ratio_after_pos": float(torch.exp(c * (l_before[:16] - l_after[:16])).mean()),
                 "ratio_after_neg": float(torch.exp(c * (l_before[16:] - l_after[16:])).mean()), "pass": dpos < 0 < dneg}
    changed_frozen = [n for n, p in policy.named_parameters() if not p.requires_grad and sha(p) != frozen[n]]
    changed_train = [n for n, p in policy.named_parameters() if p.requires_grad and sha(p) != train0[n]]
    res["V5"] = {"n_frozen": len(frozen), "n_trainable": len(train0), "frozen_changed": changed_frozen[:10],
                 "trainable_changed": len(changed_train),
                 "trainable_example": sorted(train0)[:3],
                 "pass": not changed_frozen and len(changed_train) > 0}
    print("V4", json.dumps(res["V4"]), "\nV5", json.dumps(res["V5"]), flush=True)

    # ---- V6a save / load
    tmp = data_dir / "verify_ckpt"
    save_policy(policy, cfg["init_policy"], tmp)
    for n, p in policy.named_parameters():  # in-memory weights -> checkpoint dtypes (what load_policy produces)
        if p.requires_grad and n in bf16_names:
            p.data = p.data.to(torch.bfloat16)
    policy.eval()
    a_mem = replay(policy, d, np.arange(8), G)
    del policy, opt
    torch.cuda.empty_cache()
    pol2 = load_policy(str(tmp), cfg["n_action_steps"])[0]
    a_load = replay(pol2, d, np.arange(8), G)
    err = float(np.abs(a_mem - a_load).max())
    res["V6a"] = {"chunk_max_abs_err_mem_vs_loaded": err, "n_action_steps_loaded": pol2.config.n_action_steps,
                  "differs_from_official": float(np.abs(a_load - d["chunk"][:8]).max()),
                  "pass": err < 1e-3 and pol2.config.n_action_steps == 10}
    print("V6a", json.dumps(res["V6a"]), flush=True)
    res["all_pass"] = all(v["pass"] for v in res.values() if isinstance(v, dict) and "pass" in v)
    json.dump(res, open(data_dir / "verify.json", "w"), indent=1)
    print("[done] all_pass =", res["all_pass"])


if __name__ == "__main__":
    main()
