"""Rollouts where the caller chooses the flow-matching noise of every action chunk (DSRL, noise probes).

Same loop as rollout.run_batch (lerobot_eval.rollout step by step), plus:
  * noise_fn(k, obs_batch, active) -> (n, chunk_size, max_action_dim) tensor, called at every replan (the policy's
    action queue is empty: steps 0, n_action_steps, 2 n_action_steps, ...; all sub-envs replan together);
    `k` = replan index, `obs_batch` = the preprocessed batch the policy sees, `active` = bool mask of running envs.
    SmolVLA uses the given noise (LeRobot 0.6.2 select_action(batch, noise=...); checked in scripts/dsrl_step0_policy.py).
  * per-env start sim states (LoggedLiberoEnv next_sim_state) and a step budget;
  * every replan is recorded per env: step, 64x64 agent-view image (uint8), normalised state.
src/rollout.py is not changed (the FPO v3 loop uses it).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from lerobot.envs import preprocess_observation
from lerobot.envs.utils import NEW_ROLLOUT_OPTION
from lerobot.processor import bind_relative_anchor

from rollout import MAX_STEPS, _to_uint8_hwc

IMG_KEY = "observation.images.camera1"  # agent view after the rename map
IMG_RES = 64


def small_image(batch: dict) -> np.ndarray:
    """(n, 3, H, W) float [0, 1] -> (n, 3, 64, 64) uint8 (area interpolation)."""
    x = batch[IMG_KEY].float()
    x = F.interpolate(x, size=(IMG_RES, IMG_RES), mode="area")
    return (x.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()


@torch.no_grad()
def run_batch_noise(vec, policy, pre, post, env_pre, env_post, *, task_text: str, seeds: list[int],
                    init_ids: list[int], noise_fn, sim_states: list | None = None, max_steps: int = MAX_STEPS,
                    record_frames: bool = False) -> list[dict]:
    n = vec.num_envs
    vec.set_attr("next_init_state_id", list(init_ids))
    vec.set_attr("next_sim_state", list(sim_states) if sim_states is not None else [None] * n)
    policy.reset()
    pre.reset()
    post.reset()
    observation, _ = vec.reset(seed=list(seeds), options={NEW_ROLLOUT_OPTION: True})
    bind_relative_anchor(policy, pre)
    done = np.zeros(n, bool)
    success = np.zeros(n, bool)
    length = np.zeros(n, int)
    actions = [[] for _ in range(n)]
    dec = [{"step": [], "img": [], "state": []} for _ in range(n)]
    frames = [{"image": [], "image2": []} for _ in range(n)]
    k = 0
    step = 0
    while not done.all() and step < max_steps:
        obs = preprocess_observation(observation)
        obs["task"] = [task_text] * n
        obs = env_pre(obs)
        if record_frames:
            im1 = _to_uint8_hwc(obs["observation.images.image"])
            im2 = _to_uint8_hwc(obs["observation.images.image2"])
        batch = pre(obs)
        noise = None
        if policy._check_get_actions_condition():
            noise = noise_fn(k, batch, ~done)
            img = small_image(batch)
            st = batch["observation.state"].float().cpu().numpy()
            for i in np.where(~done)[0]:
                dec[i]["step"].append(step)
                dec[i]["img"].append(img[i])
                dec[i]["state"].append(st[i])
            k += 1
        with torch.inference_mode():
            action = policy.select_action(batch, noise=noise)
        action = env_post({"action": post(action)})["action"]
        act_np = action.to("cpu").numpy()
        observation, _, terminated, truncated, info = vec.step(act_np)
        succ_now = np.asarray(info.get("final_info", {}).get("is_success", np.zeros(n, bool)) if "final_info" in info
                              else info.get("is_success", np.zeros(n, bool)), dtype=bool)
        for i in range(n):
            if done[i]:
                continue
            actions[i].append(act_np[i].astype(np.float32))
            if record_frames:
                frames[i]["image"].append(im1[i])
                frames[i]["image2"].append(im2[i])
            length[i] += 1
            if terminated[i] or truncated[i]:
                success[i] = bool(succ_now[i])
                done[i] = True
        step += 1
    logs = vec.call("get_episode_log")
    out = []
    for i in range(n):
        log = logs[i]
        rec = {"log": log, "success": bool(np.asarray(log["success"])[-1]) or bool(success[i]),
               "length": int(length[i]), "seed": int(seeds[i]), "init_id": int(init_ids[i]), "task": task_text,
               "actions": np.stack(actions[i]) if actions[i] else np.zeros((0, 7), np.float32),
               "dec_step": np.asarray(dec[i]["step"], np.int32),
               "dec_img": np.stack(dec[i]["img"]) if dec[i]["img"] else np.zeros((0, 3, IMG_RES, IMG_RES), np.uint8),
               "dec_state": np.stack(dec[i]["state"]) if dec[i]["state"] else np.zeros((0, 8), np.float32)}
        if record_frames:
            rec["frames"] = {kk: np.stack(v) for kk, v in frames[i].items()}
        out.append(rec)
    return out


def seeded_noise(seed: int, shape, scale: float = 1.0) -> torch.Tensor:
    """Noise from its own CPU generator, so a rollout's noise does not depend on the batch it ran in."""
    g = torch.Generator().manual_seed(int(seed))
    return scale * torch.randn(shape, generator=g)
