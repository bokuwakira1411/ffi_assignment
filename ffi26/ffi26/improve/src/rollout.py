"""Batched LIBERO rollouts with SmolVLA, following lerobot_eval.rollout() (LeRobot 0.6.2)
step by step, plus: per-env init state / sim state, task-string suffix, extra noise scale,
predicate logs (LoggedLiberoEnv) and optional frame recording for dataset building.
"""

from __future__ import annotations

import os
import sys
from functools import partial
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
# forkserver workers import logged_env/reward from here
os.environ["PYTHONPATH"] = f"{SRC}:{os.environ.get('PYTHONPATH', '')}"

from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.envs import make_env_pre_post_processors, preprocess_observation  # noqa: E402
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig  # noqa: E402
from lerobot.envs.libero import _get_suite  # noqa: E402
from lerobot.envs.utils import NEW_ROLLOUT_OPTION, freeze_after_episode_end  # noqa: E402
from lerobot.policies import make_policy, make_pre_post_processors  # noqa: E402
from lerobot.processor import bind_relative_anchor  # noqa: E402

from logged_env import LoggedLiberoEnv  # noqa: E402

RENAME_MAP = {
    "observation.images.image": "observation.images.camera1",
    "observation.images.image2": "observation.images.camera2",
}
SUITE = "libero_10"
MAX_STEPS = 520  # TASK_SUITE_MAX_STEPS["libero_10"]


def load_policy(path: str, n_action_steps: int = 10, device: str = "cuda", adv_label: str | None = None,
                cfg_overrides: dict | None = None):
    """Same construction as lerobot_eval.eval_main (policy + processors + env processors).
    adv_label (none/positive/negative): attach the learned advantage-label embedding
    (src/adv_embed.py, loaded from the checkpoint) and condition every forward pass on it.
    cfg_overrides: policy-config fields set before construction (FPO training uses
    freeze_vision_encoder / train_expert_only / train_state_proj = True, as lerobot-train flags)."""
    pcfg = PreTrainedConfig.from_pretrained(path)
    pcfg.pretrained_path = Path(path) if Path(path).exists() else path
    pcfg.n_action_steps = n_action_steps
    pcfg.device = device
    for k, v in (cfg_overrides or {}).items():
        setattr(pcfg, k, v)
    ecfg = LiberoEnvConfig(task=SUITE, control_mode="relative")
    policy = make_policy(cfg=pcfg, env_cfg=ecfg, rename_map=RENAME_MAP)
    if adv_label is not None:
        from adv_embed import CODES, attach

        attach(policy, init_from=path if Path(path).exists() else None)
        policy.model._adv_label = CODES[adv_label]
    policy.eval()
    pre, post = make_pre_post_processors(
        policy_cfg=pcfg,
        pretrained_path=pcfg.pretrained_path,
        preprocessor_overrides={
            "device_processor": {"device": str(policy.config.device)},
            "rename_observations_processor": {"rename_map": RENAME_MAP},
        },
    )
    env_pre, env_post = make_env_pre_post_processors(env_cfg=ecfg, policy_cfg=pcfg)
    return policy, pre, post, env_pre, env_post


def make_vec_env(task_id: int, n_envs: int, log_sim_state: bool = True):
    suite = _get_suite(SUITE)
    fns = [
        partial(
            LoggedLiberoEnv,
            suite,
            task_id,
            SUITE,
            obs_type="pixels_agent_pos",
            episode_index=i,
            n_envs=n_envs,
            control_mode="relative",
            log_sim_state=log_sim_state,
        )
        for i in range(n_envs)
    ]
    return gym.vector.AsyncVectorEnv(
        [freeze_after_episode_end(fn) for fn in fns],
        context="forkserver",
        shared_memory=True,
        autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP,
    )


def _to_uint8_hwc(img: torch.Tensor) -> np.ndarray:
    """(B,3,H,W) float in [0,1] -> (B,H,W,3) uint8."""
    return (img.clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()


@torch.no_grad()
def run_batch(
    vec,
    policy,
    pre,
    post,
    env_pre,
    env_post,
    *,
    task_text: str,
    seeds: list[int],
    init_ids: list[int],
    sim_states: list | None = None,
    max_steps: int = MAX_STEPS,
    task_suffix: str = "",
    noise_scale: float = 1.0,
    record_frames: bool = False,
) -> list[dict]:
    """Run one batch (one episode per sub-env). Returns a list of per-env dicts with the
    probe log, success, length, executed actions, and (optionally) frames."""
    n = vec.num_envs
    assert len(seeds) == n and len(init_ids) == n
    vec.set_attr("next_init_state_id", list(init_ids))
    vec.set_attr("next_sim_state", list(sim_states) if sim_states is not None else [None] * n)

    policy.reset()
    pre.reset()
    post.reset()
    observation, _ = vec.reset(seed=list(seeds), options={NEW_ROLLOUT_OPTION: True})
    bind_relative_anchor(policy, pre)

    task = task_text + task_suffix
    done = np.zeros(n, dtype=bool)
    success = np.zeros(n, dtype=bool)
    length = np.zeros(n, dtype=int)
    actions = [[] for _ in range(n)]
    frames = [{"image": [], "image2": [], "state": []} for _ in range(n)]
    chunk_size = policy.config.chunk_size
    max_action_dim = policy.config.max_action_dim

    step = 0
    while not done.all() and step < max_steps:
        obs = preprocess_observation(observation)
        obs["task"] = [task] * n
        obs = env_pre(obs)
        if record_frames:
            im1 = _to_uint8_hwc(obs["observation.images.image"])
            im2 = _to_uint8_hwc(obs["observation.images.image2"])
            st = obs["observation.state"].cpu().numpy().astype(np.float32)
        batch = pre(obs)
        noise = None
        if noise_scale != 1.0:
            noise = noise_scale * torch.randn(n, chunk_size, max_action_dim, device=policy.config.device)
        with torch.inference_mode():
            action = policy.select_action(batch, noise=noise)
        action = post(action)
        action = env_post({"action": action})["action"]
        act_np = action.to("cpu").numpy()

        observation, _, terminated, truncated, info = vec.step(act_np)
        fin = info.get("final_info", {}) if isinstance(info, dict) else {}
        succ_now = np.asarray(fin.get("is_success", np.zeros(n, bool)) if "final_info" in info else info.get("is_success", np.zeros(n, bool)), dtype=bool)

        for i in range(n):
            if done[i]:
                continue
            actions[i].append(act_np[i].astype(np.float32))
            if record_frames:
                frames[i]["image"].append(im1[i])
                frames[i]["image2"].append(im2[i])
                frames[i]["state"].append(st[i])
            length[i] += 1
            if terminated[i] or truncated[i]:
                success[i] = bool(succ_now[i])
                done[i] = True
        step += 1

    logs = vec.call("get_episode_log")
    out = []
    for i in range(n):
        log = logs[i]
        # The probe's success flag is authoritative (same check_success() call as the env).
        ok = bool(np.asarray(log["success"])[-1]) or bool(success[i])
        rec = {
            "log": log,
            "success": ok,
            "length": int(length[i]),
            "actions": np.stack(actions[i]) if actions[i] else np.zeros((0, 7), np.float32),
            "seed": int(seeds[i]),
            "init_id": int(init_ids[i]),
            "task": task,
        }
        if record_frames:
            rec["frames"] = {k: np.stack(v) for k, v in frames[i].items()}
        out.append(rec)
    return out
