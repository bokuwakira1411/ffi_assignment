"""DSRL step 0 (policy part): does SmolVLA accept external noise, and is chunk = f(obs, noise) deterministic?

    python scripts/dsrl_step0_policy.py --device cpu|cuda [--policy lerobot/smolvla_libero]
One real observation (libero_10 task 8, init 0, after reset) goes through the eval preprocessing (src/rollout.py).
Checks, expert in fp32 (as in the fpo_v2 evals):
  D1 same obs + same noise, two calls of policy._get_action_chunk -> max |diff| of the chunk (normalised space)
  D2 select_action(batch, noise=...) after reset -> first action == chunk[:, 0] of D1
  D3 different noise -> how much the chunk changes (sanity: noise matters)
  D4 noise changed only in the padded action dims (7..31) -> does the executed part (dims 0..6) change?
  D5 noise scaled by 0 (= the mean of the flow) vs N(0, I)
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"), str(Path(__file__).resolve().parent)]

import numpy as np  # noqa: E402
import torch  # noqa: E402
from lerobot.envs import preprocess_observation  # noqa: E402
from lerobot.envs.utils import NEW_ROLLOUT_OPTION  # noqa: E402
from lerobot.processor import bind_relative_anchor  # noqa: E402

from rollout import load_policy, make_vec_env  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--policy", default="lerobot/smolvla_libero")
    a = ap.parse_args()
    policy, pre, post, env_pre, env_post = load_policy(a.policy, 10, device=a.device, expert_fp32=True)
    cfg = policy.config
    print(f"chunk_size {cfg.chunk_size}, max_action_dim {cfg.max_action_dim}, num_steps {cfg.num_steps}, "
          f"action dim {cfg.action_feature.shape}", flush=True)
    vec = make_vec_env(8, 1, log_sim_state=False)
    vec.set_attr("next_init_state_id", [0])
    vec.set_attr("next_sim_state", [None])
    policy.reset(); pre.reset(); post.reset()
    ob, _ = vec.reset(seed=[3000], options={NEW_ROLLOUT_OPTION: True})
    bind_relative_anchor(policy, pre)
    obs = preprocess_observation(ob)
    obs["task"] = [vec.call("task_description")[0]]
    batch = pre(env_pre(obs))
    vec.close()
    g = torch.Generator().manual_seed(0)
    shape = (1, cfg.chunk_size, cfg.max_action_dim)
    n1 = torch.randn(shape, generator=g).to(a.device)
    n2 = torch.randn(shape, generator=g).to(a.device)

    def chunk(noise):
        with torch.inference_mode():
            return policy._get_action_chunk(dict(batch), noise=noise).float().cpu()

    c1, c1b = chunk(n1), chunk(n1)
    print(f"D1 same obs + same noise: max|diff| {float((c1 - c1b).abs().max()):.3e}; chunk shape {tuple(c1.shape)}")
    policy.reset()
    with torch.inference_mode():
        a0 = policy.select_action(dict(batch), noise=n1).float().cpu()
    print(f"D2 select_action(noise) first action vs chunk[:,0]: max|diff| {float((a0 - c1[:, 0]).abs().max()):.3e}; "
          f"queue left {len(policy._queues['action'])}")
    c2 = chunk(n2)
    print(f"D3 other noise: max|diff| {float((c1 - c2).abs().max()):.3f}, mean|diff| {float((c1 - c2).abs().mean()):.3f}; "
          f"mean|chunk| {float(c1.abs().mean()):.3f}; first 10 steps mean|diff| {float((c1 - c2)[:, :10].abs().mean()):.3f}")
    n3 = n1.clone()
    n3[..., 7:] = n2[..., 7:]
    c3 = chunk(n3)
    print(f"D4 noise changed only in padded dims 7..31: executed part max|diff| {float((c1 - c3).abs().max()):.3e}")
    c0 = chunk(torch.zeros_like(n1))
    print(f"D5 zero noise vs N(0,I): mean|diff| {float((c1 - c0).abs().mean()):.3f}")


if __name__ == "__main__":
    main()
