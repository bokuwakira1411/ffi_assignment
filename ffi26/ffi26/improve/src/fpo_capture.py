"""FPO data capture: record every action-chunk decision a SmolVLA policy makes during run_batch.

policy._get_action_chunk is wrapped on the instance (installed lerobot untouched). At every call
(= every n_action_steps env steps, all sub-envs at once) the wrapper
  * draws the sampling noise itself from a seeded generator (seed = base_seed + call index), so the
    chunk can be replayed exactly (verification V1);
  * stores the model inputs exactly as the policy saw them (after the preprocessor: uint8 images,
    normalised state, language tokens/mask), the sampled chunk in the model's normalised action
    space (before the postprocessor) and the env step of the call.

    cap = ChunkCapture(policy, base_seed); res = run_batch(...); rows = cap.decisions(lengths); cap.detach()
"""

from __future__ import annotations

import numpy as np
import torch

IMG_KEYS = ("observation.images.camera1", "observation.images.camera2")
KEEP_KEYS = (*IMG_KEYS, "observation.state", "observation.language.tokens", "observation.language.attention_mask")


def to_uint8(img: torch.Tensor) -> np.ndarray:
    """float image in [0,1] -> uint8; asserts the image was uint8/255 to begin with (lossless)."""
    u = (img.float().clamp(0, 1) * 255).round()
    err = (u / 255 - img.float()).abs().max().item()
    assert err < 1e-4, f"image is not uint8/255 (max err {err})"
    return u.to(torch.uint8).cpu().numpy()


class ChunkCapture:
    def __init__(self, policy, base_seed: int):
        self.policy, self.base_seed = policy, int(base_seed)
        self.calls: list[dict] = []
        self.n_action_steps = policy.config.n_action_steps
        self._orig = policy._get_action_chunk
        policy._get_action_chunk = self._wrapped

    def noise(self, seed: int, n: int, device) -> torch.Tensor:
        cfg = self.policy.config
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        return torch.randn(n, cfg.chunk_size, cfg.max_action_dim, generator=g).to(device)

    def _wrapped(self, batch, *, noise=None, **kw):
        assert noise is None, "capture draws its own noise (noise_scale must be 1.0)"
        k = len(self.calls)
        seed = self.base_seed + k
        n = batch["observation.state"].shape[0]
        rec = {"call": k, "step": k * self.n_action_steps, "noise_seed": seed}
        for key in KEEP_KEYS:
            v = batch[key]
            rec[key] = to_uint8(v) if key in IMG_KEYS and v.dtype != torch.uint8 else v.cpu().numpy()
        noise = self.noise(seed, n, batch["observation.state"].device)
        chunk = self._orig(batch, noise=noise, **kw)
        rec["chunk"] = chunk.float().cpu().numpy()  # (n, chunk_size, action_dim), normalised space
        self.calls.append(rec)
        return chunk

    def detach(self):
        self.policy._get_action_chunk = self._orig

    def decisions(self, lengths: list[int]) -> dict[str, np.ndarray]:
        """Stack the calls into one row per (env j, call) with step < length_j.
        Extra columns: env (row in the batch), step, n_exec (= executed steps of this chunk)."""
        rows: dict[str, list] = {k: [] for k in (*KEEP_KEYS, "chunk", "env", "step", "n_exec", "noise_seed")}
        for rec in self.calls:
            for j, L in enumerate(lengths):
                if rec["step"] >= L:
                    continue
                for key in (*KEEP_KEYS, "chunk"):
                    rows[key].append(rec[key][j])
                rows["env"].append(j)
                rows["step"].append(rec["step"])
                rows["n_exec"].append(min(self.n_action_steps, L - rec["step"]))
                rows["noise_seed"].append(rec["noise_seed"])
        return {k: np.stack(v) if v else np.zeros(0) for k, v in rows.items()}
