"""Learned advantage-label embedding fed directly into SmolVLA's action expert (fallback for the
text suffix, which did not work with the frozen VLM in round 1: positive - negative = -6 points).

A 3-row table (none / positive / negative) is added to every action-expert input token
(the output of VLAFlowMatching.embed_suffix). Row "none" is a constant zero vector, so label
"none" gives exactly the starting checkpoint; rows positive / negative are learned and start at
zero. The prompt text is not changed. The installed lerobot package is not modified: the module
is attached to an existing policy and embed_suffix is wrapped on that instance.

The table is saved by save_pretrained as `model.adv_label_emb.weight` (2 x expert_hidden) inside
model.safetensors. LeRobot loads policies with strict=False, so a vanilla load silently ignores
that key: always call attach(policy, init_from=<checkpoint dir>) after loading.

Label codes: 0 none, 1 positive, 2 negative. Set policy.model._adv_label to an int (whole batch)
or a LongTensor (B,) before the forward pass.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

KEY = "model.adv_label_emb.weight"
CODES = {"none": 0, "positive": 1, "negative": 2}


class AdvLabelEmb(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(2, hidden))

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        table = torch.cat([torch.zeros_like(self.weight[:1]), self.weight], dim=0)
        return table[labels]


def attach(policy, init_from: str | Path | None = None) -> nn.Module:
    """Add the label embedding to `policy` (SmolVLAPolicy) and wrap embed_suffix. Loads the
    table from <init_from>/model.safetensors if that file has the key. Returns the module."""
    m = policy.model
    if hasattr(m, "adv_label_emb"):
        return m.adv_label_emb
    dev = next(m.parameters()).device
    m.adv_label_emb = AdvLabelEmb(m.vlm_with_expert.expert_hidden_size).to(dev)
    m._adv_label = 0
    if init_from is not None and (Path(init_from) / "model.safetensors").exists():
        from safetensors import safe_open

        with safe_open(str(Path(init_from) / "model.safetensors"), framework="pt") as f:
            if KEY in f.keys():
                with torch.no_grad():
                    m.adv_label_emb.weight.copy_(f.get_tensor(KEY).to(dev))
                print(f"[adv_embed] loaded {KEY} from {init_from} (norms {m.adv_label_emb.weight.norm(dim=1).tolist()})")
    orig = m.embed_suffix

    def embed_suffix(noisy_actions, timestep):
        embs, pad_masks, att_masks = orig(noisy_actions, timestep)
        lab = m._adv_label
        if isinstance(lab, int):
            lab = torch.full((embs.shape[0],), lab, dtype=torch.long, device=embs.device)
        e = m.adv_label_emb(lab.to(embs.device))
        return embs + e[:, None, :].to(embs.dtype), pad_masks, att_masks

    m.embed_suffix = embed_suffix
    return m.adv_label_emb
