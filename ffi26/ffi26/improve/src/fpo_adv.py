"""Advantage weighting / normalisation for FPO training (used by scripts/fpo_train_v3.py, FPO v3 and v4).

adv_weighting:
  "decision" (v3 default): every decision has weight 1.
  "rollout"  (not used, rejected after the v4 sanity check): w = 1 / n_decisions(rollout of the decision).
  "segment"  (v4): w = 1 / n_decisions(segment of the decision); segment = the segments.csv row of the same
             (group, rollout j) whose [start, end) contains the decision's step. n counted over the decisions kept for
             training (fpo_train.load_decisions filter), so every segment has total weight 1 whatever its length.
adv_center:
  "mean" (v3 default): A^ = clip((A - mean_t) / std_t, -5, 5), mean_t / std_t unweighted over the decisions of task t
         (= fpo_train_v3.py before v4, line-for-line the same numbers).
  "none" (v4): A^ = clip(A / std_t, -5, 5), std_t = weighted std (weights w, around the weighted mean) of task t.
The per-step normalisation of w (mean over the batch = 1) is done in the trainer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def segment_keys(d: dict[str, np.ndarray], data_dir: Path) -> np.ndarray:
    """int key per decision: (group, rollout j, segment start) from <data_dir>/segments.csv; asserts exactly one
    segment contains each decision's step."""
    seg = pd.read_csv(Path(data_dir) / "segments.csv")
    seg["j"] = seg.name.str.extract(r"_j(\d+)$")[0].astype(int)
    by = {k: (g.start.to_numpy(), g.end.to_numpy()) for k, g in seg.groupby(["group", "j"])}
    out = np.empty(len(d["adv"]), np.int64)
    for i, (g, j, st) in enumerate(zip(d["group"].tolist(), d["env"].tolist(), d["step"].tolist())):
        s0, e0 = by[(g, j)]
        m = np.flatnonzero((s0 <= st) & (st < e0))
        assert len(m) == 1, (g, j, st, m)
        out[i] = (g * 1000 + j) * 10000 + int(s0[m[0]])
    return out


def rollout_weights(d: dict[str, np.ndarray], weighting: str, data_dir: Path | None = None) -> np.ndarray:
    if weighting == "decision":
        return np.ones(len(d["adv"]), np.float64)
    if weighting == "rollout":
        key = d["group"].astype(np.int64) * 1000 + d["env"].astype(np.int64)
    elif weighting == "segment":
        key = segment_keys(d, data_dir)
    else:
        raise ValueError(f"adv_weighting {weighting!r}")
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    return 1.0 / cnt[inv]


def normalise(d: dict[str, np.ndarray], w: np.ndarray, center: str) -> tuple[np.ndarray, dict]:
    a = d["adv"]
    if center == "mean":  # v3: exactly the previous expression (float32 mean / std of numpy)
        mu, sd = float(a.mean()), float(a.std())
        return np.clip((a - mu) / sd, -5, 5).astype(np.float32), {"mean": mu, "std": sd, "center": "mean"}
    if center != "none":
        raise ValueError(f"adv_center {center!r}")
    a64 = a.astype(np.float64)
    mw = float(np.sum(w * a64) / np.sum(w))
    sw = float(np.sqrt(np.sum(w * (a64 - mw) ** 2) / np.sum(w)))
    return np.clip(a64 / sw, -5, 5).astype(np.float32), {"weighted_mean": mw, "weighted_std": sw, "center": "none"}


def sign_stats(d: dict[str, np.ndarray], ahat: np.ndarray, w: np.ndarray, data_dir: Path) -> dict:
    """% decisions of successful rollouts with A^ < 0, % of failed rollouts with A^ > 0 (raw and normalised), and the
    rollout-weighted mean of A^ (each rollout's decisions averaged first), segment-weighted mean (each segment's
    decisions averaged first) and the w-weighted mean."""
    ro = pd.read_csv(Path(data_dir) / "rollouts.csv")
    ro["j"] = ro.name.str.extract(r"_j(\d+)$")[0].astype(int)
    succ = ro.set_index(["group", "j"]).success.astype(bool)
    s = succ.reindex(list(zip(d["group"].tolist(), d["env"].tolist()))).to_numpy()
    assert not pd.isna(s).any(), "decision without rollout in rollouts.csv"
    s = s.astype(bool)
    a = d["adv"]
    df = pd.DataFrame({"k": d["group"].astype(np.int64) * 1000 + d["env"], "ah": ahat,
                       "seg": segment_keys(d, data_dir)})
    return {"n": int(len(a)), "n_success_dec": int(s.sum()),
            "succ_raw_neg_pct": float(100 * (a[s] < 0).mean()), "succ_hat_neg_pct": float(100 * (ahat[s] < 0).mean()),
            "fail_raw_pos_pct": float(100 * (a[~s] > 0).mean()), "fail_hat_pos_pct": float(100 * (ahat[~s] > 0).mean()),
            "hat_mean_decision": float(ahat.mean()), "hat_mean_rollout": float(df.groupby("k").ah.mean().mean()),
            "hat_mean_segment": float(df.groupby("seg").ah.mean().mean()),
            "hat_mean_weighted": float(np.sum(w * ahat) / np.sum(w)),
            "w_sum_success_rollouts": float(w[s].sum()), "w_sum_fail_rollouts": float(w[~s].sum()),
            "n_success_rollouts": int(len(np.unique(df.k[s]))), "n_fail_rollouts": int(len(np.unique(df.k[~s])))}
