"""Automatic failure labels for LIBERO episodes from LiberoProbe logs.

Labels (an episode can have several; `primary` picks one by priority):
  A grasp failure      : a goal object whose goal predicate never became True got >= MIN_CLOSE_ATTEMPTS
                         gripper-close commands while the EE was near it, and it was never lifted.
  B stuck at transition: >= 1 goal predicate that was False at reset became True, but not all.
  C wrong object       : a non-goal movable object was grasped, or touched by the gripper and
                         then displaced > MOVE_THRESHOLD (reward.moved_mask).
  D undoing progress   : a goal predicate that was False at reset went True -> False at some step.
  E stall              : the EE moved < stall_eps per step for >= stall_k consecutive steps.
  O other              : failure with none of the above (no progress, no detected mechanism).
In LIBERO every failed episode is a timeout (520 steps), so "timeout" alone carries no
information; E only marks the stall part.
Priority for `primary`: A > C > D > E > B > O (mechanism before the progress description).

Predicates that are already True at reset (e.g. 'turnon flat_stove_1' in libero_10 task 8)
are excluded from "subtask completion" and from B/D.
"""

from __future__ import annotations

import numpy as np

from reward import lifted_mask, moved_mask, stall_segments

NEAR_DIST = 0.10  # m, EE to object centre, counts as "near"
MIN_CLOSE_ATTEMPTS = 2
PRIORITY = ["A", "C", "D", "E", "B"]


def gripper_close_events(actions: np.ndarray) -> np.ndarray:
    """Step indices where the gripper command switches open (-) -> close (+)."""
    g = actions[:, -1] > 0
    return np.where(g[1:] & ~g[:-1])[0] + 1


def _pred_object(pred: list) -> str:
    return pred[1]


def subtask_stats(log: dict) -> dict:
    preds = np.asarray(log["preds"], dtype=bool)  # (T+1, P)
    init_true = preds[0]
    new = ~init_true
    ever = preds.any(axis=0)
    done_new = (ever & new).sum()
    final_new = (preds[-1] & new).sum()
    return {
        "n_pred": int(preds.shape[1]),
        "n_pred_new": int(new.sum()),
        "n_pred_new_ever_true": int(done_new),
        "n_pred_new_final_true": int(final_new),
        "subtask_completion": float(final_new / max(new.sum(), 1)),
        "phi_0": float(log["phi"][0]),
        "phi_T": float(log["phi"][-1]),
        "phi_max": float(np.max(log["phi"])),
    }


def label_episode(log: dict, actions: np.ndarray, success: bool, stall_eps=0.002, stall_k=50) -> dict:
    preds = np.asarray(log["preds"], dtype=bool)
    init_true = preds[0]
    new = ~init_true
    goal_state = log["meta"]["goal_state"]
    names = list(log["obj_names"])
    goal_mask = np.asarray(log["goal_obj_mask"], dtype=bool)
    obj_pos = np.asarray(log["obj_pos"])
    eef = np.asarray(log["eef_pos"])
    lifted = lifted_mask(log)
    T = len(actions)

    out = {k: False for k in PRIORITY}
    detail = {}

    # A: grasp failure on an unfinished goal object
    closes = gripper_close_events(actions) if T > 1 else np.array([], int)
    a_objs = []
    for p_idx, pred in enumerate(goal_state):
        obj = _pred_object(pred)
        if obj not in names or not new[p_idx] or preds[:, p_idx].any():
            continue
        j = names.index(obj)
        # log row t+1 is the state after action t; use the state when the close command was sent (row t)
        near = [t for t in closes if np.linalg.norm(eef[t] - obj_pos[t, j]) < NEAR_DIST]
        if len(near) >= MIN_CLOSE_ATTEMPTS and not lifted[:, j].any():
            a_objs.append(obj)
            detail.setdefault("A_close_attempts", {})[obj] = len(near)
    out["A"] = bool(a_objs)

    # C: wrong object touched (grasped) or moved
    moved = moved_mask(log)
    c_objs = [
        names[j]
        for j in np.where(~goal_mask)[0]
        if np.asarray(log["grasp"])[:, j].any() or moved[:, j].any()
    ]
    out["C"] = bool(c_objs)

    # D: progress undone (a new predicate True -> False)
    flips = (preds[:-1] & ~preds[1:]) & new
    d_preds = [goal_state[i] for i in np.where(flips.any(axis=0))[0]]
    out["D"] = bool(d_preds)

    # E: long EE stall
    segs = stall_segments(eef, stall_eps, stall_k)
    out["E"] = bool(segs)

    # B: partial progress
    st = subtask_stats(log)
    out["B"] = 0 < st["n_pred_new_ever_true"] < st["n_pred_new"]

    if success:
        primary = "S"
    else:
        primary = next((k for k in PRIORITY if out[k]), "O")
    return {
        "primary": primary,
        **{f"lab_{k}": bool(v) for k, v in out.items()},
        "A_objs": ";".join(a_objs),
        "C_objs": ";".join(c_objs),
        "D_preds": ";".join(" ".join(p) for p in d_preds),
        "E_longest_stall": int(max((e - s for s, e in segs), default=0)),
        "n_close_events": int(len(closes)),
        **st,
        "lifted_goal_objs": ";".join(names[j] for j in np.where(goal_mask & lifted.any(axis=0))[0]),
    }
