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


# ----------------------------------------------------------------------------------------------- v2 (01/11/2026)
# label_episode_v2: checked against all 1380 saved fpo_v2 eval episodes (docs/WORKLOG.md, 01/11/2026).
# label_episode above is left unchanged (fpo_collect.py and the earlier reports use it).
# Changes, each backed by a check in WORKLOG:
#   A  "near" = EE within NEAR_DIST of the object centre OR gripper contact/grasp with it (black_book is grasped
#      at 0.105-0.119 m from its centre, mugs 8-10 % of grasp steps > 0.10 m); and A also fires when the gripper is
#      closed near the object for >= A_HOLD_STEPS steps (one close, then held: 19 cases missed by the old rule).
#   C  wrong object = grasped, or reward_r2 "collision" (displaced > MOVE_THRESHOLD from where it was at the
#      FIRST touch). The old rule used reward.moved_mask (displacement from the reset position): 113/239 flagged
#      objects had fallen before the first touch (task 0: ketchup / orange juice / milk start in the air).
#   D  debounced: a new predicate True for >= DEBOUNCE steps, then False for >= DEBOUNCE steps
#      (45/64 old D in failures came only from shorter flips; 57/691 successes have a <= 5-step flicker).
#   E  split by what the stall segment is doing: E_contact (>= half of its steps the gripper is closed or the EE
#      is "near" an object, as for A) and E_free (the rest). lab_E = E_contact or E_free (old meaning).
#   F_drop     reward_r2 "drop" event on a goal object (held and lifted, released, falls > DROP_FALL within
#              DROP_WINDOW, its goal predicate still False).
#   G_misplace a goal object that had been lifted is released, comes to rest (speed < SETTLE_SPEED for
#              SETTLE_STEPS, not held), its goal predicate is False and it lies < MISPLACE_DIST (xy) from where it
#              should be: the target object (basket, plate) if the target is a movable object, else `region_xy`
#              (fixture regions are not in the log; the caller passes an estimate), else G is not evaluated.
# Priority (primary): A > C > F > G > D > E_contact > E_free > B > O
#   mechanism on a goal object first (A: never got it), then the wrong object (C), then losing a goal object that
#   was held (F) or setting it down off target (G); D after F/G because a drop or misplacement of a finished
#   object also flips its predicate; stalls (E) are a symptom, contact before free; B / O only describe progress.

from reward_r2 import detect_events  # noqa: E402

A_HOLD_STEPS = 20
DEBOUNCE = 5
MISPLACE_DIST = 0.10  # m, xy
SETTLE_STEPS = 10
SETTLE_SPEED = 0.001  # m per step
PRIORITY_V2 = ["A", "C", "F", "G", "D", "E_contact", "E_free", "B"]


def _runs_bool(x: np.ndarray) -> list[tuple[int, int]]:
    m = np.concatenate([[0], x.astype(int), [0]])
    d = np.diff(m)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def debounced_flips(col: np.ndarray, k: int = DEBOUNCE) -> list[int]:
    """Rows where a True run of length >= k is followed by a False run of length >= k."""
    runs = _runs_bool(col)
    out = []
    for (s, e), nxt in zip(runs, runs[1:] + [(len(col), None)]):
        if e - s >= k and nxt[0] - e >= k:
            out.append(int(e))
    return out


def near_matrix(log: dict) -> np.ndarray:
    """(T+1, n_obj) bool: EE within NEAR_DIST of the object centre, or gripper contact / grasp with it."""
    eef = np.asarray(log["eef_pos"])
    pos = np.asarray(log["obj_pos"])
    dist = np.linalg.norm(eef[:, None, :] - pos, axis=-1)
    return (dist < NEAR_DIST) | np.asarray(log["contact"], bool) | np.asarray(log["grasp"], bool)


def _target_xy(log: dict, pred: list, row: int, region_xy: dict | None):
    names = list(log["obj_names"])
    if len(pred) < 3:
        return None
    tgt = pred[2]
    for cand in (tgt, tgt.replace("_contain_region", "")):
        if cand in names:
            return np.asarray(log["obj_pos"])[row, names.index(cand), :2]
    if region_xy and (tgt, pred[1]) in region_xy:
        return np.asarray(region_xy[(tgt, pred[1])])
    return None


def label_episode_v2(log: dict, actions: np.ndarray, success: bool, region_xy: dict | None = None,
                     stall_eps=0.002, stall_k=50) -> dict:
    """region_xy: {(target_region, object): (x, y)} for targets that are fixtures (not in obj_names)."""
    preds = np.asarray(log["preds"], dtype=bool)
    new = ~preds[0]
    goal_state = log["meta"]["goal_state"]
    names = list(log["obj_names"])
    goal_mask = np.asarray(log["goal_obj_mask"], dtype=bool)
    pos = np.asarray(log["obj_pos"])
    held = np.asarray(log["grasp"], bool) | np.asarray(log["contact"], bool)
    lifted = lifted_mask(log)
    T = len(actions)
    R = preds.shape[0]
    close_cmd = np.zeros(R, bool)
    if T:
        close_cmd[:T] = actions[:, -1] > 0
        close_cmd[T:] = actions[-1, -1] > 0
    near = near_matrix(log)
    events = detect_events(log, actions, success) if T else []

    out = {k: False for k in PRIORITY_V2}
    detail: dict = {}

    # A
    closes = gripper_close_events(actions) if T > 1 else np.array([], int)
    a_objs = []
    for p_idx, pred in enumerate(goal_state):
        obj = _pred_object(pred)
        if obj not in names or not new[p_idx] or preds[:, p_idx].any():
            continue
        j = names.index(obj)
        if lifted[:, j].any() or obj in a_objs:
            continue
        n_close = int(sum(near[t, j] for t in closes))
        n_hold = int((close_cmd[:T] & near[:T, j]).sum())
        if n_close >= MIN_CLOSE_ATTEMPTS or n_hold >= A_HOLD_STEPS:
            a_objs.append(obj)
            detail[obj] = (n_close, n_hold)
    out["A"] = bool(a_objs)

    # C
    coll = {e["obj"] for e in events if e["type"] == "collision"}
    grasp = np.asarray(log["grasp"], bool)
    c_objs = [names[j] for j in np.where(~goal_mask)[0] if grasp[:, j].any() or names[j] in coll]
    out["C"] = bool(c_objs)

    # F
    f_objs = sorted({e["obj"] for e in events if e["type"] == "drop"})
    out["F"] = bool(f_objs)

    # G
    g_objs, g_dist, g_eval = [], [], False
    speed = np.concatenate([np.zeros((1, len(names))), np.linalg.norm(np.diff(pos, axis=0), axis=-1)])
    for p_idx, pred in enumerate(goal_state):
        obj = _pred_object(pred)
        if obj not in names or len(pred) < 3 or _target_xy(log, pred, 0, region_xy) is None:
            continue
        g_eval = True
        j = names.index(obj)
        if not lifted[:, j].any():
            continue
        releases = [r for r in range(1, R) if held[r - 1, j] and not held[r, j] and lifted[:r, j].any()]
        for r in releases:
            for s in range(r, R - SETTLE_STEPS + 1):
                w = slice(s, s + SETTLE_STEPS)
                if held[w, j].any():
                    break  # picked up again before it came to rest
                if (speed[w, j] < SETTLE_SPEED).all():
                    row = s + SETTLE_STEPS - 1
                    xy = _target_xy(log, pred, row, region_xy)
                    d = float(np.linalg.norm(pos[row, j, :2] - xy))
                    if not preds[row, p_idx] and d < MISPLACE_DIST and obj not in g_objs:
                        g_objs.append(obj)
                        g_dist.append(d)
                    break
    out["G"] = bool(g_objs)

    # D
    d_preds = [goal_state[i] for i in np.where(new)[0] if debounced_flips(preds[:, i])]
    out["D"] = bool(d_preds)

    # E
    segs = stall_segments(np.asarray(log["eef_pos"]), stall_eps, stall_k)
    contact_step = close_cmd | near.any(axis=1)
    seg_kind = ["contact" if contact_step[s:e].mean() >= 0.5 else "free" for s, e in segs]
    out["E_contact"] = "contact" in seg_kind
    out["E_free"] = "free" in seg_kind

    st = subtask_stats(log)
    out["B"] = 0 < st["n_pred_new_ever_true"] < st["n_pred_new"]

    primary = "S" if success else next((k for k in PRIORITY_V2 if out[k]), "O")
    return {
        "primary": primary,
        **{f"lab_{k}": bool(v) for k, v in out.items()},
        "lab_E": bool(out["E_contact"] or out["E_free"]),
        "A_objs": ";".join(a_objs),
        "A_close_hold": ";".join(f"{o}:{c}/{h}" for o, (c, h) in detail.items()),
        "C_objs": ";".join(c_objs),
        "F_objs": ";".join(f_objs),
        "G_objs": ";".join(g_objs),
        "G_dist": ";".join(f"{d:.3f}" for d in g_dist),
        "G_evaluable": g_eval,
        "D_preds": ";".join(" ".join(p) for p in d_preds),
        "E_longest_stall": int(max((e - s for s, e in segs), default=0)),
        "E_segments": ";".join(f"{s}-{e}:{k}" for (s, e), k in zip(segs, seg_kind)),
        "n_close_events": int(len(closes)),
        **st,
        "lifted_goal_objs": ";".join(names[j] for j in np.where(goal_mask & lifted.any(axis=0))[0]),
    }
