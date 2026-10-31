"""Round-2 reward, failure events, segments and group-relative segment advantages.

Reward (per action step t, transition s_t -> s_{t+1}), discounted with the same gamma:
    r_t = success_bonus * 1[first success at t+1]
        + lam * (gamma * Phi(s_{t+1}) - Phi(s_t))          (potential-based shaping)
        - sum of failure-event costs charged at step t
Return-to-go G_t = sum_{k>=t} gamma^(k-t) r_k. With only the shaping term the discounted sum
telescopes to lam * (gamma^T Phi_T - Phi_t), so shaping does not change the optimal policy.

Failure events (each charged once per event; detectors read the LiberoProbe log + executed actions):
  drop      : a goal object that was held-and-lifted (reward.lifted_mask) stops being held and
              then falls > DROP_FALL m within DROP_WINDOW steps, and no goal predicate on that
              object is True at the end of the window.
  slip      : the gripper held a goal object (contact or grasp) for >= SLIP_HOLD steps while the
              close command was on, then lost it for >= SLIP_HOLD steps while the close command
              stayed on AND the fingers closed fully (width < SLIP_WIDTH, i.e. nothing between
              them), and no goal predicate on that object is True at that point.
  collision : a non-goal movable object was touched by the robot and later displaced
              > MOVE_THRESHOLD from where it was at the first touch. First time per object.
              (Not reward.moved_mask: that measures from the reset position, and in libero_10
              task 0 some objects start in the air and fall ~0.5 m in the first 10 steps.)
  timeout   : the episode ended without success (charged at the last step).
Known gap (detector check, task 8 Phase-1 ep003): a goal object knocked over by the open gripper
(pushed, never grasped) triggers none of these; it is only penalised via timeout / no Phi progress.

Segments: milestone k = first log row at which k distinct goal predicates that were False at the
episode start have been True at least once (monotone count). Segment k = action steps
[m_k, m_{k+1}); segment 0 starts at 0, the last one ends at T.

Segment advantage (no critic; the "value" of a stage is a group Monte-Carlo average):
    V(k)  = mean_j G_{m_k(j)}(j) over rollouts j of the same group that reached stage k
    A_seg = sum_{t in [s,e)} gamma^(t-s) r_t + gamma^(e-s) * V(k+1) - V(k)
where the bootstrap term is used only if the segment ends at a milestone (otherwise the episode
ended inside the segment and the sum already holds the outcome: success bonus or timeout cost).
So a segment is judged by what it achieved (reaching the next milestone, and how fast), not by
what later segments did. With < 2 rollouts at stage k there is nothing to compare against -> "none".
Label: positive if A > deadzone, negative if A < -deadzone, else none. All-success group -> positive.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from reward import MOVE_THRESHOLD, lifted_mask

DROP_FALL = 0.05  # m
DROP_WINDOW = 30  # steps
SLIP_HOLD = 5  # steps
SLIP_WIDTH = 0.015  # m, |finger qpos| sum; ~0.002-0.004 fully closed, ~0.02-0.06 on an object


@dataclass
class RewardR2Config:
    lam: float = 0.5
    gamma: float = 0.998
    success_bonus: float = 1.0
    costs: dict = field(default_factory=lambda: {"drop": 0.2, "slip": 0.1, "collision": 0.1, "timeout": 0.2})

    def to_dict(self) -> dict:
        return asdict(self)


def _held(log: dict) -> np.ndarray:
    return np.asarray(log["grasp"], bool) | np.asarray(log["contact"], bool)


def _obj_goal_preds(log: dict) -> dict[int, list[int]]:
    """object column -> indices of goal predicates whose first argument is that object."""
    names = list(log["obj_names"])
    out: dict[int, list[int]] = {}
    for p, pred in enumerate(log["meta"]["goal_state"]):
        if len(pred) > 1 and pred[1] in names:
            out.setdefault(names.index(pred[1]), []).append(p)
    return out


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Maximal runs [s, e) of True."""
    m = np.concatenate([[False], mask, [False]]).astype(int)
    d = np.diff(m)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def detect_events(log: dict, actions: np.ndarray, success: bool) -> list[dict]:
    """List of {type, step, obj}; `step` is the action index t whose transition the cost is charged to."""
    T = len(actions)
    if T == 0:
        return []
    names = list(log["obj_names"])
    goal_mask = np.asarray(log["goal_obj_mask"], bool)
    preds = np.asarray(log["preds"], bool)
    z = np.asarray(log["obj_pos"])[:, :, 2]
    held = _held(log)
    lifted = lifted_mask(log)
    close_cmd = np.concatenate([actions[:, -1] > 0, [actions[-1, -1] > 0]])  # row-aligned (row t <- action t)
    width = np.abs(np.asarray(log["gripper_qpos"])).sum(axis=1)
    gp = _obj_goal_preds(log)
    ev = []

    def goal_done(j, row):
        return any(preds[min(row, T), p] for p in gp.get(j, []))

    for j in np.where(goal_mask)[0]:
        # drop: end of a held-and-lifted run, then a fall within the window
        for s, e in _runs(lifted[:, j]):
            if e > T or held[min(e, T), j]:
                continue  # run ends because it was lowered while still held, or at the episode end
            w = min(e + DROP_WINDOW, T)
            if z[e - 1, j] - z[e:w + 1, j].min() > DROP_FALL and not goal_done(j, w):
                ev.append({"type": "drop", "step": int(e - 1), "obj": names[j]})
        # slip: held with gripper closed, then lost while still closed
        hold_closed = held[:, j] & close_cmd
        for s, e in _runs(hold_closed):
            if e - s < SLIP_HOLD or e > T - SLIP_HOLD:
                continue
            lost = ~held[e : e + SLIP_HOLD, j] & close_cmd[e : e + SLIP_HOLD]
            if lost.all() and width[e + SLIP_HOLD] < SLIP_WIDTH and not goal_done(j, e + SLIP_HOLD):
                ev.append({"type": "slip", "step": int(e - 1), "obj": names[j]})

    pos = np.asarray(log["obj_pos"])
    for j in np.where(~goal_mask)[0]:
        if not held[:, j].any():
            continue
        t0 = int(np.argmax(held[:, j]))
        moved = np.linalg.norm(pos[t0:, j] - pos[t0, j], axis=-1) > MOVE_THRESHOLD
        if moved.any():
            row = t0 + int(np.argmax(moved))
            ev.append({"type": "collision", "step": max(row - 1, 0), "obj": names[j]})

    if not success:
        ev.append({"type": "timeout", "step": T - 1, "obj": ""})
    return sorted(ev, key=lambda d: d["step"])


def step_rewards_r2(log: dict, actions: np.ndarray, success: bool, cfg: RewardR2Config,
                    events: list[dict] | None = None) -> np.ndarray:
    phi = np.asarray(log["phi"], np.float64)
    T = len(actions)
    phi = phi[: T + 1]
    succ = np.asarray(log["success"], bool)[: T + 1]
    r = cfg.lam * (cfg.gamma * phi[1:] - phi[:-1])
    if succ[1:].any():
        r[int(np.argmax(succ[1:]))] += cfg.success_bonus
    for e in events if events is not None else detect_events(log, actions, success):
        r[min(e["step"], T - 1)] -= cfg.costs.get(e["type"], 0.0)
    return r


def returns_to_go(r: np.ndarray, gamma: float) -> np.ndarray:
    """(T+1,) discounted return-to-go, G[T] = 0."""
    G = np.zeros(len(r) + 1)
    for t in range(len(r) - 1, -1, -1):
        G[t] = r[t] + gamma * G[t + 1]
    return G


def milestones(log: dict, T: int) -> list[int]:
    """Start step of each stage: [0, m_1, m_2, ...] (rows where the count of distinct new
    predicates ever True increases), capped to < T."""
    preds = np.asarray(log["preds"], bool)[: T + 1]
    new = ~preds[0]
    cnt = np.maximum.accumulate(preds[:, new], axis=0).sum(axis=1) if new.any() else np.zeros(T + 1, int)
    ms = [0]
    for k in range(1, int(cnt.max()) + 1):
        m = int(np.argmax(cnt >= k))
        if m < T:
            ms.append(m)
    return ms


def group_segment_labels(rollouts: list[dict], cfg: RewardR2Config, min_spread: float = 0.1,
                         deadzone: float = 0.02) -> dict:
    """rollouts: list of {log, actions, success}. Returns group info + per-rollout segments
    [{stage, start, end, adv, n_stage, label}] with label in {positive, negative, none}."""
    per = []
    for ro in rollouts:
        T = len(ro["actions"])
        ev = detect_events(ro["log"], ro["actions"], ro["success"])
        r = step_rewards_r2(ro["log"], ro["actions"], ro["success"], cfg, ev)
        G = returns_to_go(r, cfg.gamma)
        per.append({"T": T, "events": ev, "G": G, "ms": milestones(ro["log"], T)})
    G0 = np.array([p["G"][0] for p in per])
    spread = float(G0.max() - G0.min())
    all_success = all(ro["success"] for ro in rollouts)
    rejected = (spread < min_spread) and not all_success
    stage_vals: dict[int, list[float]] = {}
    for p in per:
        for k, m in enumerate(p["ms"]):
            stage_vals.setdefault(k, []).append(p["G"][m])
    V = {k: float(np.mean(v)) for k, v in stage_vals.items()}
    out = []
    for p, ro in zip(per, rollouts):
        segs = []
        bounds = p["ms"] + [p["T"]]
        for k in range(len(p["ms"])):
            s, e = bounds[k], bounds[k + 1]
            if e <= s:
                continue
            # discounted reward inside the segment: G_s - gamma^(e-s) G_e
            seg_ret = p["G"][s] - cfg.gamma ** (e - s) * p["G"][e]
            ends_at_milestone = k + 1 < len(p["ms"])
            boot = cfg.gamma ** (e - s) * V[k + 1] if ends_at_milestone else 0.0
            n_k = len(stage_vals[k])
            adv = seg_ret + boot - V[k]
            if all_success:
                label = "positive"
            elif n_k < 2:
                label = "none"
            else:
                label = "positive" if adv > deadzone else "negative" if adv < -deadzone else "none"
            segs.append({"stage": k, "start": int(s), "end": int(e), "adv": float(adv), "n_stage": n_k,
                         "label": label})
        out.append({"return": float(p["G"][0]), "events": p["events"], "segments": segs})
    return {"spread": spread, "all_success": all_success, "rejected": rejected,
            "return_mean": float(G0.mean()), "rollouts": out}
