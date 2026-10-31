"""Stage-progress reward for LIBERO (B.1).

Two parts:
  * `LiberoProbe`: runs inside the env process, reads the BDDL goal predicates and
    a few physical signals (gripper, EE, object poses, grasp/contact) after every step.
  * `step_rewards` / `episode_return`: compute the reward offline from a probe log, so
    the reward definition can be changed without re-running rollouts.

Phi(s) = (#goal predicates True) / (#goal predicates)
r_t    = 1[success_{t+1}] + lam * (Phi(s_{t+1}) - Phi(s_t)) - c   (+ optional terms below)

Optional terms (default 0 = off). None of them can be farmed:
  * grasp_bonus: paid once per goal object, the first time it is grasped AND lifted.
  * wrong_obj_cost: paid once per non-goal object, the first time it is grasped or moved.
  * stall_cost: paid once per stall segment that reaches `stall_k` steps.

LIBERO object access (verified on a real env, see outputs/probe_env.txt):
  LiberoEnv._env (OffScreenRenderEnv) .env (Libero_*_Manipulation domain) ->
  parsed_problem["goal_state"], _eval_predicate(pred), objects_dict, obj_body_id,
  _check_grasp(gripper, object_geoms), check_contact(gripper, obj).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

LIFT_THRESHOLD = 0.03  # m above the object's height at reset
MOVE_THRESHOLD = 0.05  # m displacement of a non-goal object counts as "moved"


def get_domain(env):
    """Walk env wrappers until the LIBERO domain object (has parsed_problem) is found."""
    seen = set()
    stack = [env]
    while stack:
        e = stack.pop()
        if e is None or id(e) in seen:
            continue
        seen.add(id(e))
        if hasattr(e, "parsed_problem") and hasattr(e, "_eval_predicate"):
            return e
        for attr in ("_env", "env", "unwrapped"):
            child = getattr(e, attr, None)
            if child is not None and child is not e:
                stack.append(child)
    raise AttributeError("LIBERO domain with parsed_problem not found")


class LiberoProbe:
    """Reads predicates and physical signals from the LIBERO domain each step."""

    def __init__(self, domain):
        self.dom = domain
        self.goal = [list(p) for p in domain.parsed_problem["goal_state"]]
        interest = set(domain.parsed_problem.get("obj_of_interest") or [])
        self.movable = list(domain.objects_dict.keys())
        self.goal_objs = [o for o in self.movable if o in interest]
        self.other_objs = [o for o in self.movable if o not in interest]

    def meta(self) -> dict:
        return {
            "goal_state": self.goal,
            "goal_objs": self.goal_objs,
            "other_objs": self.other_objs,
        }

    def read(self) -> dict:
        dom = self.dom
        preds = np.array([bool(dom._eval_predicate(p)) for p in self.goal], dtype=bool)
        robot = dom.robots[0]
        gripper = robot.gripper
        obj_pos = np.stack(
            [np.array(dom.sim.data.body_xpos[dom.obj_body_id[n]], dtype=np.float32) for n in self.movable]
        )
        grasp = np.array(
            [bool(dom._check_grasp(gripper=gripper, object_geoms=dom.objects_dict[n].contact_geoms)) for n in self.movable],
            dtype=bool,
        )
        contact = np.array([bool(dom.check_contact(gripper, dom.objects_dict[n])) for n in self.movable], dtype=bool)
        eef = np.array(dom.sim.data.site_xpos[robot.eef_site_id], dtype=np.float32)
        qpos = np.array(dom.sim.data.qpos[robot._ref_gripper_joint_pos_indexes], dtype=np.float32)
        return {
            "preds": preds,
            "phi": float(preds.mean()) if len(preds) else 0.0,
            "obj_pos": obj_pos,
            "grasp": grasp,
            "contact": contact,
            "eef_pos": eef,
            "gripper_qpos": qpos,
            "success": bool(dom._check_success()),
        }


# ----------------------------------------------------------------------------- offline reward


@dataclass
class RewardConfig:
    lam: float = 0.5
    c: float = 0.001
    grasp_bonus: float = 0.0
    wrong_obj_cost: float = 0.0
    stall_cost: float = 0.0
    stall_k: int = 50  # steps
    stall_eps: float = 0.002  # m per step: EE moves less than this -> "still"

    def to_dict(self) -> dict:
        return asdict(self)


def lifted_mask(log: dict) -> np.ndarray:
    """(T+1, n_obj) bool: held (grasp or gripper contact) and raised LIFT_THRESHOLD above the
    reset height. Contact is accepted because robosuite's _check_grasp misses some real grasps
    (e.g. moka pot handles, task 8 ep005 in outputs/baseline/logged)."""
    z = log["obj_pos"][:, :, 2]
    return (log["grasp"] | log["contact"]) & (z > z[0:1] + LIFT_THRESHOLD)


def moved_mask(log: dict) -> np.ndarray:
    """(T+1, n_obj) bool: the robot has touched the object (grasp/contact, latched) and it is
    displaced > MOVE_THRESHOLD from its reset position. Displacement alone is not enough: some
    LIBERO init states start objects in the air (libero_10 task 0: ketchup/milk/orange juice at
    z~0.9-1.05 m) and they fall without being touched."""
    d = np.linalg.norm(log["obj_pos"] - log["obj_pos"][0:1], axis=-1)
    touched = np.maximum.accumulate(log["grasp"] | log["contact"], axis=0)
    return touched & (d > MOVE_THRESHOLD)


def stall_segments(eef_pos: np.ndarray, eps: float, k: int) -> list[tuple[int, int]]:
    """Maximal runs [start, end) of steps where the EE moved < eps per step, length >= k."""
    v = np.linalg.norm(np.diff(eef_pos, axis=0), axis=-1) < eps
    segs, start = [], None
    for t, still in enumerate(list(v) + [False]):
        if still and start is None:
            start = t
        elif not still and start is not None:
            if t - start >= k:
                segs.append((start, t))
            start = None
    return segs


def _first_true(mask: np.ndarray) -> np.ndarray:
    """Per column, index of first True (or -1)."""
    any_ = mask.any(axis=0)
    idx = mask.argmax(axis=0)
    return np.where(any_, idx, -1)


def step_rewards(log: dict, cfg: RewardConfig) -> np.ndarray:
    """Per-step reward r_t for t = 0..T-1 (transition s_t -> s_{t+1}).

    `log` arrays have T+1 rows (row 0 = state after reset). Required keys:
    phi (T+1,), success (T+1,), and for the optional terms obj_pos, grasp, eef_pos,
    goal_obj_mask (n_obj,) bool.
    """
    phi = np.asarray(log["phi"], dtype=np.float64)
    succ = np.asarray(log["success"], dtype=bool)
    T = len(phi) - 1
    r = succ[1:].astype(np.float64) + cfg.lam * (phi[1:] - phi[:-1]) - cfg.c

    goal_mask = np.asarray(log.get("goal_obj_mask", []), dtype=bool)
    if cfg.grasp_bonus and goal_mask.any():
        first = _first_true(lifted_mask(log))
        for j in np.where(goal_mask)[0]:
            if first[j] > 0:
                r[first[j] - 1] += cfg.grasp_bonus
    if cfg.wrong_obj_cost and (~goal_mask).any():
        first = _first_true(log["grasp"] | moved_mask(log))
        for j in np.where(~goal_mask)[0]:
            if first[j] > 0:
                r[first[j] - 1] -= cfg.wrong_obj_cost
    if cfg.stall_cost:
        for s, _ in stall_segments(log["eef_pos"], cfg.stall_eps, cfg.stall_k):
            r[min(s + cfg.stall_k - 1, T - 1)] -= cfg.stall_cost
    return r


def episode_return(log: dict, cfg: RewardConfig) -> float:
    """Undiscounted return G = sum_t r_t. With only the base terms this equals
    1[success] + lam*(Phi_T - Phi_0) - c*T (the episode ends at the first success)."""
    return float(step_rewards(log, cfg).sum())
