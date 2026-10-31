"""LiberoEnv subclass that (1) logs predicates/physics every step via LiberoProbe,
(2) lets the caller choose the init state per episode, and (3) can start from a saved
full MuJoCo sim state (reset-to-failure). The installed lerobot package is not modified.

Control from the vector env (set before reset):
    vec.set_attr("next_init_state_id", [k0, k1, ...])   # index into the task's init states
    vec.set_attr("next_sim_state", [arr_or_None, ...])  # flattened sim state (overrides init state)
Read after the rollout:
    vec.call("get_episode_log")  -> list of dicts (one per sub-env)
"""

from __future__ import annotations

import numpy as np
from lerobot.envs.libero import LiberoEnv

from reward import LiberoProbe, get_domain


class LoggedLiberoEnv(LiberoEnv):
    def __init__(self, *args, log_sim_state: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_sim_state = log_sim_state
        self.next_init_state_id: int | None = None
        self.next_sim_state: np.ndarray | None = None
        self._probe: LiberoProbe | None = None
        self._log: list[dict] = []
        self._sim_states: list[np.ndarray] = []
        self._start_info: dict = {}

    # -- logging -------------------------------------------------------------
    def _record(self):
        rec = self._probe.read()
        self._log.append(rec)
        if self.log_sim_state:
            self._sim_states.append(np.asarray(self._env.get_sim_state(), dtype=np.float64).copy())

    def get_episode_log(self) -> dict:
        keys = self._log[0].keys() if self._log else []
        out = {k: np.stack([np.asarray(r[k]) for r in self._log]) for k in keys}
        if self.log_sim_state and self._sim_states:
            out["sim_state"] = np.stack(self._sim_states)
        out["meta"] = {**self._probe.meta(), **self._start_info, "task": self.task_description}
        out["goal_obj_mask"] = np.array([o in self._probe.goal_objs for o in self._probe.movable], dtype=bool)
        out["obj_names"] = list(self._probe.movable)
        return out

    # -- gym API -------------------------------------------------------------
    def reset(self, seed=None, **kwargs):
        if self.next_init_state_id is not None:
            self.init_state_id = int(self.next_init_state_id)
        used_id = self.init_state_id % len(self._init_states) if self._init_states is not None else None
        observation, info = super().reset(seed=seed, **kwargs)
        self._start_info = {"init_state_id": used_id, "seed": seed, "from_sim_state": False}
        if self.next_sim_state is not None:
            raw = self._env.regenerate_obs_from_state(np.asarray(self.next_sim_state, dtype=np.float64))
            observation = self._format_raw_obs(raw)
            self._start_info["from_sim_state"] = True
        if self._probe is None or self._probe.dom is not get_domain(self._env):
            self._probe = LiberoProbe(get_domain(self._env))
        self._log, self._sim_states = [], []
        self._record()
        return observation, info

    def step(self, action):
        out = super().step(action)
        self._record()
        return out
