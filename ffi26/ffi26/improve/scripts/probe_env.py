"""Probe LIBERO env attribute names used by src/reward.py (prints goal predicates, objects, APIs)."""
import os
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np
from lerobot.envs.libero import LiberoEnv, _get_suite

suite = _get_suite("libero_10")
for tid in (0, 6, 8):
    env = LiberoEnv(suite, tid, "libero_10", obs_type="pixels_agent_pos")
    obs, _ = env.reset(seed=0)
    dom = env._env.env
    print(f"=== task {tid}: {env.task_description}")
    print("domain class:", type(dom).__name__, "| wrapper:", type(env._env).__name__)
    print("goal_state:", dom.parsed_problem["goal_state"])
    print("obj_of_interest:", dom.parsed_problem.get("obj_of_interest"))
    print("objects_dict:", list(dom.objects_dict.keys()))
    print("fixtures:", list(dom.fixtures_dict.keys()))
    print("preds now:", [bool(dom._eval_predicate(p)) for p in dom.parsed_problem["goal_state"]])
    print("sim state len:", env._env.get_sim_state().shape, "| init_states:", env._init_states.shape)
    print("gripper qpos:", obs["robot_state"]["gripper"]["qpos"], "eef:", obs["robot_state"]["eef"]["pos"])
    g = dom.robots[0].gripper
    print("gripper type:", type(g).__name__, "| has _check_grasp:", hasattr(dom, "_check_grasp"))
    for name, o in dom.objects_dict.items():
        pos = dom.sim.data.body_xpos[dom.obj_body_id[name]]
        print(f"  {name}: pos={np.round(pos,3)} grasp={dom._check_grasp(gripper=g, object_geoms=o.contact_geoms)} contact={dom.check_contact(g, o)}")
    env.close()
