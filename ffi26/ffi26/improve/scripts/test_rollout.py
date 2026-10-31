import os, sys, time
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, torch
from lerobot.utils.random_utils import set_seed
from rollout import load_policy, make_vec_env, run_batch
def main():
    set_seed(1000)
    t0 = time.time()
    pol = load_policy("lerobot/smolvla_libero", 10)
    vec = make_vec_env(8, 2)
    task = vec.call("task_description")[0]
    print("task:", task, "load", time.time() - t0)
    t0 = time.time()
    res = run_batch(vec, *pol, task_text=task, seeds=[1000, 1001], init_ids=[0, 1], max_steps=40, record_frames=True)
    print("rollout", time.time() - t0)
    for r in res:
        L = r["log"]
        print(r["init_id"], r["success"], r["length"], r["actions"].shape, {k: getattr(v, "shape", None) for k, v in L.items() if k != "meta"}, L["meta"])
        print(" phi", L["phi"][:3], L["phi"][-1], "eef", L["eef_pos"][-1], "grip", L["gripper_qpos"][-1], "frames", {k: v.shape for k, v in r["frames"].items()})
    # reset-to-state test: start env0 from env1's step-20 state
    st = res[1]["log"]["sim_state"][20]
    res2 = run_batch(vec, *pol, task_text=task, seeds=[1002, 1003], init_ids=[0, 0], sim_states=[st, None], max_steps=5)
    print("restore obj_pos diff:", np.abs(res2[0]["log"]["obj_pos"][0] - res[1]["log"]["obj_pos"][20]).max(), "eef diff", np.abs(res2[0]["log"]["eef_pos"][0]-res[1]["log"]["eef_pos"][20]).max())
    vec.close()

if __name__ == '__main__':
    main()
