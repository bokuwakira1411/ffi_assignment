"""DSRL step 0 (env part, no policy): can a saved eval episode be brought back to step t0?

    python scripts/dsrl_step0_env.py --npz <eval ep.npz> --task 8 --init 12 --seed 3000 --t0 280
R1  replay the executed actions of the episode open-loop from reset (same init state, env seed = seed + init),
    with sim-state logging on; compare EE / object positions / predicates with the original log, row by row.
R2  restart from the replayed sim state at row t0 (LoggedLiberoEnv next_sim_state) and replay actions t0.. again;
    compare with the original rows t0.. .
Prints max |diff| per signal and the first row where EE differs by > 1 mm. Writes nothing.
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"), str(Path(__file__).resolve().parent)]

import numpy as np  # noqa: E402
from lerobot.envs.utils import NEW_ROLLOUT_OPTION  # noqa: E402

from phase1_logging import load_episode  # noqa: E402
from rollout import make_vec_env  # noqa: E402


def play(vec, actions, seed, init, sim_state=None):
    vec.set_attr("next_init_state_id", [init])
    vec.set_attr("next_sim_state", [sim_state])
    vec.reset(seed=[seed], options={NEW_ROLLOUT_OPTION: True})
    for a in actions:
        _, _, term, trunc, _ = vec.step(a[None])
        if term[0] or trunc[0]:
            break
    return vec.call("get_episode_log")[0]


def compare(name, new, ref):
    n = min(len(new["eef_pos"]), len(ref["eef_pos"]))
    de = np.linalg.norm(new["eef_pos"][:n] - ref["eef_pos"][:n], axis=-1)
    do = np.linalg.norm(new["obj_pos"][:n] - ref["obj_pos"][:n], axis=-1).max(axis=-1)
    dp = (new["preds"][:n] != ref["preds"][:n]).any(axis=-1)
    bad = np.where(de > 1e-3)[0]
    print(f"[{name}] rows {n} (new {len(new['eef_pos'])}, ref {len(ref['eef_pos'])}): max|dEE| {de.max()*1000:.4f} mm, "
          f"max|dObj| {do.max()*1000:.4f} mm, pred mismatch rows {int(dp.sum())}, first EE>1mm row "
          f"{int(bad[0]) if len(bad) else -1}; success new {bool(new['success'][-1])} ref {bool(ref['success'][-1])}",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--task", type=int, required=True)
    ap.add_argument("--init", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True, help="eval seed; env seed = seed + init")
    ap.add_argument("--t0", type=int, required=True)
    a = ap.parse_args()
    ref, actions, meta = load_episode(Path(a.npz))
    assert meta["init_state_id"] == a.init, meta
    vec = make_vec_env(a.task, 1, log_sim_state=True)
    r1 = play(vec, actions, a.seed + a.init, a.init)
    compare("R1 replay from reset", r1, ref)
    st = r1["sim_state"][a.t0]
    r2 = play(vec, actions[a.t0:], a.seed + a.init, a.init, sim_state=st)
    ref2 = {k: ref[k][a.t0:] for k in ("eef_pos", "obj_pos", "preds", "success")}
    compare(f"R2 restart at row {a.t0}", r2, ref2)
    print("R2 start row vs original row t0: dEE "
          f"{np.linalg.norm(r2['eef_pos'][0] - ref['eef_pos'][a.t0])*1000:.4f} mm; meta {r2['meta'].get('from_sim_state')}")
    vec.close()


if __name__ == "__main__":
    main()
