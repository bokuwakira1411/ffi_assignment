"""Unit tests for failure_labels.label_episode_v2 on synthetic probe logs (CPU only, no env).

    python -m pytest tests/test_failure_labels_v2.py -q      (from ~/ffi26/improve)
Scene: two goal objects g0, g1 (goal: g0 in basket, g1 in basket), one non-goal object w, one basket b.
Log rows: T+1 (row t = state before action t); actions: T x 7, last column = gripper (+1 close, -1 open).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from failure_labels import label_episode_v2  # noqa: E402

T = 300
NAMES = ["g0", "g1", "w", "basket_1"]
GOAL = [["in", "g0", "basket_1_contain_region"], ["in", "g1", "basket_1_contain_region"]]
P0 = np.array([[0.0, 0.0, 0.5], [0.3, 0.0, 0.5], [0.0, 0.3, 0.5], [0.6, 0.0, 0.45]])


def base_log():
    """EE moving steadily far from every object (no stall, no contact), gripper open, nothing happens."""
    eef = np.zeros((T + 1, 3))
    eef[:, 0] = -1.0
    eef[:, 1] = -1.0 + 0.005 * (np.arange(T + 1) % 100)  # 5 mm / step, stays far from the objects (y in [-1, -0.5])
    eef[:, 2] = 1.0
    log = {
        "preds": np.zeros((T + 1, 2), bool),
        "phi": np.zeros(T + 1),
        "obj_pos": np.repeat(P0[None], T + 1, axis=0).astype(float),
        "grasp": np.zeros((T + 1, 4), bool),
        "contact": np.zeros((T + 1, 4), bool),
        "eef_pos": eef,
        "gripper_qpos": np.full((T + 1, 2), 0.04),
        "success": np.zeros(T + 1, bool),
        "goal_obj_mask": np.array([1, 1, 0, 1], bool),
        "obj_names": NAMES,
        "meta": {"goal_state": GOAL},
    }
    act = np.zeros((T, 7))
    act[:, -1] = -1
    return log, act


def lab(log, act, success=False):
    return label_episode_v2(log, act, success)


def test_base_is_O():
    out = lab(*base_log())
    assert out["primary"] == "O"
    assert not any(out[f"lab_{k}"] for k in ["A", "C", "F", "G", "D", "E_contact", "E_free", "B"])


def test_A_single_close_held():
    """One close near g0, held 40 steps, g0 never lifted -> A (old rule needed >= 2 closes)."""
    log, act = base_log()
    log["eef_pos"][100:141] = P0[0] + [0, 0, 0.02]
    act[100:140, -1] = 1
    out = lab(log, act)
    assert out["lab_A"] and out["A_objs"] == "g0" and out["primary"] == "A"


def test_A_contact_counts_as_near():
    """EE 0.12 m from the centre (outside NEAR_DIST) but in contact, two closes -> A."""
    log, act = base_log()
    log["eef_pos"][100:160] = P0[0] + [0.12, 0, 0]
    log["contact"][100:160, 0] = True
    act[100:105, -1] = 1
    act[120:125, -1] = 1
    assert lab(log, act)["lab_A"]


def test_A_not_when_lifted():
    log, act = base_log()
    log["eef_pos"][100:141] = P0[0] + [0, 0, 0.02]
    act[100:140, -1] = 1
    log["grasp"][110:140, 0] = True
    log["obj_pos"][115:140, 0, 2] = 0.6
    assert not lab(log, act)["lab_A"]


def test_C_fallen_before_touch_is_not_C():
    """w falls 0.4 m at the start (no touch), touched later without moving -> not C (old rule: C)."""
    log, act = base_log()
    log["obj_pos"][0, 2, 2] = 0.9
    log["contact"][200:205, 2] = True
    assert not lab(log, act)["lab_C"]


def test_C_pushed_after_touch():
    log, act = base_log()
    log["contact"][150:160, 2] = True
    log["obj_pos"][155:, 2, 0] += 0.10
    out = lab(log, act)
    assert out["lab_C"] and out["C_objs"] == "w"


def test_C_grasp_wrong_object():
    log, act = base_log()
    log["grasp"][150:152, 2] = True
    assert lab(log, act)["lab_C"]


def test_C_basket_is_goal():
    """Pushing the basket (a goal object: the target) is not C."""
    log, act = base_log()
    log["contact"][150:160, 3] = True
    log["obj_pos"][155:, 3, 0] += 0.10
    assert not lab(log, act)["lab_C"]


def _lift_and_release(log, j, t_grab=100, t_rel=150, z_up=0.15):
    log["grasp"][t_grab:t_rel, j] = True
    log["obj_pos"][t_grab + 5:t_rel, j, 2] = P0[j, 2] + z_up


def test_F_drop():
    """g0 lifted 15 cm, released at 150, falls back to the table within 5 steps, predicate False -> F."""
    log, act = base_log()
    _lift_and_release(log, 0)
    out = lab(log, act)
    assert out["lab_F"] and out["F_objs"] == "g0" and out["primary"] == "F"


def test_G_misplace_near_basket():
    """g0 carried next to the basket (8 cm in xy), lowered while held, released, rests; predicate False -> G."""
    log, act = base_log()
    j = 0
    log["grasp"][100:150, j] = True
    log["obj_pos"][105:130, j, 2] = 0.65
    log["obj_pos"][130:, j, :2] = P0[3, :2] + [0.08, 0]
    log["obj_pos"][130:150, j, 2] = 0.60
    log["obj_pos"][140:, j, 2] = 0.5  # lowered while still held, then released at 150
    out = lab(log, act)
    assert out["lab_G"] and not out["lab_F"] and out["G_objs"] == "g0" and out["primary"] == "G"


def test_G_not_when_far():
    log, act = base_log()
    log["grasp"][100:150, 0] = True
    log["obj_pos"][105:130, 0, 2] = 0.65
    log["obj_pos"][130:, 0, :2] = P0[3, :2] + [0.30, 0]
    log["obj_pos"][140:, 0, 2] = 0.5
    assert not lab(log, act)["lab_G"]


def test_D_debounce():
    log, act = base_log()
    log["preds"][100:103, 0] = True  # 3-step flicker: not D
    log["preds"][103:110, 0] = False
    log["preds"][110:113, 0] = True
    out = lab(log, act)
    assert not out["lab_D"] and out["lab_B"]
    log["preds"][150:170, 1] = True  # True 20 steps, then False to the end: D
    assert lab(log, act)["lab_D"]


def test_E_contact_vs_free():
    log, act = base_log()
    log["eef_pos"][200:] = log["eef_pos"][200]  # stall 100 steps far from everything, gripper open
    out = lab(log, act)
    assert out["lab_E_free"] and not out["lab_E_contact"] and out["primary"] == "E_free"
    log, act = base_log()
    log["eef_pos"][200:] = P0[1] + [0.05, 0, 0.05]  # stall next to g1
    out = lab(log, act)
    assert out["lab_E_contact"] and not out["lab_E_free"] and out["primary"] == "E_contact"


def test_priority_and_success():
    log, act = base_log()
    _lift_and_release(log, 0)  # F
    log["grasp"][200:202, 2] = True  # C
    out = lab(log, act)
    assert out["lab_F"] and out["lab_C"] and out["primary"] == "C"
    assert lab(log, act, success=True)["primary"] == "S"
