"""Failure labels v2 (src/failure_labels.label_episode_v2) on every saved fpo_v2 eval episode. CPU only.

    CUDA_VISIBLE_DEVICES="" python scripts/failure_labels_run.py            (from ~/ffi26/improve)
Inputs: outputs/fpo_v2/eval_s{3000,4000,5000}/<arm>/task{0,8}/ep*.npz and outputs/fpo_v2/eval_s1000/<arm>/task{1..7,9}
(arms official, fpo_v1_it3, fpo_v2_sel; an arm "fpo_v3_sel" is added if its directories exist).
Outputs (outputs/failure_labels/):
  labels.csv               one row per episode: v2 primary, every lab_* column, details, old primary (label_episode)
  region_centres.csv       xy of fixture target regions, estimated from the logs (see region_centres())
  primary_by_arm_task.csv  failures per v2 primary label, arm x task
  multilabel_by_arm_task.csv  failures carrying each label (multi-label), arm x task
  combos.csv               most frequent label combinations among failures, per arm
  old_vs_new.csv           old primary x new primary (failures)
  task127_C.csv            tasks 1, 2, 7: wrong-object (C) episodes and objects, official vs v1 vs v2, paired McNemar
  manual_check.csv         20 failures for blind manual labelling (label columns left empty)
  render/*.yaml, render_commands.sh   video replay configs for those 20 (NOT run here: needs the GPU)
  summary.md               tables in Vietnamese
"""

import collections
import json
import sys
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from failure_labels import PRIORITY_V2, label_episode, label_episode_v2  # noqa: E402
from phase1_logging import load_episode  # noqa: E402

EVAL = ROOT / "outputs/fpo_v2"
OUT = ROOT / "outputs/failure_labels"
ARMS = ["official", "fpo_v1_it3", "fpo_v2_sel", "fpo_v3_sel"]
SETS = [(s, [0, 8]) for s in (3000, 4000, 5000)] + [(1000, [1, 2, 3, 4, 5, 6, 7, 9])]
LABS = PRIORITY_V2
MANUAL_N, MANUAL_SEED = 20, 20261101
CKPT = {  # same as outputs/fpo_v2/configs/eval_fpo_v2_s*.yaml (all arms expert fp32)
    "official": "lerobot/smolvla_libero",
    "fpo_v1_it3": "/home/student/ffi26/improve/outputs/fpo/iter2/train/pretrained_model",
    "fpo_v2_sel": "/home/student/ffi26/improve/outputs/fpo_v2/iter3/train/pretrained_model",
}


def mcnemar_exact(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2**n)


def load_all():
    eps, missing = [], []
    for arm in ARMS:
        for seed, tasks in SETS:
            for t in tasks:
                d = EVAL / f"eval_s{seed}" / arm / f"task{t}"
                files = sorted(d.glob("ep*.npz"))
                if not files:
                    missing.append(f"{arm} s{seed} task{t}")
                    continue
                for npz in files:
                    log, act, meta = load_episode(npz)
                    need = {"preds", "eef_pos", "obj_pos", "grasp", "contact"} - set(log)
                    if need:
                        missing.append(f"{npz}: thiếu {sorted(need)}")
                        continue
                    eps.append({"arm": arm, "seed": seed, "task": t, "init": int(npz.stem[2:]),
                                "success": bool(meta["success"]), "length": len(act), "log": log, "act": act,
                                "npz": str(npz)})
    return eps, missing


def region_centres(eps):
    """Fixture target regions (target not in obj_names): xy of the object at the last row where its goal predicate
    is True, averaged per init state, then over init states. One table per (task, region, object)."""
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    for e in eps:
        log = e["log"]
        names = list(log["obj_names"])
        P = np.asarray(log["preds"], bool)
        for p, pred in enumerate(log["meta"]["goal_state"]):
            if len(pred) < 3 or pred[1] not in names:
                continue
            if pred[2] in names or pred[2].replace("_contain_region", "") in names:
                continue
            rows = np.where(P[:, p])[0]
            if len(rows):
                acc[(e["task"], pred[2], pred[1])][e["init"]].append(log["obj_pos"][rows[-1], names.index(pred[1]), :2])
    rows = []
    for (t, reg, obj), d in sorted(acc.items()):
        per = np.array([np.mean(v, 0) for v in d.values()])
        rows.append({"task": t, "region": reg, "object": obj, "x": per[:, 0].mean(), "y": per[:, 1].mean(),
                     "std_across_inits_max": per.std(0).max(), "n_init": len(d),
                     "n_episodes": sum(len(v) for v in d.values())})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    eps, missing = load_all()
    reg = region_centres(eps)
    reg.to_csv(OUT / "region_centres.csv", index=False)
    rxy = {t: {(r.region, r.object): (r.x, r.y) for r in g.itertuples()} for t, g in reg.groupby("task")}

    rows = []
    for e in eps:
        new = label_episode_v2(e["log"], e["act"], e["success"], region_xy=rxy.get(e["task"]))
        old = label_episode(e["log"], e["act"], e["success"])
        rows.append({"arm": e["arm"], "seed": e["seed"], "task": e["task"], "init": e["init"], "success": e["success"],
                     "length": e["length"], **new, "old_primary": old["primary"],
                     **{f"old_lab_{k}": old[f"lab_{k}"] for k in "ABCDE"}, "old_C_objs": old["C_objs"],
                     "npz": e["npz"]})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "labels.csv", index=False)
    f = df[~df.success].copy()
    f["at"] = f.arm + " / task " + f.task.astype(str)

    prim = pd.crosstab([f.arm, f.task], f.primary).reindex(columns=LABS + ["O"], fill_value=0)
    n = df.groupby(["arm", "task"]).size().rename("n_ep")
    nf = f.groupby(["arm", "task"]).size().rename("n_fail")
    prim = pd.concat([n, nf, prim], axis=1).fillna(0).astype(int)
    prim.to_csv(OUT / "primary_by_arm_task.csv")
    multi = pd.concat([nf, f.groupby(["arm", "task"])[[f"lab_{k}" for k in LABS]].sum()], axis=1).astype(int)
    multi.columns = ["n_fail"] + LABS
    multi.to_csv(OUT / "multilabel_by_arm_task.csv")

    f["combo"] = f.apply(lambda r: "∧".join(k for k in LABS if r[f"lab_{k}"]) or "(không nhãn)", axis=1)
    combos = (f.groupby(["arm", "combo"]).size().rename("n").reset_index()
              .sort_values(["arm", "n"], ascending=[True, False]))
    combos["frac_of_arm_fail"] = combos.n / combos.arm.map(f.groupby("arm").size())
    combos.to_csv(OUT / "combos.csv", index=False)
    combos_t = f.groupby(["task", "combo"]).size().rename("n").reset_index().sort_values(["task", "n"], ascending=[True, False])
    old_new = pd.crosstab(f.old_primary, f.primary).reindex(columns=LABS + ["O"], fill_value=0)
    old_new.to_csv(OUT / "old_vs_new.csv")

    # tasks 1, 2, 7: wrong object, paired by init (seed 1000, same env seed per init)
    c_rows = []
    for t in (1, 2, 7):
        g = df[(df.task == t) & (df.seed == 1000)]
        ref = g[g.arm == "official"].set_index("init")
        for arm in [a for a in ARMS if a in set(g.arm)]:
            a = g[g.arm == arm].set_index("init")
            cc = a.lab_C & ~a.success
            rc = ref.lab_C & ~ref.success
            objs = collections.Counter(o for s in a[cc].C_objs for o in s.split(";") if o)
            c_rows.append({"task": t, "arm": arm, "n_ep": len(a), "n_fail": int((~a.success).sum()),
                           "n_fail_C_any": int((a.lab_C & ~a.success).sum()), "n_primary_C": int((a.primary == "C").sum()),
                           "C_objs": "; ".join(f"{o} ×{k}" for o, k in objs.most_common()),
                           "only_arm_C": int((cc & ~rc).sum()), "only_official_C": int((~cc & rc).sum()),
                           "mcnemar_p_vs_official": mcnemar_exact(int((cc & ~rc).sum()), int((~cc & rc).sum()))})
    c127 = pd.DataFrame(c_rows)
    c127.to_csv(OUT / "task127_C.csv", index=False)

    # manual check: 20 failures. Cycle over primary labels (random order); for each label take one random episode
    # from the arm picked least so far (ties: random order) among arms that have that label. Rows are then shuffled so the order does not reveal the label.
    rng = np.random.default_rng(MANUAL_SEED)
    pools = {lab: {arm: list(rng.permutation(g.index)) for arm, g in fl.groupby("arm")} for lab, fl in f.groupby("primary")}
    labs = list(pools)
    rng.shuffle(labs)
    arm_order = {lab: list(rng.permutation(list(pools[lab]))) for lab in labs}
    turn = collections.Counter()
    pick = []
    while len(pick) < MANUAL_N and any(any(v.values()) for v in pools.values()):
        for lab in labs:
            arms = [a for a in arm_order[lab] if pools[lab][a]]
            if not arms or len(pick) >= MANUAL_N:
                continue
            a = min(arms, key=lambda x: turn[x])
            turn[a] += 1
            pick.append(pools[lab][a].pop())
    pick = list(rng.permutation(pick))
    m = df.loc[pick, ["arm", "seed", "task", "init", "length", "npz"]].copy()
    m.insert(0, "id", range(1, len(m) + 1))
    m["video"] = [str(OUT / "videos" / f"{r.arm}_{'s1000' if r.seed == 1000 else 's3k5k'}"
                      / f"{r.arm}_t{r.task}_init{r.init}_s{r.seed}_fail.mp4") for r in m.itertuples()]
    m["video_exists"] = [Path(v).exists() for v in m.video]
    m["nhan_tay"] = ""
    m["nhan_script"] = ""
    m["ghi_chu"] = ""
    m.to_csv(OUT / "manual_check.csv", index=False)
    render_configs(m)

    write_summary(df, f, prim, multi, combos, combos_t, old_new, c127, reg, missing, m)
    print(f"{len(df)} episodes ({len(f)} failures) -> {OUT}")
    if missing:
        print("MISSING:", *missing, sep="\n  ")


def render_configs(m):
    """One config per (arm, seed group): n_episodes must equal the original eval (50 for tasks 0/8, 20 otherwise) so
    the batches, and hence the policy noise, are the same as in the original run."""
    rdir = OUT / "render"
    rdir.mkdir(exist_ok=True)
    for old in rdir.glob("*.yaml"):
        old.unlink()  # configs of an earlier selection
    cmds = ["#!/bin/bash", "# Chỉ chạy khi GPU rảnh. Mỗi lệnh replay các episode trong manual_check.csv và ghi video vào",
            f"# {OUT / 'videos'}/<config>/. Kiểm tra tất định: replay_check_<arm>.csv (so với file eval gốc).",
            "source /opt/miniconda3/etc/profile.d/conda.sh && conda activate lerobot && cd ~/ffi26/improve",
            "export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl"]
    for (arm, big), g in m.groupby([m.arm, m.seed != 1000]):
        nep = 50 if big else 20
        name = f"{arm}_{'s3k5k' if big else 's1000'}"
        cfg = ["# Replay for manual failure-label check (scripts/failure_labels_run.py). Settings = outputs/fpo_v2 evals.",
               "n_action_steps: 10", "batch_size: 8", f"n_episodes: {nep}", f"output_dir: {OUT / 'videos' / name}",
               "arms:", f"  - name: {arm}", f"    label: {arm}", f"    path: {CKPT[arm]}", "    expert_fp32: true",
               "originals:", f"  {arm}:"]
        cfg += [f"    {s}: {EVAL / f'eval_s{s}' / arm}/task{{task}}" for s in sorted(set(g.seed))]
        cfg += ["episodes:"] + [f"  - {{seed: {r.seed}, task: {r.task}, init: {r.init}}}" for r in g.itertuples()]
        p = rdir / f"{name}.yaml"
        p.write_text("\n".join(cfg) + "\n")
        cmds.append(f"python scripts/phase3_eval_video.py --config {p} --blind  # {len(g)} episode")
    (OUT / "render_commands.sh").write_text("\n".join(cmds) + "\n")


def md_table(d: pd.DataFrame) -> str:
    d = d.reset_index() if not isinstance(d.index, pd.RangeIndex) else d
    lines = ["| " + " | ".join(map(str, d.columns)) + " |", "|" + "---|" * len(d.columns)]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join(lines)


def write_summary(df, f, prim, multi, combos, combos_t, old_new, c127, reg, missing, m):
    s = ["# Nhãn thất bại v2 (`src/failure_labels.label_episode_v2`)", "",
         "Sinh bởi `scripts/failure_labels_run.py`. Dữ liệu: `outputs/fpo_v2/eval_s{3000,4000,5000}` (task 0, 8 × 50 init) và "
         "`outputs/fpo_v2/eval_s1000` (8 task khác × 20 init), expert fp32 cho mọi arm. Task 0, 8 gộp 3 seed (150 episode / arm / task).",
         f"Tổng {len(df)} episode, {len(f)} thất bại.", ""]
    miss_arm = sorted({x.split()[0] for x in missing if " s" in x and ":" not in x})
    miss_arm = [a for a in miss_arm if all(a not in e for e in set(df.arm))]
    other = [x for x in missing if x.split()[0] not in miss_arm]
    s += ["Thiếu dữ liệu: " + ("; ".join([f"{a}: chưa có thư mục eval nào" for a in miss_arm] + other) or "không"), ""]
    s += ["## Phân bố nhãn chính (chỉ episode thất bại)", "", md_table(prim), ""]
    s += ["## Tần suất từng nhãn (đa nhãn, chỉ episode thất bại)", "", md_table(multi), ""]
    top = combos.groupby("arm").head(6).reset_index(drop=True)
    top["frac_of_arm_fail"] = top.frac_of_arm_fail.map(lambda x: f"{100*x:.0f}%")
    s += ["## Tổ hợp nhãn hay gặp nhất (6 tổ hợp đầu mỗi arm)", "", md_table(top), ""]
    s += ["### Theo task (mọi arm gộp, 3 tổ hợp đầu)", "", md_table(combos_t.groupby("task").head(3).reset_index(drop=True)), ""]
    s += ["## Nhãn cũ (hàng) so với nhãn mới (cột), episode thất bại", "", md_table(old_new), ""]
    c = c127.copy()
    c["mcnemar_p_vs_official"] = c.mcnemar_p_vs_official.map(lambda x: f"{x:.2f}")
    s += ["## Task 1, 2, 7: nhầm vật (C), seed 1000, 20 init", "",
          "`n_fail_C_any`: số episode thất bại có nhãn C (đa nhãn). McNemar chính xác theo cặp init so với official.", "",
          md_table(c), ""]
    r = reg.copy()
    for k in ("x", "y", "std_across_inits_max"):
        r[k] = r[k].map(lambda v: f"{v:.3f}")
    s += ["## Tâm vùng đích dạng fixture (dùng cho G)", "",
          "Ước lượng từ log: vị trí xy của vật tại hàng cuối predicate True, trung bình theo init rồi theo các init. "
          "Vùng đích là vật di động (giỏ, đĩa) thì dùng vị trí vật đó trong log.", "", md_table(r), ""]
    s += ["## Kiểm chứng bằng tay", "", f"`manual_check.csv`: {len(m)} episode thất bại, chọn ngẫu nhiên (seed {MANUAL_SEED}) "
          "xoay vòng qua các nhóm (arm, nhãn chính). Cột `nhan_tay`, `nhan_script` để trống. "
          f"Video có sẵn: {int(m.video_exists.sum())}/{len(m)}; lệnh render: `render_commands.sh` (chưa chạy, cần GPU).", ""]
    (OUT / "summary.md").write_text("\n".join(s))


if __name__ == "__main__":
    main()
