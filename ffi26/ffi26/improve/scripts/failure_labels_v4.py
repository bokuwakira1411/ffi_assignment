"""Failure labels v2 (src/failure_labels.label_episode_v2) on the FPO v4 eval episodes, compared with official / v2 / v3.
(= failure_labels_v3.py with a v4 arm; v3 labels are recomputed and checked against outputs/failure_labels_v3/labels.csv.)
CPU only. Does NOT overwrite outputs/failure_labels/ (committed run of failure_labels_run.py).

    CUDA_VISIBLE_DEVICES="" python scripts/failure_labels_v4.py            (from ~/ffi26/improve)
Inputs: official, fpo_v2_sel: outputs/fpo_v2/eval_s*/<arm>/task*; fpo_v3_sel: outputs/fpo_v3/eval_s*/fpo_v3_sel/task*;
fpo_v4_sel: outputs/fpo_v4/eval_s*/fpo_v4_sel/task*
(seeds 3000 / 4000 / 5000 tasks 0, 8; seed 1000 tasks 1..7, 9). Fixture region centres: the committed
outputs/failure_labels/region_centres.csv (estimated from official / v1 / v2), reused as-is so the official / v2 labels
are reproducible; check: official and v2 labels are recomputed and compared row by row with outputs/failure_labels/labels.csv
(script stops if any label column differs).
Outputs (outputs/failure_labels_v4/): labels.csv (v4 only), primary_by_arm_task.csv, multilabel_by_arm_task.csv,
task8_pots.csv, task0_Efree.csv, task4_pairs.csv, summary.md (Vietnamese).
Kettles on the stove = n_pred_new_ever_true (task 8: 2 predicates, moka_pot_1 / moka_pot_2 on the stove), as in the
WORKLOG table of 01/11/2026 08:05 and dsrl_train.py.
"""

import sys
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import pandas as pd  # noqa: E402

from failure_labels import PRIORITY_V2, label_episode_v2  # noqa: E402
from phase1_logging import load_episode  # noqa: E402

OUT = ROOT / "outputs/failure_labels_v4"
REF = ROOT / "outputs/failure_labels"
ARMS = {"official": ROOT / "outputs/fpo_v2", "fpo_v2_sel": ROOT / "outputs/fpo_v2", "fpo_v3_sel": ROOT / "outputs/fpo_v3",
        "fpo_v4_sel": ROOT / "outputs/fpo_v4"}
NAME = {"official": "official", "fpo_v2_sel": "v2 θ4", "fpo_v3_sel": "v3 θ4", "fpo_v4_sel": "v4"}
SETS = [(s, [0, 8]) for s in (3000, 4000, 5000)] + [(1000, [1, 2, 3, 4, 5, 6, 7, 9])]
LABS = PRIORITY_V2
LABCOLS = ["primary"] + [f"lab_{k}" for k in LABS]


def mcnemar_exact(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2**n)


def md(d: pd.DataFrame) -> str:
    cols = list(d.columns)
    return "\n".join(["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
                     + ["| " + " | ".join(str(v) for v in r) + " |" for r in d.itertuples(index=False)])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    reg = pd.read_csv(REF / "region_centres.csv")
    rxy = {t: {(r.region, r.object): (r.x, r.y) for r in g.itertuples()} for t, g in reg.groupby("task")}
    rows, missing = [], []
    for arm, root in ARMS.items():
        for seed, tasks in SETS:
            for t in tasks:
                files = sorted((root / f"eval_s{seed}" / arm / f"task{t}").glob("ep*.npz"))
                if not files:
                    missing.append(f"{arm} s{seed} task{t}")
                for npz in files:
                    log, act, meta = load_episode(npz)
                    lab = label_episode_v2(log, act, bool(meta["success"]), region_xy=rxy.get(t))
                    rows.append({"arm": arm, "seed": seed, "task": t, "init": int(npz.stem[2:]),
                                 "success": bool(meta["success"]), "length": len(act), **lab, "npz": str(npz)})
    if missing:
        sys.exit("MISSING: " + ", ".join(missing))
    df = pd.DataFrame(rows)

    # reproducibility check vs the committed labels (official, v2)
    ref = pd.read_csv(REF / "labels.csv")
    key = ["arm", "seed", "task", "init"]
    a = df[df.arm.isin(["official", "fpo_v2_sel"])].set_index(key).sort_index()
    b = ref[ref.arm.isin(["official", "fpo_v2_sel"])].set_index(key).sort_index()
    assert a.index.equals(b.index), "episode sets differ from outputs/failure_labels/labels.csv"
    diff = {c: int((a[c].astype(str) != b[c].astype(str)).sum()) for c in LABCOLS + ["success", "n_pred_new_ever_true"]}
    if any(diff.values()):
        sys.exit(f"labels differ from committed labels.csv: {diff}")
    print(f"check: official + v2 = {len(a)} episodes, every label column identical to {REF / 'labels.csv'}")
    r3 = pd.read_csv(ROOT / "outputs/failure_labels_v3/labels.csv").set_index(key).sort_index()
    a3 = df[df.arm == "fpo_v3_sel"].set_index(key).sort_index()
    assert a3.index.equals(r3.index), "v3 episode set differs from outputs/failure_labels_v3/labels.csv"
    d3 = {c: int((a3[c].astype(str) != r3[c].astype(str)).sum()) for c in LABCOLS + ["success"]}
    if any(d3.values()):
        sys.exit(f"v3 labels differ from outputs/failure_labels_v3/labels.csv: {d3}")
    print(f"check: v3 = {len(a3)} episodes identical to outputs/failure_labels_v3/labels.csv")
    df[df.arm == "fpo_v4_sel"].to_csv(OUT / "labels.csv", index=False)

    f = df[~df.success]
    n = df.groupby(["arm", "task"]).size().rename("n_ep")
    nf = f.groupby(["arm", "task"]).size().rename("n_fail")
    prim = pd.concat([n, nf, pd.crosstab([f.arm, f.task], f.primary).reindex(columns=LABS + ["O"], fill_value=0)],
                     axis=1).fillna(0).astype(int)
    prim.to_csv(OUT / "primary_by_arm_task.csv")
    multi = pd.concat([nf, f.groupby(["arm", "task"])[[f"lab_{k}" for k in LABS]].sum()], axis=1).fillna(0).astype(int)
    multi.columns = ["n_fail"] + LABS
    multi.to_csv(OUT / "multilabel_by_arm_task.csv")

    # task 8: kettles on the stove among failures; A, E_contact
    t8 = []
    for arm in ARMS:
        g = f[(f.arm == arm) & (f.task == 8)]
        pots = g.n_pred_new_ever_true
        t8.append({"arm": NAME[arm], "thành công": f"{int(df[(df.arm == arm) & (df.task == 8)].success.sum())}/150",
                   "thất bại": len(g), "0 ấm": int((pots == 0).sum()), "1 ấm": int((pots == 1).sum()),
                   "2 ấm (D)": int((pots == 2).sum()),
                   "lab_A": f"{int(g.lab_A.sum())} ({100 * g.lab_A.mean():.0f}%)",
                   "lab_E_contact": f"{int(g.lab_E_contact.sum())} ({100 * g.lab_E_contact.mean():.0f}%)",
                   "E_contact ∧ B": int((g.lab_E_contact & g.lab_B).sum()),
                   "nhãn chính A": int((g.primary == "A").sum()), "nhãn chính E_contact": int((g.primary == "E_contact").sum())})
    t8 = pd.DataFrame(t8)
    t8.to_csv(OUT / "task8_pots.csv", index=False)

    # task 0: E_free
    t0 = []
    for arm in ARMS:
        g = f[(f.arm == arm) & (f.task == 0)]
        e = g[g.lab_E_free]
        starts = [int(s.split("-")[0]) for segs in e.E_segments.dropna() for s in str(segs).split(";") if s.endswith(":free")]
        t0.append({"arm": NAME[arm], "thành công": f"{int(df[(df.arm == arm) & (df.task == 0)].success.sum())}/150",
                   "thất bại": len(g), "thất bại có E_free": len(e),
                   "nhãn chính E_free": int((g.primary == "E_free").sum()),
                   "E_free ở thành công": int(df[(df.arm == arm) & (df.task == 0) & df.success].lab_E_free.sum()),
                   "bước bắt đầu đoạn free (trung vị)": int(pd.Series(starts).median()) if starts else "—",
                   "lab_A": int(g.lab_A.sum()), "nhãn chính A": int((g.primary == "A").sum())})
    t0 = pd.DataFrame(t0)
    t0.to_csv(OUT / "task0_Efree.csv", index=False)

    # task 4: primary labels + paired episodes (seed 1000, same init)
    g4 = df[df.task == 4].set_index(["arm", "init"])
    pairs = []
    for i in sorted(df[df.task == 4].init.unique()):
        r = {"init": i}
        for arm in ARMS:
            x = g4.loc[(arm, i)]
            r[NAME[arm]] = "S" if x.success else x.primary
            r[f"{NAME[arm]} nhãn"] = "" if x.success else "∧".join(k for k in LABS if x[f"lab_{k}"])
        pairs.append(r)
    pairs = pd.DataFrame(pairs)
    pairs.to_csv(OUT / "task4_pairs.csv", index=False)

    L = ["# Failure labeler v2 trên eval FPO v4, so với official, v2 và v3", "",
         "Script `improve/scripts/failure_labels_v4.py` (CPU). Labeler `src/failure_labels.label_episode_v2`, tâm vùng "
         "fixture lấy từ `outputs/failure_labels/region_centres.csv` (đã commit). Kiểm tra: nhãn official và v2 tính lại "
         f"trùng 100 % với `outputs/failure_labels/labels.csv` ({len(a)} episode, mọi cột nhãn); nhãn v3 trùng với "
         f"`outputs/failure_labels_v3/labels.csv` ({len(a3)} episode). Đây là nhãn tự động, "
         "chưa xem video để xác nhận. Task 0, 8: 150 episode/arm (50 init × seed 3000/4000/5000); task khác: 20 episode/arm "
         "(seed 1000). \"k ấm\" = `n_pred_new_ever_true` (số ấm từng lên bếp), như bảng WORKLOG 01/11 08:05.", "",
         "## Task 8: thất bại theo số ấm đã lên bếp", "", md(t8), "",
         "## Task 0: E_free", "", md(t0), "",
         "## Nhãn chính của thất bại, theo arm × task", "",
         md(prim.reset_index().assign(arm=lambda d: d.arm.map(NAME))), "",
         "## Nhãn đa nhãn (số thất bại có nhãn), theo arm × task", "",
         md(multi.reset_index().assign(arm=lambda d: d.arm.map(NAME))), "",
         "## Task 4 (seed 1000), từng init: S = thành công, nếu không là nhãn chính (cột \"nhãn\": mọi nhãn)", "",
         md(pairs), ""]
    o, v3 = pairs["official"] == "S", pairs["v4"] == "S"
    L.append(f"Task 4: chỉ official thành công {int((o & ~v3).sum())}, chỉ v4 thành công {int((~o & v3).sum())} "
             f"(McNemar p = {mcnemar_exact(int((o & ~v3).sum()), int((~o & v3).sum())):.3f}).")
    (OUT / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
