"""Task-0 diagnosis, part A (CPU only, no training): failure labels v2 (src/failure_labels.label_episode_v2) on
  A1/A2  the eval episodes of official / v2 θ4 / v3 θ4 -- READ from the committed labels (outputs/failure_labels/labels.csv
         for official and v2, outputs/failure_labels_v3/labels.csv for v3), nothing relabelled;
  A3     the COLLECTION rollouts of FPO v3, iterations 0..4 (outputs/fpo_v3/iter*/logs, tasks 0 and 8), labelled here.
Fixture region centres: outputs/failure_labels/region_centres.csv (as failure_labels_v3.py).
Stall start of a failure = first step of its first E segment of that kind ("free" / "contact") in E_segments.

    CUDA_VISIBLE_DEVICES="" nice -n 19 python scripts/task0_diag_labels.py          (from ~/ffi26/improve)
Outputs (outputs/task0_diag/): collect_labels.csv, A1_task0.csv, A2_task8.csv, A2_task4.csv, A2_forget.csv,
A3_collect_task0.csv, A3_collect_task8.csv.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import pandas as pd  # noqa: E402

from failure_labels import PRIORITY_V2, label_episode_v2  # noqa: E402
from phase1_logging import load_episode  # noqa: E402

OUT = ROOT / "outputs/task0_diag"
LABS = PRIORITY_V2
NAME = {"official": "official", "fpo_v2_sel": "v2 θ4", "fpo_v3_sel": "v3 θ4"}


def stall_starts(g: pd.DataFrame, kind: str) -> list[int]:
    out = []
    for segs in g.E_segments.dropna():
        st = [int(s.split("-")[0]) for s in str(segs).split(";") if s.endswith(":" + kind)]
        if st:
            out.append(min(st))
    return out


def med(x):
    return int(pd.Series(x).median()) if len(x) else "—"


def task_rows(df: pd.DataFrame, key: str, task: int) -> pd.DataFrame:
    rows = []
    for k, g in df[df.task == task].groupby(key, sort=False):
        f = g[~g.success]
        r = {key: k, "n": len(g), "thành công": int(g.success.sum()), "thất bại": len(f)}
        r.update({f"chính {l}": int((f.primary == l).sum()) for l in LABS + ["O"]})
        r.update({f"có {l}": int(f[f"lab_{l}"].sum()) for l in LABS})
        r["trung vị bắt đầu E_free"] = med(stall_starts(f[f.lab_E_free], "free"))
        r["trung vị bắt đầu E_contact"] = med(stall_starts(f[f.lab_E_contact], "contact"))
        r["độ dài trung vị thất bại"] = med(f.length.tolist())
        if task == 8:
            pots = f.n_pred_new_ever_true
            r.update({"0 ấm": int((pots == 0).sum()), "1 ấm": int((pots == 1).sum()), "2 ấm": int((pots == 2).sum())})
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ev = pd.concat([pd.read_csv(ROOT / "outputs/failure_labels/labels.csv").query("arm in ['official', 'fpo_v2_sel']"),
                    pd.read_csv(ROOT / "outputs/failure_labels_v3/labels.csv")])
    ev["arm"] = ev.arm.map(NAME)
    ev = ev[ev.arm.notna()]
    big = ev[ev.seed.isin([3000, 4000, 5000])]
    task_rows(big, "arm", 0).to_csv(OUT / "A1_task0.csv", index=False)
    task_rows(big, "arm", 8).to_csv(OUT / "A2_task8.csv", index=False)
    s1k = ev[ev.seed == 1000]
    task_rows(s1k, "arm", 4).to_csv(OUT / "A2_task4.csv", index=False)
    fg = []
    for arm, g in s1k.groupby("arm", sort=False):
        f = g[~g.success]
        fg.append({"arm": arm, "n": len(g), "thành công": int(g.success.sum()), "thất bại": len(f),
                   **{f"chính {l}": int((f.primary == l).sum()) for l in LABS + ["O"]}})
    pd.DataFrame(fg).to_csv(OUT / "A2_forget.csv", index=False)

    # A3: collection rollouts of v3, iterations 0..4
    reg = pd.read_csv(ROOT / "outputs/failure_labels/region_centres.csv")
    rxy = {t: {(r.region, r.object): (r.x, r.y) for r in g.itertuples()} for t, g in reg.groupby("task")}
    rows = []
    for it in range(5):
        d = ROOT / f"outputs/fpo_v3/iter{it}"
        ro = pd.read_csv(d / "rollouts.csv").set_index("name")
        for npz in sorted((d / "logs").glob("*.npz")):
            log, act, meta = load_episode(npz)
            t = int(ro.loc[npz.stem, "task_id"])
            assert bool(meta["success"]) == bool(ro.loc[npz.stem, "success"]), npz
            lab = label_episode_v2(log, act, bool(meta["success"]), region_xy=rxy.get(t))
            rows.append({"iter": it, "name": npz.stem, "task": t, "init": int(meta["init_id"]),
                         "success": bool(meta["success"]), "length": len(act), **lab})
    cl = pd.DataFrame(rows)
    cl.to_csv(OUT / "collect_labels.csv", index=False)
    for t in (0, 8):
        task_rows(cl, "iter", t).to_csv(OUT / f"A3_collect_task{t}.csv", index=False)
    for f in sorted(OUT.glob("A*.csv")):
        print(f"== {f.name}\n{pd.read_csv(f).to_string(index=False)}\n")


if __name__ == "__main__":
    main()
