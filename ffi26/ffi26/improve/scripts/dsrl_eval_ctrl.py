"""DSRL control: evaluate the untrained actor (actor_iter0.pt, mean = 0 -> noise N(0, I) = official SmolVLA) with the
SAME seeded noise as the run1 eval (common random numbers), then pair it with the trained actor episode by episode.

    python scripts/dsrl_eval_ctrl.py --run outputs/dsrl/run1 --out outputs/dsrl/run1_ctrl0
Same env seeds, init states, batch order and noise seeds as dsrl_train.evaluate; the only difference between the two
arms is the actor mean. Writes <out>/eval_s*/..., <out>/eval_labels.csv, <out>/paired.csv, <out>/summary.md.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from dsrl_train import EVAL_SEEDS, TASK, Actor, evaluate, mcnemar_exact  # noqa: E402
from failure_labels import label_episode_v2  # noqa: E402
from phase1_logging import load_episode  # noqa: E402
from rollout import load_policy  # noqa: E402


def labels(out):
    reg = pd.read_csv(ROOT / "outputs/failure_labels/region_centres.csv")
    rxy = {(r.region, r.object): (r.x, r.y) for r in reg[reg.task == TASK].itertuples()}
    rows = []
    for seed in EVAL_SEEDS:
        for npz in sorted((out / f"eval_s{seed}" / "dsrl" / f"task{TASK}").glob("ep*.npz")):
            log, act, meta = load_episode(npz)
            lab = label_episode_v2(log, act, bool(meta["success"]), region_xy=rxy)
            rows.append({"seed": seed, "init": int(npz.stem[2:]), "success": bool(meta["success"]), **lab})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(ROOT / "outputs/dsrl/run1"))
    ap.add_argument("--out", default=str(ROOT / "outputs/dsrl/run1_ctrl0"))
    ap.add_argument("--policy", default="lerobot/smolvla_libero")
    a = ap.parse_args()
    run, out = Path(a.run), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pol = load_policy(a.policy, 10, expert_fp32=True)
    actor = Actor(pol[0].config.chunk_size).to(pol[0].config.device)
    actor.load_state_dict(torch.load(run / "actor_iter0.pt"))
    assert all(float(p.abs().sum()) == 0 for p in actor.head[-1].parameters()), "actor_iter0 must have mean 0"
    evaluate(actor, pol, out)
    c = labels(out)
    c.to_csv(out / "eval_labels.csv", index=False)
    t = pd.read_csv(run / "eval_labels.csv", keep_default_na=False)
    ref = pd.read_csv(ROOT / "outputs/failure_labels/labels.csv", keep_default_na=False)
    ref = ref[(ref.task == TASK) & ref.seed.isin(EVAL_SEEDS) & (ref.arm == "official")]
    idx = ["seed", "init"]
    p = (t.set_index(idx)[["success", "primary"]].add_prefix("dsrl_")
         .join(c.set_index(idx)[["success", "primary"]].add_prefix("ctrl0_"))
         .join(ref.set_index(idx)[["success"]].add_prefix("official_")))
    p.to_csv(out / "paired.csv")
    arms = {"dsrl (actor_iter4)": p.dsrl_success, "ctrl0 (actor_iter0, cùng noise)": p.ctrl0_success,
            "official (eval cũ, noise khác)": p.official_success.astype(bool)}
    lines = ["# Đối chứng DSRL: actor chưa train, cùng noise", "",
             "Sinh bởi `scripts/dsrl_eval_ctrl.py`. ctrl0 = `run1/actor_iter0.pt` (mean = 0, tức SmolVLA official với "
             "noise N(0, I)), cùng seed env, cùng init, cùng seed noise với eval của dsrl. Hai arm chỉ khác nhau ở mean "
             "của actor.", "", "| arm | s3000 | s4000 | s5000 | gộp |", "|---|---|---|---|---|"]
    for name, s in arms.items():
        s = s.astype(bool)
        cells = [f"{int(s.xs(sd, level='seed').sum())}/50" for sd in EVAL_SEEDS]
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {int(s.sum())}/{len(s)} = {100 * s.mean():.1f}% |")
    lines += ["", "McNemar chính xác, cặp theo (seed, init):", ""]
    for x_name, y_name, x, y in [("dsrl", "ctrl0", p.dsrl_success, p.ctrl0_success),
                                 ("ctrl0", "official", p.ctrl0_success, p.official_success.astype(bool))]:
        b, cc = int((x & ~y).sum()), int((~x & y).sum())
        lines.append(f"- {x_name} vs {y_name}: chỉ {x_name} {b}, chỉ {y_name} {cc}, p = {mcnemar_exact(b, cc):.3f}")
    f = c[~c.success]
    lines += ["", f"Nhãn chính v2 của các thất bại ctrl0 ({len(f)}): "
              + ", ".join(f"{k} {v}" for k, v in f.primary.value_counts().items())]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
