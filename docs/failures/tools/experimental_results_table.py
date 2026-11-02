"""Bảng tổng hợp eval cuối cho experimental_results/README.md: SmolVLA gốc và FPO v1–v4 trên cùng episode.

Tỉ lệ thành công + McNemar chính xác hai phía theo cặp với official (cùng seed, task, init):
  task 0, 8 × 50 init × seed 3000/4000/5000: results/round2/fpo_v4/seed*/episodes.csv (official, v2, v3, v4),
  results/round2/fpo_v2/seed*/episodes.csv (fpo_v1_it3);
  8 task khác × 20 init × seed 1000: .../seed1000_other/episodes.csv.
Kiểu thất bại (task 0 + 8, chỉ episode thất bại): nhãn tự động v2 (improve/src/failure_labels.label_episode_v2),
  improve/outputs/failure_labels{,_v3,_v4}/labels.csv; slip: cột n_slip của episodes.csv.
Chỉ đọc, không ghi file. Chạy: python3 docs/failures/tools/experimental_results_table.py  (từ thư mục gốc repo)
"""

import csv
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3] / "ffi26/ffi26"
R2, OUT = ROOT / "results/round2", ROOT / "improve/outputs"
SEEDS = ("3000", "4000", "5000")
ARMS = [("SmolVLA gốc", "official", "fpo_v4", "failure_labels"), ("FPO v1", "fpo_v1_it3", "fpo_v2", "failure_labels"),
        ("FPO v2", "fpo_v2_sel", "fpo_v4", "failure_labels"), ("FPO v3", "fpo_v3_sel", "fpo_v4", "failure_labels_v3"),
        ("FPO v4", "fpo_v4_sel", "fpo_v4", "failure_labels_v4")]


def mcnemar(b: int, c: int) -> float:
    n = b + c
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2**n) if n else 1.0


def episodes(src: str, seeds) -> dict:
    return {(r["model"], s, r["task"], r["init"]): r for s in seeds
            for r in csv.DictReader(open(R2 / f"{src}/seed{s}/episodes.csv"))}


ok = lambda r: r["success"] == "True"  # noqa: E731
for name, arm, src, lab in ARMS:
    main, ref = episodes(src, SEEDS), episodes("fpo_v4", SEEDS)
    other, ref_o = episodes(src, ("1000_other",)), episodes("fpo_v4", ("1000_other",))
    cells = []
    for eps, base, sel in ((main, ref, ("0",)), (main, ref, ("8",)), (main, ref, ("0", "8")),
                           (other, ref_o, tuple(str(t) for t in range(10) if t not in (0, 8)))):
        ks = [k[1:] for k in base if k[0] == "official" and k[2] in sel]
        x = [ok(eps[(arm,) + k]) for k in ks]
        o = [ok(base[("official",) + k]) for k in ks]
        b = sum(a and not c for a, c in zip(x, o))
        c = sum(c and not a for a, c in zip(x, o))
        cells.append(f"{sum(x)}/{len(ks)} = {100 * sum(x) / len(ks):.1f}% (p {mcnemar(b, c):.3f})")
    labs = [r for r in csv.DictReader(open(OUT / lab / "labels.csv"))
            if r["arm"] == arm and r["seed"] in SEEDS and r["task"] in ("0", "8") and r["success"] == "False"]
    nA = sum(r["lab_A"] == "True" for r in labs)
    nslip = sum(int(main[(arm, r["seed"], r["task"], r["init"])]["n_slip"]) > 0 for r in labs)
    nhalf = sum(r["subtask_completion"] == "0.5" for r in labs)
    print(f"{name}: task0 {cells[0]} | task8 {cells[1]} | task0+8 {cells[2]} | 8 task khác {cells[3]} | "
          f"thất bại {len(labs)}, gắp hỏng {nA}, slip {nslip}, xong 1/2 goal {nhalf}")
