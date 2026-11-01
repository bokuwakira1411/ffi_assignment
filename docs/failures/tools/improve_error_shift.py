"""So sánh kiểu lỗi giữa official và FPO v4 trên task 0, 8 (theo cặp seed, task, init), dùng cho report/main.tex
mục "Kiểm tra giả thuyết". Chỉ đọc results/round2/fpo_v4/seed{3000,4000,5000}/episodes.csv, không ghi file.

Các cột dùng: success, subtask_completion (tỉ lệ predicate goal đúng ở cuối episode), n_slip, n_collision, n_drop,
primary (nhãn tự động bản cũ). McNemar chính xác hai phía trên các cặp lệch nhau.
Chạy: python3 docs/failures/tools/improve_error_shift.py  (từ thư mục gốc repo)
"""

import csv
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3] / "ffi26/ffi26/results/round2/fpo_v4"
ARM = "fpo_v4_sel"


def mcnemar(b: int, c: int) -> float:
    n = b + c
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2**n) if n else 1.0


ep = {}
for seed in (3000, 4000, 5000):
    for r in csv.DictReader(open(ROOT / f"seed{seed}/episodes.csv")):
        if r["task"] in ("0", "8"):
            ep[(r["model"], seed, r["task"], r["init"])] = r

failed = lambda r: r["success"] == "False"  # noqa: E731
checks = {
    "thất bại, xong đúng một nửa goal": lambda r: failed(r) and r["subtask_completion"] == "0.5",
    "thất bại, chưa xong vật nào": lambda r: failed(r) and r["subtask_completion"] == "0.0",
    "có slip": lambda r: int(r["n_slip"]) > 0,
    "có collision": lambda r: int(r["n_collision"]) > 0,
    "có drop": lambda r: int(r["n_drop"]) > 0,
    "thất bại, nhãn chính A": lambda r: failed(r) and r["primary"] == "A",
}
pairs = [k[1:] for k in ep if k[0] == "official"]
for name, f in checks.items():
    for task in ("0", "8", "0+8"):
        ks = [k for k in pairs if task == "0+8" or k[1] == task]
        o = [f(ep[("official", *k)]) for k in ks]
        v = [f(ep[(ARM, *k)]) for k in ks]
        b = sum(x and not y for x, y in zip(o, v))
        c = sum(y and not x for x, y in zip(o, v))
        print(f"{name:34s} task {task:4s} official {sum(o):3d}  v4 {sum(v):3d}  "
              f"chỉ official {b:3d}  chỉ v4 {c:3d}  p = {mcnemar(b, c):.3f}")
