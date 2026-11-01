# FPO v3: tách task × init (held-out / init đã train)

Script `improve/scripts/fpo_v3_split.py`; dữ liệu = file gốc `ep*.meta.json` như `pooled.md` (official, v2: `outputs/fpo_v2/eval_s*`; v3: `outputs/fpo_v3/eval_s*/fpo_v3_sel`). Init đã train = `collect.init_ids` trong `fpo_v3.yaml` (16 init/task, = v2), còn lại 34 init/task là held-out. Cặp = cùng (seed, task, init). McNemar chính xác hai phía; "chỉ A / chỉ B" = số cặp chỉ A thành công / chỉ B thành công. 4 ô × 2 so sánh, không hiệu chỉnh nhiều lần.

## gộp 3 seed

| task | init | official | v2 θ4 | v3 θ4 | v3 vs official: chỉ v3 / chỉ off., p | v3 vs v2: chỉ v3 / chỉ v2, p |
|---|---|---|---|---|---|---|
| 0 | held-out | 39/102 = 38.2% | 34/102 = 33.3% | 29/102 = 28.4% | 5/15, p = 0.041 | 6/11, p = 0.332 |
| 0 | đã train | 22/48 = 45.8% | 16/48 = 33.3% | 17/48 = 35.4% | 4/9, p = 0.267 | 6/5, p = 1.000 |
| 8 | held-out | 45/102 = 44.1% | 52/102 = 51.0% | 58/102 = 56.9% | 19/6, p = 0.015 | 23/17, p = 0.430 |
| 8 | đã train | 15/48 = 31.2% | 24/48 = 50.0% | 30/48 = 62.5% | 17/2, p = 0.001 | 14/8, p = 0.286 |

## seed 3000

| task | init | official | v2 θ4 | v3 θ4 | v3 vs official: chỉ v3 / chỉ off., p | v3 vs v2: chỉ v3 / chỉ v2, p |
|---|---|---|---|---|---|---|
| 0 | held-out | 14/34 = 41.2% | 12/34 = 35.3% | 9/34 = 26.5% | 2/7, p = 0.180 | 1/4, p = 0.375 |
| 0 | đã train | 6/16 = 37.5% | 7/16 = 43.8% | 6/16 = 37.5% | 2/2, p = 1.000 | 1/2, p = 1.000 |
| 8 | held-out | 14/34 = 41.2% | 18/34 = 52.9% | 17/34 = 50.0% | 6/3, p = 0.508 | 8/9, p = 1.000 |
| 8 | đã train | 4/16 = 25.0% | 8/16 = 50.0% | 10/16 = 62.5% | 7/1, p = 0.070 | 4/2, p = 0.688 |

## seed 4000

| task | init | official | v2 θ4 | v3 θ4 | v3 vs official: chỉ v3 / chỉ off., p | v3 vs v2: chỉ v3 / chỉ v2, p |
|---|---|---|---|---|---|---|
| 0 | held-out | 11/34 = 32.4% | 11/34 = 32.4% | 10/34 = 29.4% | 2/3, p = 1.000 | 2/3, p = 1.000 |
| 0 | đã train | 8/16 = 50.0% | 5/16 = 31.2% | 5/16 = 31.2% | 1/4, p = 0.375 | 2/2, p = 1.000 |
| 8 | held-out | 16/34 = 47.1% | 20/34 = 58.8% | 23/34 = 67.6% | 8/1, p = 0.039 | 7/4, p = 0.549 |
| 8 | đã train | 6/16 = 37.5% | 9/16 = 56.2% | 11/16 = 68.8% | 5/0, p = 0.062 | 5/3, p = 0.727 |

## seed 5000

| task | init | official | v2 θ4 | v3 θ4 | v3 vs official: chỉ v3 / chỉ off., p | v3 vs v2: chỉ v3 / chỉ v2, p |
|---|---|---|---|---|---|---|
| 0 | held-out | 14/34 = 41.2% | 11/34 = 32.4% | 10/34 = 29.4% | 1/5, p = 0.219 | 3/4, p = 1.000 |
| 0 | đã train | 8/16 = 50.0% | 4/16 = 25.0% | 6/16 = 37.5% | 1/3, p = 0.625 | 3/1, p = 0.625 |
| 8 | held-out | 15/34 = 44.1% | 14/34 = 41.2% | 18/34 = 52.9% | 5/2, p = 0.453 | 8/4, p = 0.388 |
| 8 | đã train | 5/16 = 31.2% | 7/16 = 43.8% | 9/16 = 56.2% | 5/1, p = 0.219 | 5/3, p = 0.727 |
