# FPO v4: sanity check trước khi train (CPU)

Script `improve/scripts/fpo_v4_sanity.py`, dùng `src/fpo_adv.py` (cùng code với trainer). Dữ liệu iter 0: `outputs/fpo_v2/iter0` (= v3 iter 0 = v4 iter 0), lọc như `load_decisions`. TC = rollout thành công, TB = thất bại.

## 1. Unit test (nhóm giả: 1 rollout thành công 3 decision, 1 thất bại 10 decision; segment giả): ĐẠT

- segment: tổng w của 3 segment (kỳ vọng 1 / 1 / 1): 1.000 / 1.000 / 1.000
- tổng w rollout thành công ngắn (3 decision): 1.0
- tổng w rollout thất bại dài (10 decision): 1.0
- weighted mean A (kỳ vọng 0,15): 0.1499999985098838
- chế độ decision/mean = công thức v3: True

## 2. Iter 0: dấu của Â theo cách chuẩn hoá

Số v3 phải trùng với `results/round2/fpo_v3/adv_check.md` (iter 0: TC Â < 0 = 1,1 % / 1,6 %, TB Â > 0 = 52,6 % / 37,0 %).

| task | cách | decision | TC: Â < 0 | TB: Â > 0 | Â TB theo segment | Â TB theo rollout | Â TB theo decision | tổng w rollout TC / TB | số rollout TC / TB | TC: A thô < 0 | TB: A thô > 0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | v2 (A / std chung) | 5054 | 29.5% | 19.9% | -0.003 | -0.145 | -0.306 | 1674 / 3380 | 55 / 65 | 29.5% | 19.9% |
| 0 | v3 (decision, trừ mean_t) | 5054 | 1.1% | 52.6% | 0.265 | 0.141 | -0.0 | 1674 / 3380 | 55 / 65 | 29.5% | 19.9% |
| 0 | v4-rollout (bỏ, không chạy) | 5054 | 29.5% | 19.9% | -0.002 | -0.128 | -0.271 | 55 / 65 | 55 / 65 | 29.5% | 19.9% |
| 0 | v4-segment (chạy) | 5054 | 29.5% | 19.9% | -0.002 | -0.135 | -0.286 | 110 / 98 | 55 / 65 | 29.5% | 19.9% |
| 8 | v2 (A / std chung) | 6191 | 16.6% | 21.5% | -0.002 | -0.094 | -0.157 | 1719 / 4472 | 42 / 86 | 16.6% | 21.5% |
| 8 | v3 (decision, trừ mean_t) | 6191 | 1.6% | 37.0% | 0.18 | 0.073 | 0.0 | 1719 / 4472 | 42 / 86 | 16.6% | 21.5% |
| 8 | v4-rollout (bỏ, không chạy) | 6191 | 16.6% | 21.5% | -0.002 | -0.107 | -0.178 | 42 / 86 | 42 / 86 | 16.6% | 21.5% |
| 8 | v4-segment (chạy) | 6191 | 16.6% | 21.5% | -0.002 | -0.107 | -0.178 | 84 / 141 | 42 / 86 | 16.6% | 21.5% |
