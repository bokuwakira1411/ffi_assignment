### A1. Task 0, eval seed 3000/4000/5000 (150 episode/arm)

Nguồn: `improve/outputs/task0_diag/A1_task0.csv`

| arm | n | thành công | thất bại | chính A | chính C | chính F | chính E_contact | chính O | có A | có C | có F | có E_contact | có E_free | có B | trung vị bắt đầu E_free | trung vị bắt đầu E_contact | độ dài trung vị thất bại |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| official | 150 | 61 | 89 | 73 | 8 | 3 | 4 | 1 | 73 | 33 | 6 | 37 | 14 | 21 | 351 | 185 | 520 |
| v2 θ4 | 150 | 50 | 100 | 79 | 16 | 2 | 3 | 0 | 79 | 43 | 4 | 35 | 51 | 14 | 322 | 173 | 520 |
| v3 θ4 | 150 | 46 | 104 | 75 | 19 | 8 | 2 | 0 | 75 | 45 | 12 | 42 | 18 | 19 | 363 | 175 | 520 |

### A2a. Task 8, eval seed 3000/4000/5000 (150 episode/arm)

Nguồn: `improve/outputs/task0_diag/A2_task8.csv`

| arm | n | thành công | thất bại | chính A | chính F | chính G | chính D | chính E_contact | chính E_free | chính B | chính O | có A | có F | có G | có D | có E_contact | có E_free | có B | trung vị bắt đầu E_free | trung vị bắt đầu E_contact | độ dài trung vị thất bại | 0 ấm | 1 ấm | 2 ấm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| official | 150 | 60 | 90 | 25 | 3 | 4 | 1 | 31 | 3 | 23 | 0 | 25 | 3 | 4 | 5 | 55 | 4 | 73 | 377 | 271 | 520 | 14 | 73 | 3 |
| v2 θ4 | 150 | 76 | 74 | 42 | 2 | 2 | 1 | 24 | 2 | 0 | 1 | 42 | 2 | 2 | 6 | 67 | 7 | 22 | 369 | 136 | 520 | 50 | 22 | 2 |
| v3 θ4 | 150 | 88 | 62 | 25 | 4 | 4 | 1 | 25 | 0 | 3 | 0 | 25 | 4 | 4 | 5 | 52 | 2 | 46 | 330 | 274 | 520 | 13 | 46 | 3 |

### A2b. Task 4, eval seed 1000 (20 episode/arm)

Nguồn: `improve/outputs/task0_diag/A2_task4.csv`

| arm | n | thành công | thất bại | chính A | chính F | chính G | chính E_contact | chính E_free | chính O | có A | có F | có G | có E_contact | có E_free | có B | trung vị bắt đầu E_free | trung vị bắt đầu E_contact | độ dài trung vị thất bại |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| official | 20 | 13 | 7 | 2 | 0 | 3 | 2 | 0 | 0 | 2 | 0 | 4 | 3 | 3 | 4 | 301 | 260 | 520 |
| v2 θ4 | 20 | 12 | 8 | 1 | 0 | 3 | 2 | 1 | 1 | 1 | 0 | 3 | 4 | 4 | 5 | 357 | 247 | 520 |
| v3 θ4 | 20 | 9 | 11 | 2 | 1 | 6 | 1 | 0 | 1 | 2 | 1 | 7 | 2 | 6 | 5 | 280 | 228 | 520 |

### A2c. Forgetting: 8 task × 20 episode, seed 1000

Nguồn: `improve/outputs/task0_diag/A2_forget.csv`

| arm | n | thành công | thất bại | chính A | chính C | chính F | chính G | chính D | chính E_contact | chính E_free | chính B | chính O |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| official | 160 | 127 | 33 | 13 | 1 | 3 | 7 | 1 | 6 | 0 | 1 | 1 |
| v2 θ4 | 160 | 108 | 52 | 32 | 0 | 3 | 6 | 0 | 7 | 3 | 0 | 1 |
| v3 θ4 | 160 | 115 | 45 | 18 | 1 | 6 | 7 | 1 | 8 | 0 | 2 | 2 |

### A3a. Collection v3, task 0 (128 rollout/iteration; iter 4 = θ4)

Nguồn: `improve/outputs/task0_diag/A3_collect_task0.csv`

| iter | n | thành công | thất bại | chính A | chính C | chính F | chính G | chính E_contact | chính B | có A | có C | có F | có G | có D | có E_contact | có E_free | có B | trung vị bắt đầu E_free | trung vị bắt đầu E_contact | độ dài trung vị thất bại |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 128 | 55 | 73 | 55 | 4 | 7 | 0 | 4 | 3 | 55 | 10 | 11 | 0 | 1 | 31 | 17 | 33 | 391 | 193 | 520 |
| 1 | 128 | 58 | 70 | 57 | 3 | 8 | 1 | 0 | 1 | 57 | 8 | 11 | 2 | 0 | 28 | 18 | 27 | 350 | 193 | 520 |
| 2 | 128 | 61 | 67 | 56 | 4 | 5 | 0 | 1 | 1 | 56 | 11 | 9 | 0 | 0 | 33 | 18 | 25 | 419 | 165 | 520 |
| 3 | 128 | 54 | 74 | 62 | 4 | 6 | 0 | 1 | 1 | 62 | 13 | 9 | 0 | 0 | 32 | 23 | 29 | 380 | 195 | 520 |
| 4 | 128 | 49 | 79 | 72 | 0 | 5 | 0 | 2 | 0 | 72 | 20 | 5 | 0 | 0 | 34 | 27 | 20 | 369 | 192 | 520 |

### A3b. Collection v3, task 8 (128 rollout/iteration)

Nguồn: `improve/outputs/task0_diag/A3_collect_task8.csv`

| iter | n | thành công | thất bại | chính A | chính F | chính G | chính D | chính E_contact | chính E_free | chính B | có A | có F | có G | có D | có E_contact | có E_free | có B | trung vị bắt đầu E_free | trung vị bắt đầu E_contact | độ dài trung vị thất bại | 0 ấm | 1 ấm | 2 ấm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 128 | 42 | 86 | 33 | 0 | 1 | 0 | 34 | 2 | 16 | 33 | 0 | 1 | 0 | 62 | 6 | 55 | 453 | 110 | 520 | 31 | 55 | 0 |
| 1 | 128 | 69 | 59 | 26 | 2 | 0 | 1 | 25 | 0 | 5 | 26 | 2 | 0 | 1 | 52 | 1 | 41 | 443 | 271 | 520 | 17 | 41 | 1 |
| 2 | 128 | 65 | 63 | 27 | 4 | 1 | 2 | 27 | 0 | 2 | 27 | 4 | 2 | 3 | 55 | 3 | 45 | 394 | 273 | 520 | 16 | 45 | 2 |
| 3 | 128 | 59 | 69 | 28 | 3 | 2 | 3 | 27 | 0 | 6 | 28 | 3 | 2 | 4 | 54 | 2 | 47 | 451 | 270 | 520 | 21 | 47 | 1 |
| 4 | 128 | 75 | 53 | 22 | 3 | 1 | 0 | 23 | 1 | 3 | 22 | 3 | 1 | 1 | 47 | 3 | 43 | 454 | 281 | 520 | 9 | 43 | 1 |

### B. Cosine gradient, CPU, n = 32/nửa, DỪNG giữa chừng (làm chậm v4) (mean ± std qua các cách chia)

Nguồn: `improve/outputs/task0_diag/grad_cos.csv` (từng lần chia), `grad_cos_summary.csv`

| θ_k (iter k) | n chia | đối chứng cos(g0a, g0b) | đối chứng cos(g8a, g8b) | cos(g0 nửa, g8 nửa) | cos(g0, g8) | cos(g0, g_demo) | cos(g8, g_demo) | ‖g0‖ | ‖g8‖ | ‖g_demo‖ |
|---|---|---|---|---|---|---|---|---|---|---|
| θ0 | 5 | -0.070 ± 0.069 | +0.071 ± 0.039 | +0.003 ± 0.026 | -0.025 ± 0.028 | -0.007 ± 0.052 | +0.035 ± 0.128 | 10.66 ± 3.08 | 18.13 ± 16.26 | 2.94 ± 1.11 |
| θ1 | 3 | -0.164 ± 0.179 | -0.012 ± 0.100 | +0.016 ± 0.036 | +0.052 ± 0.126 | -0.033 ± 0.070 | -0.020 ± 0.080 | 10.22 ± 2.37 | 8.41 ± 0.51 | 3.52 ± 1.00 |

