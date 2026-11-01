# Improve: RL cho SmolVLA

Xuất phát từ `lerobot/smolvla_libero` (NAS = 10). Train trên task 0 và task 8 của libero_10, chỉ action expert (VLM đóng băng).

## Các phương pháp đã thử

| | Ý tưởng | Kết quả |
|---|---|---|
| A1 | gắn nhãn advantage (text hoặc embedding) vào rollout rồi fine-tune | không hơn checkpoint gốc |
| AWR | fine-tune có trọng số theo advantage | không hơn checkpoint gốc |
| FPO v1 | PPO-clip với tỉ số xấp xỉ bằng loss flow matching, on-policy 4 vòng | tốt hơn ở seed eval 1000, không lặp lại ở seed 2000 |
| FPO v2 | v1 + expert fp32, batch 128, neo BC | task 8 tăng, task 0 giảm, quên nhiều |
| FPO v3 | v2 + demo replay, chuẩn hoá advantage theo task | giảm quên, task 0 vẫn giảm |
| **FPO v4** | v3 + trọng số advantage theo segment, không trừ mean | **tốt nhất** |

## Kết quả cuối (task 0 + 8, 3 seed × 50 init; quên: 8 task khác × 20 init)

| | official | v2 | v3 | v4 |
|---|---|---|---|---|
| task 0 + 8 | 40,3% | 42,0% | 44,7% | **46,7%** |
| 8 task khác | 79,4% | 67,5% | 71,9% | 78,1% |

v4 hơn official có ý nghĩa thống kê, nhờ task 8; task 0 ngang official. Bảng đầy đủ, so sánh theo cặp và p-value: `fpo_v4/pooled.md`.

## Thư mục

| | |
|---|---|
| `configs/` | config train / eval / report của mọi phương pháp (`fpo_v{2,3,4}.yaml`, `eval_*`, `report_*`) |
| `fpo_v{2,3,4}/seed*/` | kết quả theo từng seed eval |
| `fpo_v3/`, `fpo_v4/` | `pooled.md` (bảng gộp các seed, so sánh theo cặp), `episodes_all.csv`; kiểm tra phụ: advantage, số bước gradient (v3), sanity (v4) |
| `summary.md`, `round1/`, `seed2000/` | vòng sàng lọc A1 / AWR / FPO v1 |
| `detector_check/` | kiểm tra bộ phát hiện drop / slip / collision |
| `fpo_plan.md` | thiết kế FPO |

Cách chạy lại: `readme.md` ở gốc repo.
