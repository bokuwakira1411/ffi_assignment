# Tóm tắt kết quả round 2 (RL có điều kiện theo advantage)

Nguồn: tính lại từ file từng episode (`episodes.csv`); CI = Wilson 95%. Held-out = init state không dùng để thu rollout (34 / task). McNemar chính xác so với `official_bf16` trên cùng init state + seed env (task đã train).

| model | task 0 | task 8 | task đã train (tất cả) | held-out | 8 task khác | chỉ model / chỉ official_bf16 | McNemar p |
|---|---|---|---|---|---|---|---|
| checkpoint chính thức, expert bf16 (outputs/eval_baseline), seed 1000 | 21/50 = 42% (29–56) | 23/50 = 46% (33–60) | 44/100 = 44% (35–54) | 30/68 = 44% (33–56) | — | 0 / 0 | — |
| checkpoint chính thức, expert fp32, seed 1000 | 19/50 = 38% (26–52) | 24/50 = 48% (35–61) | 43/100 = 43% (34–53) | 28/68 = 41% (30–53) | — | 9 / 10 | 1.00 |

## Kiểm tra khoảng cách positive vs negative


## Kiểu thất bại trên task đã train (nhãn chính, chỉ episode thất bại) và số sự kiện

Cột A–O: số episode thất bại theo nhãn chính (`src/failure_labels.py`); n_drop/n_slip/n_collision: số sự kiện (`src/reward_r2.py`) trên mọi episode.

| model | task | A gắp hỏng | B kẹt chuyển subtask | C nhầm vật | D làm hỏng tiến độ | E đứng yên | O khác | n_drop | n_slip | n_collision |
|---|---|---|---|---|---|---|---|---|---|---|
| official_bf16 | 0 | 25 | 0 | 3 | 0 | 0 | 1 | 0 | 43 | 12 |
| official_bf16 | 8 | 10 | 6 | 0 | 1 | 10 | 0 | 2 | 13 | 0 |
| official_fp32 | 0 | 23 | 0 | 5 | 0 | 3 | 0 | 1 | 40 | 12 |
| official_fp32 | 8 | 8 | 6 | 0 | 1 | 10 | 1 | 1 | 8 | 0 |
