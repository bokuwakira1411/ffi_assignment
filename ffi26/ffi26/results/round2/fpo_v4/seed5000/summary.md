# Tóm tắt kết quả round 2 (RL có điều kiện theo advantage)

Nguồn: tính lại từ file từng episode (`episodes.csv`); CI = Wilson 95%. Held-out = init state không dùng để thu rollout (34 / task). McNemar chính xác so với `official` trên cùng init state + seed env (task đã train).

| model | task 0 | task 8 | task đã train (tất cả) | held-out | 8 task khác | chỉ model / chỉ official | McNemar p |
|---|---|---|---|---|---|---|---|
| checkpoint chính thức, expert fp32, seed 5000 | 22/50 = 44% (31–58) | 20/50 = 40% (28–54) | 42/100 = 42% (33–52) | 29/68 = 43% (32–54) | — | 0 / 0 | — |
| FPO v2 θ4, expert fp32, seed 5000 | 15/50 = 30% (19–44) | 21/50 = 42% (29–56) | 36/100 = 36% (27–46) | 25/68 = 37% (26–49) | — | 12 / 18 | 0.36 |
| FPO v3 θ4, expert fp32, seed 5000 | 16/50 = 32% (21–46) | 27/50 = 54% (40–67) | 43/100 = 43% (34–53) | 28/68 = 41% (30–53) | — | 12 / 11 | 1.00 |
| FPO v4 (checkpoint theo quy tắc collapse), expert fp32, seed 5000 | 22/50 = 44% (31–58) | 24/50 = 48% (35–61) | 46/100 = 46% (37–56) | 29/68 = 43% (32–54) | — | 14 / 10 | 0.54 |

## Kiểm tra khoảng cách positive vs negative


## Kiểu thất bại trên task đã train (nhãn chính, chỉ episode thất bại) và số sự kiện

Cột A–O: số episode thất bại theo nhãn chính (`src/failure_labels.py`); n_drop/n_slip/n_collision: số sự kiện (`src/reward_r2.py`) trên mọi episode.

| model | task | A gắp hỏng | B kẹt chuyển subtask | C nhầm vật | D làm hỏng tiến độ | E đứng yên | O khác | n_drop | n_slip | n_collision |
|---|---|---|---|---|---|---|---|---|---|---|
| fpo_v2_sel | 0 | 21 | 0 | 12 | 0 | 2 | 0 | 3 | 33 | 14 |
| fpo_v2_sel | 8 | 6 | 1 | 0 | 4 | 18 | 0 | 1 | 23 | 0 |
| fpo_v3_sel | 0 | 21 | 0 | 11 | 0 | 2 | 0 | 5 | 30 | 18 |
| fpo_v3_sel | 8 | 7 | 2 | 0 | 2 | 12 | 0 | 2 | 7 | 0 |
| fpo_v4_sel | 0 | 17 | 0 | 10 | 0 | 1 | 0 | 4 | 30 | 14 |
| fpo_v4_sel | 8 | 8 | 1 | 0 | 6 | 10 | 1 | 8 | 10 | 0 |
| official | 0 | 18 | 1 | 6 | 0 | 3 | 0 | 4 | 32 | 14 |
| official | 8 | 10 | 6 | 0 | 3 | 11 | 0 | 2 | 11 | 0 |
