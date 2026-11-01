# Tóm tắt kết quả round 2 (RL có điều kiện theo advantage)

Nguồn: tính lại từ file từng episode (`episodes.csv`); CI = Wilson 95%. Held-out = init state không dùng để thu rollout (34 / task). McNemar chính xác so với `official` trên cùng init state + seed env (task đã train).

| model | task 0 | task 8 | task đã train (tất cả) | held-out | 8 task khác | chỉ model / chỉ official | McNemar p |
|---|---|---|---|---|---|---|---|
| checkpoint chính thức, expert fp32, seed 3000 | 20/50 = 40% (28–54) | 18/50 = 36% (24–50) | 38/100 = 38% (29–48) | 28/68 = 41% (30–53) | — | 0 / 0 | — |
| FPO v2 θ4, expert fp32, seed 3000 | 19/50 = 38% (26–52) | 26/50 = 52% (39–65) | 45/100 = 45% (36–55) | 30/68 = 44% (33–56) | — | 18 / 11 | 0.26 |
| FPO v3 (checkpoint theo quy tắc collapse), expert fp32, seed 3000 | 15/50 = 30% (19–44) | 27/50 = 54% (40–67) | 42/100 = 42% (33–52) | 26/68 = 38% (28–50) | — | 17 / 13 | 0.58 |

## Kiểm tra khoảng cách positive vs negative


## Kiểu thất bại trên task đã train (nhãn chính, chỉ episode thất bại) và số sự kiện

Cột A–O: số episode thất bại theo nhãn chính (`src/failure_labels.py`); n_drop/n_slip/n_collision: số sự kiện (`src/reward_r2.py`) trên mọi episode.

| model | task | A gắp hỏng | B kẹt chuyển subtask | C nhầm vật | D làm hỏng tiến độ | E đứng yên | O khác | n_drop | n_slip | n_collision |
|---|---|---|---|---|---|---|---|---|---|---|
| fpo_v2_sel | 0 | 20 | 0 | 9 | 0 | 2 | 0 | 1 | 32 | 16 |
| fpo_v2_sel | 8 | 9 | 0 | 0 | 1 | 13 | 1 | 1 | 11 | 0 |
| fpo_v3_sel | 0 | 16 | 1 | 16 | 0 | 2 | 0 | 7 | 36 | 14 |
| fpo_v3_sel | 8 | 10 | 2 | 0 | 3 | 8 | 0 | 2 | 9 | 0 |
| official | 0 | 23 | 0 | 6 | 0 | 0 | 1 | 0 | 39 | 13 |
| official | 8 | 7 | 7 | 0 | 4 | 14 | 0 | 1 | 9 | 0 |
