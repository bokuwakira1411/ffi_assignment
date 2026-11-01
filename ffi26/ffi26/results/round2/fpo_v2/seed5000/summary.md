# Tóm tắt kết quả round 2 (RL có điều kiện theo advantage)

Nguồn: tính lại từ file từng episode (`episodes.csv`); CI = Wilson 95%. Held-out = init state không dùng để thu rollout (34 / task). McNemar chính xác so với `official` trên cùng init state + seed env (task đã train).

| model | task 0 | task 8 | task đã train (tất cả) | held-out | 8 task khác | chỉ model / chỉ official | McNemar p |
|---|---|---|---|---|---|---|---|
| checkpoint chính thức, expert fp32, seed 5000 | 22/50 = 44% (31–58) | 20/50 = 40% (28–54) | 42/100 = 42% (33–52) | 29/68 = 43% (32–54) | — | 0 / 0 | — |
| FPO v1 θ3, expert fp32, seed 5000 | 18/50 = 36% (24–50) | 8/50 = 16% (8–29) | 26/100 = 26% (18–35) | 19/68 = 28% (19–40) | — | 9 / 25 | 0.01 |
| FPO v2 (checkpoint theo quy tắc collapse), expert fp32, seed 5000 | 15/50 = 30% (19–44) | 21/50 = 42% (29–56) | 36/100 = 36% (27–46) | 25/68 = 37% (26–49) | — | 12 / 18 | 0.36 |

## Kiểm tra khoảng cách positive vs negative


## Kiểu thất bại trên task đã train (nhãn chính, chỉ episode thất bại) và số sự kiện

Cột A–O: số episode thất bại theo nhãn chính (`src/failure_labels.py`); n_drop/n_slip/n_collision: số sự kiện (`src/reward_r2.py`) trên mọi episode.

| model | task | A gắp hỏng | B kẹt chuyển subtask | C nhầm vật | D làm hỏng tiến độ | E đứng yên | n_drop | n_slip | n_collision |
|---|---|---|---|---|---|---|---|---|---|
| fpo_v1_it3 | 0 | 20 | 3 | 2 | 1 | 6 | 18 | 38 | 14 |
| fpo_v1_it3 | 8 | 18 | 4 | 0 | 2 | 18 | 1 | 27 | 0 |
| fpo_v2_sel | 0 | 21 | 0 | 12 | 0 | 2 | 3 | 33 | 14 |
| fpo_v2_sel | 8 | 6 | 1 | 0 | 4 | 18 | 1 | 23 | 0 |
| official | 0 | 18 | 1 | 6 | 0 | 3 | 4 | 32 | 14 |
| official | 8 | 10 | 6 | 0 | 3 | 11 | 2 | 11 | 0 |
