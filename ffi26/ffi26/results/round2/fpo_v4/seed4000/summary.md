# Tóm tắt kết quả round 2 (RL có điều kiện theo advantage)

Nguồn: tính lại từ file từng episode (`episodes.csv`); CI = Wilson 95%. Held-out = init state không dùng để thu rollout (34 / task). McNemar chính xác so với `official` trên cùng init state + seed env (task đã train).

| model | task 0 | task 8 | task đã train (tất cả) | held-out | 8 task khác | chỉ model / chỉ official | McNemar p |
|---|---|---|---|---|---|---|---|
| checkpoint chính thức, expert fp32, seed 4000 | 19/50 = 38% (26–52) | 22/50 = 44% (31–58) | 41/100 = 41% (32–51) | 27/68 = 40% (29–52) | — | 0 / 0 | — |
| FPO v2 θ4, expert fp32, seed 4000 | 16/50 = 32% (21–46) | 29/50 = 58% (44–71) | 45/100 = 45% (36–55) | 31/68 = 46% (34–57) | — | 19 / 15 | 0.61 |
| FPO v3 θ4, expert fp32, seed 4000 | 15/50 = 30% (19–44) | 34/50 = 68% (54–79) | 49/100 = 49% (39–59) | 33/68 = 49% (37–60) | — | 16 / 8 | 0.15 |
| FPO v4 (checkpoint theo quy tắc collapse), expert fp32, seed 4000 | 19/50 = 38% (26–52) | 28/50 = 56% (42–69) | 47/100 = 47% (38–57) | 33/68 = 49% (37–60) | — | 12 / 6 | 0.24 |

## Kiểm tra khoảng cách positive vs negative


## Kiểu thất bại trên task đã train (nhãn chính, chỉ episode thất bại) và số sự kiện

Cột A–O: số episode thất bại theo nhãn chính (`src/failure_labels.py`); n_drop/n_slip/n_collision: số sự kiện (`src/reward_r2.py`) trên mọi episode.

| model | task | A gắp hỏng | B kẹt chuyển subtask | C nhầm vật | D làm hỏng tiến độ | E đứng yên | O khác | n_drop | n_slip | n_collision |
|---|---|---|---|---|---|---|---|---|---|---|
| fpo_v2_sel | 0 | 25 | 0 | 8 | 0 | 1 | 0 | 1 | 35 | 15 |
| fpo_v2_sel | 8 | 9 | 0 | 0 | 1 | 10 | 1 | 1 | 13 | 0 |
| fpo_v3_sel | 0 | 25 | 1 | 7 | 0 | 2 | 0 | 2 | 30 | 18 |
| fpo_v3_sel | 8 | 7 | 1 | 0 | 2 | 6 | 0 | 2 | 8 | 0 |
| fpo_v4_sel | 0 | 24 | 0 | 5 | 1 | 1 | 0 | 3 | 24 | 12 |
| fpo_v4_sel | 8 | 7 | 1 | 0 | 5 | 9 | 0 | 0 | 9 | 0 |
| official | 0 | 26 | 1 | 3 | 0 | 1 | 0 | 2 | 26 | 11 |
| official | 8 | 6 | 8 | 0 | 5 | 9 | 0 | 2 | 7 | 0 |
