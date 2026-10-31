# Tóm tắt kết quả round 2 (RL có điều kiện theo advantage)

Nguồn: tính lại từ file từng episode (`episodes.csv`); CI = Wilson 95%. Held-out = init state không dùng để thu rollout (34 / task). McNemar chính xác so với `official` trên cùng init state + seed env (task đã train).

| model | task 0 | task 8 | task đã train (tất cả) | held-out | 8 task khác | chỉ model / chỉ official | McNemar p |
|---|---|---|---|---|---|---|---|
| checkpoint chính thức, seed eval 2000 | 21/50 = 42% (29–56) | 24/50 = 48% (35–61) | 45/100 = 45% (36–55) | 29/68 = 43% (32–54) | — | 0 / 0 | — |
| FPO sau vòng 3, seed eval 2000 | 19/50 = 38% (26–52) | 27/50 = 54% (40–67) | 46/100 = 46% (37–56) | 31/68 = 46% (34–57) | — | 18 / 17 | 1.00 |

## Kiểm tra khoảng cách positive vs negative


## Kiểu thất bại trên task đã train (nhãn chính, chỉ episode thất bại) và số sự kiện

Cột A–O: số episode thất bại theo nhãn chính (`src/failure_labels.py`); n_drop/n_slip/n_collision: số sự kiện (`src/reward_r2.py`) trên mọi episode.

| model | task | A gắp hỏng | B kẹt chuyển subtask | C nhầm vật | D làm hỏng tiến độ | E đứng yên | n_drop | n_slip | n_collision |
|---|---|---|---|---|---|---|---|---|---|
| fpo_it3 | 0 | 23 | 1 | 5 | 0 | 2 | 4 | 58 | 12 |
| fpo_it3 | 8 | 9 | 1 | 0 | 1 | 12 | 1 | 15 | 0 |
| official | 0 | 19 | 0 | 5 | 0 | 5 | 4 | 29 | 13 |
| official | 8 | 5 | 2 | 0 | 2 | 17 | 1 | 6 | 0 |
