# Tóm tắt kết quả round 2 (RL có điều kiện theo advantage)

Nguồn: tính lại từ file từng episode (`episodes.csv`); CI = Wilson 95%. Held-out = init state không dùng để thu rollout (34 / task). McNemar chính xác so với `official` trên cùng init state + seed env (task đã train).

| model | task 0 | task 8 | task đã train (tất cả) | held-out | 8 task khác | chỉ model / chỉ official | McNemar p |
|---|---|---|---|---|---|---|---|
| checkpoint chính thức, expert fp32, seed 1000 | 19/50 = 38% (26–52) | 24/50 = 48% (35–61) | 43/100 = 43% (34–53) | 28/68 = 41% (30–53) | 127/160 = 79% (72–85) | 0 / 0 | — |
| FPO v2 θ4, expert fp32, seed 1000 | — | — | — | — | 108/160 = 68% (60–74) | 0 / 0 | 1.00 |
| FPO v3 θ4, expert fp32, seed 1000 | — | — | — | — | 115/160 = 72% (64–78) | 0 / 0 | 1.00 |
| FPO v4 (checkpoint theo quy tắc collapse), expert fp32, seed 1000 | — | — | — | — | 125/160 = 78% (71–84) | 0 / 0 | 1.00 |

## Kiểm tra khoảng cách positive vs negative


## Kiểu thất bại trên task đã train (nhãn chính, chỉ episode thất bại) và số sự kiện

Cột A–O: số episode thất bại theo nhãn chính (`src/failure_labels.py`); n_drop/n_slip/n_collision: số sự kiện (`src/reward_r2.py`) trên mọi episode.

| model | task | A gắp hỏng | B kẹt chuyển subtask | C nhầm vật | D làm hỏng tiến độ | E đứng yên | O khác | n_drop | n_slip | n_collision |
|---|---|---|---|---|---|---|---|---|---|---|
| official | 0 | 23 | 0 | 5 | 0 | 3 | 0 | 1 | 40 | 12 |
| official | 8 | 8 | 6 | 0 | 1 | 10 | 1 | 1 | 8 | 0 |
