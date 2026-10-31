# Tóm tắt kết quả round 2 (RL có điều kiện theo advantage)

Nguồn: tính lại từ file từng episode (`episodes.csv`); CI = Wilson 95%. Held-out = init state không dùng để thu rollout (34 / task). McNemar chính xác so với `official` trên cùng init state + seed env (task đã train).

| model | task 0 | task 8 | task đã train (tất cả) | held-out | 8 task khác | chỉ model / chỉ official | McNemar p |
|---|---|---|---|---|---|---|---|
| checkpoint chính thức (lerobot/smolvla_libero) | 21/50 = 42% (29–56) | 23/50 = 46% (33–60) | 44/100 = 44% (35–54) | 30/68 = 44% (33–56) | 126/160 = 79% (72–84) | 0 / 0 | — |
| A1 round 1, prompt positive | 8/50 = 16% (8–29) | 23/50 = 46% (33–60) | 31/100 = 31% (23–41) | 18/68 = 26% (17–38) | 88/160 = 55% (47–63) | 11 / 24 | 0.04 |
| A1 round 1, không hậu tố | 23/50 = 46% (33–60) | 20/50 = 40% (28–54) | 43/100 = 43% (34–53) | 33/68 = 49% (37–60) | 123/160 = 77% (70–83) | 12 / 13 | 1.00 |
| A1 round 1, prompt negative | 13/50 = 26% (16–40) | 24/50 = 48% (35–61) | 37/100 = 37% (28–47) | 26/68 = 38% (28–50) | — | 10 / 17 | 0.25 |
| A1-emb round 1, nhãn positive | 14/50 = 28% (17–42) | 24/50 = 48% (35–61) | 38/100 = 38% (29–48) | 23/68 = 34% (24–46) | 118/160 = 74% (66–80) | 11 / 17 | 0.34 |
| A1-emb round 1, nhãn none | 22/50 = 44% (31–58) | 22/50 = 44% (31–58) | 44/100 = 44% (35–54) | 31/68 = 46% (34–57) | — | 11 / 11 | 1.00 |
| A1-emb round 1, nhãn negative | 18/50 = 36% (24–50) | 26/50 = 52% (39–65) | 44/100 = 44% (35–54) | 28/68 = 41% (30–53) | — | 13 / 13 | 1.00 |
| A2 AWR round 1 (không nhãn) | 17/50 = 34% (22–48) | 26/50 = 52% (39–65) | 43/100 = 43% (34–53) | 32/68 = 47% (36–59) | 128/160 = 80% (73–85) | 13 / 14 | 1.00 |
| A2 AWR 6000 bước, checkpoint 2000 | 23/50 = 46% (33–60) | 24/50 = 48% (35–61) | 47/100 = 47% (38–57) | 32/68 = 47% (36–59) | 121/160 = 76% (68–82) | 13 / 10 | 0.68 |
| A2 AWR 6000 bước, checkpoint 4000 | 16/50 = 32% (21–46) | 25/50 = 50% (37–63) | 41/100 = 41% (32–51) | 27/68 = 40% (29–52) | 123/160 = 77% (70–83) | 12 / 15 | 0.70 |
| A2 AWR 6000 bước, checkpoint 6000 | 18/50 = 36% (24–50) | 23/50 = 46% (33–60) | 41/100 = 41% (32–51) | 29/68 = 43% (32–54) | 126/160 = 79% (72–84) | 10 / 13 | 0.68 |

## Kiểm tra khoảng cách positive vs negative

- A1 round 1, prompt positive − A1 round 1, prompt negative: -6.0 điểm % (cặp: 8 chỉ positive / 14 chỉ negative, McNemar p 0.29).
- A1-emb round 1, nhãn positive − A1-emb round 1, nhãn negative: -6.0 điểm % (cặp: 8 chỉ positive / 14 chỉ negative, McNemar p 0.29).

## Kiểu thất bại trên task đã train (nhãn chính, chỉ episode thất bại) và số sự kiện

Cột A–O: số episode thất bại theo nhãn chính (`src/failure_labels.py`); n_drop/n_slip/n_collision: số sự kiện (`src/reward_r2.py`) trên mọi episode.

| model | task | A gắp hỏng | B kẹt chuyển subtask | C nhầm vật | D làm hỏng tiến độ | E đứng yên | O khác | n_drop | n_slip | n_collision |
|---|---|---|---|---|---|---|---|---|---|---|
| a1_r1_neg | 0 | 25 | 0 | 11 | 0 | 1 | 0 | 2 | 24 | 21 |
| a1_r1_neg | 8 | 7 | 3 | 0 | 5 | 11 | 0 | 1 | 4 | 0 |
| a1_r1_none | 0 | 21 | 1 | 5 | 0 | 0 | 0 | 5 | 32 | 12 |
| a1_r1_none | 8 | 15 | 5 | 0 | 2 | 8 | 0 | 4 | 11 | 0 |
| a1_r1_pos | 0 | 25 | 0 | 16 | 0 | 1 | 0 | 3 | 30 | 36 |
| a1_r1_pos | 8 | 11 | 0 | 0 | 3 | 13 | 0 | 1 | 3 | 0 |
| a1emb_r1_neg | 0 | 21 | 2 | 5 | 1 | 1 | 2 | 6 | 26 | 8 |
| a1emb_r1_neg | 8 | 12 | 2 | 0 | 2 | 8 | 0 | 3 | 7 | 0 |
| a1emb_r1_none | 0 | 17 | 1 | 9 | 0 | 1 | 0 | 5 | 27 | 14 |
| a1emb_r1_none | 8 | 9 | 3 | 0 | 1 | 15 | 0 | 1 | 6 | 0 |
| a1emb_r1_pos | 0 | 31 | 0 | 4 | 0 | 1 | 0 | 3 | 34 | 12 |
| a1emb_r1_pos | 8 | 13 | 3 | 0 | 1 | 9 | 0 | 2 | 4 | 0 |
| awr6k_r1_s2000 | 0 | 19 | 1 | 5 | 1 | 1 | 0 | 7 | 28 | 11 |
| awr6k_r1_s2000 | 8 | 12 | 3 | 0 | 4 | 7 | 0 | 1 | 9 | 0 |
| awr6k_r1_s4000 | 0 | 23 | 1 | 9 | 0 | 1 | 0 | 5 | 35 | 17 |
| awr6k_r1_s4000 | 8 | 13 | 2 | 0 | 1 | 9 | 0 | 0 | 7 | 0 |
| awr6k_r1_s6000 | 0 | 26 | 0 | 5 | 0 | 0 | 1 | 3 | 34 | 11 |
| awr6k_r1_s6000 | 8 | 14 | 2 | 0 | 4 | 7 | 0 | 0 | 12 | 0 |
| awr_r1 | 0 | 22 | 1 | 8 | 0 | 2 | 0 | 5 | 35 | 14 |
| awr_r1 | 8 | 10 | 4 | 0 | 1 | 9 | 0 | 1 | 2 | 0 |
| official | 0 | 25 | 0 | 3 | 0 | 0 | 1 | 0 | 43 | 12 |
| official | 8 | 10 | 6 | 0 | 1 | 10 | 0 | 2 | 13 | 0 |
