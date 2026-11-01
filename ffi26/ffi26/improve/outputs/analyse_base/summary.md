# Analyse: SmolVLA gốc, 10 task × 50 init × 3 seed (1500 episode)

Script: `docs/failures/tools/analyse_base.py`. Thiết lập: `lerobot/smolvla_libero`, NAS=10, expert fp32, seed 3000/4000/5000; task 0, 8 từ `outputs/fpo_v2/eval_s*/official`, task khác từ `outputs/analyse_base/eval_s*/official`. Nhãn: `label_episode_v2`; sự kiện: `detect_events`.

## 1. Tỉ lệ thành công theo task

| task | câu lệnh | thành công /150 | % (Wilson 95%) |
|---|---|---|---|
| 0 | put both the alphabet soup and the tomato sauce in the basket | 61 | 40.7 (33.1–48.7) |
| 1 | put both the cream cheese box and the butter in the basket | 124 | 82.7 (75.8–87.9) |
| 2 | turn on the stove and put the moka pot on it | 136 | 90.7 (84.9–94.4) |
| 3 | put the black bowl in the bottom drawer of the cabinet and close it | 146 | 97.3 (93.3–99.0) |
| 4 | put the white mug on the left plate and put the yellow and white mug on the right plate | 81 | 54.0 (46.0–61.8) |
| 5 | pick up the book and place it in the back compartment of the caddy | 133 | 88.7 (82.6–92.8) |
| 6 | put the white mug on the plate and put the chocolate pudding to the right of the plate | 78 | 52.0 (44.1–59.8) |
| 7 | put both the alphabet soup and the cream cheese box in the basket | 104 | 69.3 (61.5–76.2) |
| 8 | put both moka pots on the stove | 60 | 40.0 (32.5–48.0) |
| 9 | put the yellow and white mug in the microwave and close it | 113 | 75.3 (67.9–81.5) |
| tổng | | 1036/1500 | 69.1 (66.7–71.4) |

## 2. Thất bại: xong một phần hay chưa xong gì (cuối episode)

`subtask_completion` = tỉ lệ predicate goal (sai lúc đầu) đúng ở bước cuối.

| task | thất bại | xong một phần | chưa xong gì |
|---|---|---|---|
| 0 | 89 | 21 | 68 |
| 1 | 26 | 15 | 11 |
| 2 | 14 | 14 | 0 |
| 3 | 4 | 3 | 1 |
| 4 | 69 | 57 | 12 |
| 5 | 17 | 0 | 17 |
| 6 | 72 | 45 | 27 |
| 7 | 46 | 15 | 31 |
| 8 | 90 | 75 | 15 |
| 9 | 37 | 8 | 29 |
| tổng | 464 | 253 | 211 |

## 3. Lỗi thao tác (đa nhãn): số episode có nhãn, thất bại / thành công

| nhãn | task 0 | task 1 | task 2 | task 3 | task 4 | task 5 | task 6 | task 7 | task 8 | task 9 | tổng thất bại | tổng thành công |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A gắp hỏng | 73/0 | 13/0 | 7/0 | 0/0 | 6/0 | 0/0 | 38/0 | 25/0 | 25/0 | 6/0 | 193/464 | 0/1036 |
| C chạm/cầm vật ngoài goal | 33/0 | 12/8 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 1/0 | 0/0 | 0/0 | 46/464 | 8/1036 |
| F làm rơi | 6/0 | 10/3 | 1/0 | 1/1 | 3/0 | 3/0 | 2/0 | 13/1 | 3/2 | 9/0 | 51/464 | 7/1036 |
| G đặt gần đích nhưng chưa đúng | 0/0 | 1/0 | 0/0 | 0/0 | 36/0 | 0/0 | 28/5 | 1/0 | 4/0 | 6/7 | 76/464 | 12/1036 |
| D làm mất tiến độ | 0/0 | 5/0 | 0/0 | 1/0 | 5/1 | 0/0 | 5/0 | 0/0 | 5/0 | 4/3 | 25/464 | 4/1036 |
| slip (trượt khỏi kẹp) | 58/7 | 12/31 | 8/1 | 1/1 | 20/1 | 12/0 | 32/12 | 30/25 | 17/6 | 13/0 | 203/464 | 84/1036 |
| drop | 6/0 | 10/3 | 1/0 | 1/1 | 3/0 | 3/0 | 2/0 | 13/1 | 3/2 | 9/0 | 51/464 | 7/1036 |
| collision | 32/0 | 12/8 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 1/0 | 0/0 | 0/0 | 45/464 | 8/1036 |
| có ≥1 nhãn A/C/F/G/D | 84/0 | 26/11 | 8/0 | 1/1 | 42/1 | 3/0 | 57/5 | 36/1 | 33/2 | 20/7 | 310/464 | 28/1036 |

Ô task: thất bại có nhãn / thành công có nhãn.

## 4. Nhãn chính (một nhãn mỗi thất bại, thứ tự ưu tiên A > C > F > G > D > E_contact > E_free > B > O)

| task | A | B | C | D | E_contact | E_free | F | G | O |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 73 | 0 | 8 | 0 | 4 | 0 | 3 | 0 | 1 |
| 1 | 13 | 0 | 7 | 1 | 0 | 0 | 5 | 0 | 0 |
| 2 | 7 | 2 | 0 | 0 | 4 | 0 | 1 | 0 | 0 |
| 3 | 0 | 0 | 0 | 0 | 3 | 0 | 1 | 0 | 0 |
| 4 | 6 | 3 | 0 | 0 | 20 | 3 | 3 | 33 | 1 |
| 5 | 0 | 0 | 0 | 0 | 11 | 0 | 3 | 0 | 3 |
| 6 | 38 | 0 | 0 | 2 | 12 | 0 | 2 | 15 | 3 |
| 7 | 25 | 1 | 1 | 0 | 9 | 0 | 10 | 0 | 0 |
| 8 | 25 | 23 | 0 | 1 | 31 | 3 | 3 | 4 | 0 |
| 9 | 6 | 0 | 0 | 0 | 14 | 3 | 9 | 5 | 0 |

## 5. Độ ổn định theo cảnh (500 cảnh × 3 seed)

- thành công cả 3 seed: 269; thất bại cả 3: 88; lẫn lộn: 143
