# Failure labeler v2 trên eval FPO v3, so với official và v2

Script `improve/scripts/failure_labels_v3.py` (CPU). Labeler `src/failure_labels.label_episode_v2`, tâm vùng fixture lấy từ `outputs/failure_labels/region_centres.csv` (đã commit). Kiểm tra: nhãn official và v2 tính lại trùng 100 % với `outputs/failure_labels/labels.csv` (920 episode, mọi cột nhãn). Đây là nhãn tự động, chưa xem video để xác nhận. Task 0, 8: 150 episode/arm (50 init × seed 3000/4000/5000); task khác: 20 episode/arm (seed 1000). "k ấm" = `n_pred_new_ever_true` (số ấm từng lên bếp), như bảng WORKLOG 01/11 08:05.

## Task 8: thất bại theo số ấm đã lên bếp

| arm | thành công | thất bại | 0 ấm | 1 ấm | 2 ấm (D) | lab_A | lab_E_contact | E_contact ∧ B | nhãn chính A | nhãn chính E_contact |
|---|---|---|---|---|---|---|---|---|---|---|
| official | 60/150 | 90 | 14 | 73 | 3 | 25 (28%) | 55 (61%) | 39 | 25 | 31 |
| v2 θ4 | 76/150 | 74 | 50 | 22 | 2 | 42 (57%) | 67 (91%) | 20 | 42 | 24 |
| v3 θ4 | 88/150 | 62 | 13 | 46 | 3 | 25 (40%) | 52 (84%) | 39 | 25 | 25 |

## Task 0: E_free

| arm | thành công | thất bại | thất bại có E_free | nhãn chính E_free | E_free ở thành công | bước bắt đầu đoạn free (trung vị) | lab_A | nhãn chính A |
|---|---|---|---|---|---|---|---|---|
| official | 61/150 | 89 | 14 | 0 | 0 | 353 | 73 | 73 |
| v2 θ4 | 50/150 | 100 | 51 | 0 | 0 | 336 | 79 | 79 |
| v3 θ4 | 46/150 | 104 | 18 | 0 | 0 | 363 | 75 | 75 |

## Nhãn chính của thất bại, theo arm × task

| arm | task | n_ep | n_fail | A | C | F | G | D | E_contact | E_free | B | O |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v2 θ4 | 0 | 150 | 100 | 79 | 16 | 2 | 0 | 0 | 3 | 0 | 0 | 0 |
| v2 θ4 | 1 | 20 | 8 | 6 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 |
| v2 θ4 | 2 | 20 | 9 | 8 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| v2 θ4 | 3 | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| v2 θ4 | 4 | 20 | 8 | 1 | 0 | 0 | 3 | 0 | 2 | 1 | 0 | 1 |
| v2 θ4 | 5 | 20 | 3 | 0 | 0 | 0 | 2 | 0 | 1 | 0 | 0 | 0 |
| v2 θ4 | 6 | 20 | 11 | 9 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 |
| v2 θ4 | 7 | 20 | 10 | 8 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 |
| v2 θ4 | 8 | 150 | 74 | 42 | 0 | 2 | 2 | 1 | 24 | 2 | 0 | 1 |
| v2 θ4 | 9 | 20 | 3 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 0 |
| v3 θ4 | 0 | 150 | 104 | 75 | 19 | 8 | 0 | 0 | 2 | 0 | 0 | 0 |
| v3 θ4 | 1 | 20 | 4 | 3 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| v3 θ4 | 2 | 20 | 4 | 2 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 |
| v3 θ4 | 3 | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| v3 θ4 | 4 | 20 | 11 | 2 | 0 | 1 | 6 | 0 | 1 | 0 | 0 | 1 |
| v3 θ4 | 5 | 20 | 3 | 0 | 0 | 1 | 0 | 0 | 2 | 0 | 0 | 0 |
| v3 θ4 | 6 | 20 | 10 | 6 | 0 | 1 | 1 | 0 | 1 | 0 | 0 | 1 |
| v3 θ4 | 7 | 20 | 8 | 4 | 0 | 2 | 0 | 0 | 1 | 0 | 1 | 0 |
| v3 θ4 | 8 | 150 | 62 | 25 | 0 | 4 | 4 | 1 | 25 | 0 | 3 | 0 |
| v3 θ4 | 9 | 20 | 5 | 1 | 0 | 1 | 0 | 1 | 2 | 0 | 0 | 0 |
| official | 0 | 150 | 89 | 73 | 8 | 3 | 0 | 0 | 4 | 0 | 0 | 1 |
| official | 1 | 20 | 3 | 2 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| official | 2 | 20 | 4 | 2 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 |
| official | 3 | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| official | 4 | 20 | 7 | 2 | 0 | 0 | 3 | 0 | 2 | 0 | 0 | 0 |
| official | 5 | 20 | 2 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 |
| official | 6 | 20 | 11 | 5 | 0 | 0 | 3 | 1 | 1 | 0 | 0 | 1 |
| official | 7 | 20 | 4 | 2 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 |
| official | 8 | 150 | 90 | 25 | 0 | 3 | 4 | 1 | 31 | 3 | 23 | 0 |
| official | 9 | 20 | 2 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |

## Nhãn đa nhãn (số thất bại có nhãn), theo arm × task

| arm | task | n_fail | A | C | F | G | D | E_contact | E_free | B |
|---|---|---|---|---|---|---|---|---|---|---|
| v2 θ4 | 0 | 100 | 79 | 43 | 4 | 0 | 0 | 35 | 51 | 14 |
| v2 θ4 | 1 | 8 | 6 | 1 | 1 | 0 | 0 | 4 | 2 | 3 |
| v2 θ4 | 2 | 9 | 8 | 0 | 0 | 0 | 0 | 2 | 1 | 9 |
| v2 θ4 | 4 | 8 | 1 | 0 | 0 | 3 | 0 | 4 | 4 | 5 |
| v2 θ4 | 5 | 3 | 0 | 0 | 0 | 2 | 0 | 3 | 0 | 0 |
| v2 θ4 | 6 | 11 | 9 | 0 | 2 | 4 | 1 | 10 | 0 | 6 |
| v2 θ4 | 7 | 10 | 8 | 1 | 0 | 0 | 1 | 7 | 0 | 4 |
| v2 θ4 | 8 | 74 | 42 | 0 | 2 | 2 | 6 | 67 | 7 | 22 |
| v2 θ4 | 9 | 3 | 0 | 0 | 1 | 1 | 1 | 2 | 2 | 2 |
| v3 θ4 | 0 | 104 | 75 | 45 | 12 | 0 | 0 | 42 | 18 | 19 |
| v3 θ4 | 1 | 4 | 3 | 2 | 1 | 0 | 0 | 0 | 2 | 2 |
| v3 θ4 | 2 | 4 | 2 | 0 | 0 | 0 | 0 | 2 | 0 | 4 |
| v3 θ4 | 4 | 11 | 2 | 0 | 1 | 7 | 0 | 2 | 6 | 5 |
| v3 θ4 | 5 | 3 | 0 | 0 | 1 | 0 | 0 | 3 | 0 | 0 |
| v3 θ4 | 6 | 10 | 6 | 0 | 1 | 3 | 0 | 8 | 0 | 4 |
| v3 θ4 | 7 | 8 | 4 | 1 | 2 | 0 | 0 | 5 | 0 | 3 |
| v3 θ4 | 8 | 62 | 25 | 0 | 4 | 4 | 5 | 52 | 2 | 46 |
| v3 θ4 | 9 | 5 | 1 | 0 | 1 | 0 | 1 | 3 | 4 | 2 |
| official | 0 | 89 | 73 | 33 | 6 | 0 | 0 | 37 | 14 | 21 |
| official | 1 | 3 | 2 | 1 | 0 | 0 | 0 | 0 | 1 | 1 |
| official | 2 | 4 | 2 | 0 | 0 | 0 | 0 | 2 | 0 | 4 |
| official | 4 | 7 | 2 | 0 | 0 | 4 | 0 | 3 | 3 | 4 |
| official | 5 | 2 | 0 | 0 | 1 | 0 | 0 | 2 | 0 | 0 |
| official | 6 | 11 | 5 | 1 | 0 | 4 | 3 | 9 | 0 | 8 |
| official | 7 | 4 | 2 | 0 | 2 | 0 | 0 | 3 | 0 | 0 |
| official | 8 | 90 | 25 | 0 | 3 | 4 | 5 | 55 | 4 | 73 |
| official | 9 | 2 | 0 | 0 | 1 | 1 | 0 | 1 | 1 | 0 |

## Task 4 (seed 1000), từng init: S = thành công, nếu không là nhãn chính (cột "nhãn": mọi nhãn)

| init | official | official nhãn | v2 θ4 | v2 θ4 nhãn | v3 θ4 | v3 θ4 nhãn |
|---|---|---|---|---|---|---|
| 0 | S |  | S |  | S |  |
| 1 | A | A∧G | A | A∧E_contact∧B | G | G |
| 2 | E_contact | E_contact | O |  | F | F |
| 3 | S |  | G | G∧E_free | G | G∧E_free |
| 4 | S |  | S |  | S |  |
| 5 | S |  | S |  | S |  |
| 6 | E_contact | E_contact∧B | G | G∧E_contact∧E_free∧B | E_contact | E_contact∧B |
| 7 | S |  | S |  | S |  |
| 8 | S |  | S |  | O |  |
| 9 | S |  | E_contact | E_contact∧B | S |  |
| 10 | S |  | S |  | S |  |
| 11 | G | G∧E_free∧B | E_contact | E_contact | A | A∧G∧E_free |
| 12 | S |  | S |  | G | G∧E_free∧B |
| 13 | G | G∧E_free∧B | S |  | S |  |
| 14 | S |  | S |  | S |  |
| 15 | S |  | S |  | G | G∧E_free∧B |
| 16 | S |  | S |  | G | G∧E_free∧B |
| 17 | A | A∧E_free∧B | E_free | E_free∧B | A | A∧E_contact∧B |
| 18 | G | G∧E_contact | S |  | G | G∧E_free |
| 19 | S |  | G | G∧E_free∧B | S |  |

Task 4: chỉ official thành công 5, chỉ v3 thành công 1 (McNemar p = 0.219).
