# Nhãn thất bại v2 (`src/failure_labels.label_episode_v2`)

Sinh bởi `scripts/failure_labels_run.py`. Dữ liệu: `outputs/fpo_v2/eval_s{3000,4000,5000}` (task 0, 8 × 50 init) và `outputs/fpo_v2/eval_s1000` (8 task khác × 20 init), expert fp32 cho mọi arm. Task 0, 8 gộp 3 seed (150 episode / arm / task).
Tổng 1380 episode, 689 thất bại.

Thiếu dữ liệu: fpo_v3_sel: chưa có thư mục eval nào

## Phân bố nhãn chính (chỉ episode thất bại)

| arm | task | n_ep | n_fail | A | C | F | G | D | E_contact | E_free | B | O |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fpo_v1_it3 | 0 | 150 | 88 | 56 | 13 | 12 | 0 | 0 | 4 | 0 | 3 | 0 |
| fpo_v1_it3 | 1 | 20 | 2 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| fpo_v1_it3 | 2 | 20 | 7 | 6 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| fpo_v1_it3 | 3 | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| fpo_v1_it3 | 4 | 20 | 10 | 0 | 0 | 0 | 6 | 0 | 3 | 1 | 0 | 0 |
| fpo_v1_it3 | 5 | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| fpo_v1_it3 | 6 | 20 | 13 | 5 | 1 | 0 | 4 | 0 | 3 | 0 | 0 | 0 |
| fpo_v1_it3 | 7 | 20 | 13 | 8 | 0 | 1 | 0 | 0 | 2 | 0 | 2 | 0 |
| fpo_v1_it3 | 8 | 150 | 114 | 59 | 0 | 3 | 2 | 2 | 35 | 3 | 10 | 0 |
| fpo_v1_it3 | 9 | 20 | 4 | 0 | 0 | 0 | 1 | 0 | 2 | 0 | 0 | 1 |
| fpo_v2_sel | 0 | 150 | 100 | 79 | 16 | 2 | 0 | 0 | 3 | 0 | 0 | 0 |
| fpo_v2_sel | 1 | 20 | 8 | 6 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 |
| fpo_v2_sel | 2 | 20 | 9 | 8 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| fpo_v2_sel | 3 | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| fpo_v2_sel | 4 | 20 | 8 | 1 | 0 | 0 | 3 | 0 | 2 | 1 | 0 | 1 |
| fpo_v2_sel | 5 | 20 | 3 | 0 | 0 | 0 | 2 | 0 | 1 | 0 | 0 | 0 |
| fpo_v2_sel | 6 | 20 | 11 | 9 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 |
| fpo_v2_sel | 7 | 20 | 10 | 8 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 |
| fpo_v2_sel | 8 | 150 | 74 | 42 | 0 | 2 | 2 | 1 | 24 | 2 | 0 | 1 |
| fpo_v2_sel | 9 | 20 | 3 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 0 |
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

## Tần suất từng nhãn (đa nhãn, chỉ episode thất bại)

| arm | task | n_fail | A | C | F | G | D | E_contact | E_free | B |
|---|---|---|---|---|---|---|---|---|---|---|
| fpo_v1_it3 | 0 | 88 | 56 | 30 | 19 | 0 | 2 | 55 | 9 | 22 |
| fpo_v1_it3 | 1 | 2 | 1 | 1 | 0 | 0 | 0 | 0 | 1 | 1 |
| fpo_v1_it3 | 2 | 7 | 6 | 0 | 0 | 0 | 0 | 2 | 1 | 7 |
| fpo_v1_it3 | 4 | 10 | 0 | 0 | 0 | 6 | 0 | 6 | 5 | 8 |
| fpo_v1_it3 | 6 | 13 | 5 | 2 | 0 | 4 | 1 | 12 | 0 | 9 |
| fpo_v1_it3 | 7 | 13 | 8 | 1 | 1 | 0 | 0 | 8 | 2 | 6 |
| fpo_v1_it3 | 8 | 114 | 59 | 0 | 3 | 2 | 3 | 88 | 11 | 59 |
| fpo_v1_it3 | 9 | 4 | 0 | 0 | 0 | 1 | 0 | 2 | 2 | 1 |
| fpo_v2_sel | 0 | 100 | 79 | 43 | 4 | 0 | 0 | 35 | 51 | 14 |
| fpo_v2_sel | 1 | 8 | 6 | 1 | 1 | 0 | 0 | 4 | 2 | 3 |
| fpo_v2_sel | 2 | 9 | 8 | 0 | 0 | 0 | 0 | 2 | 1 | 9 |
| fpo_v2_sel | 4 | 8 | 1 | 0 | 0 | 3 | 0 | 4 | 4 | 5 |
| fpo_v2_sel | 5 | 3 | 0 | 0 | 0 | 2 | 0 | 3 | 0 | 0 |
| fpo_v2_sel | 6 | 11 | 9 | 0 | 2 | 4 | 1 | 10 | 0 | 6 |
| fpo_v2_sel | 7 | 10 | 8 | 1 | 0 | 0 | 1 | 7 | 0 | 4 |
| fpo_v2_sel | 8 | 74 | 42 | 0 | 2 | 2 | 6 | 67 | 7 | 22 |
| fpo_v2_sel | 9 | 3 | 0 | 0 | 1 | 1 | 1 | 2 | 2 | 2 |
| official | 0 | 89 | 73 | 33 | 6 | 0 | 0 | 37 | 14 | 21 |
| official | 1 | 3 | 2 | 1 | 0 | 0 | 0 | 0 | 1 | 1 |
| official | 2 | 4 | 2 | 0 | 0 | 0 | 0 | 2 | 0 | 4 |
| official | 4 | 7 | 2 | 0 | 0 | 4 | 0 | 3 | 3 | 4 |
| official | 5 | 2 | 0 | 0 | 1 | 0 | 0 | 2 | 0 | 0 |
| official | 6 | 11 | 5 | 1 | 0 | 4 | 3 | 9 | 0 | 8 |
| official | 7 | 4 | 2 | 0 | 2 | 0 | 0 | 3 | 0 | 0 |
| official | 8 | 90 | 25 | 0 | 3 | 4 | 5 | 55 | 4 | 73 |
| official | 9 | 2 | 0 | 0 | 1 | 1 | 0 | 1 | 1 | 0 |

## Tổ hợp nhãn hay gặp nhất (6 tổ hợp đầu mỗi arm)

| arm | combo | n | frac_of_arm_fail |
|---|---|---|---|
| fpo_v1_it3 | A∧E_contact | 55 | 22% |
| fpo_v1_it3 | E_contact∧B | 31 | 12% |
| fpo_v1_it3 | A∧E_contact∧B | 21 | 8% |
| fpo_v1_it3 | E_contact | 16 | 6% |
| fpo_v1_it3 | B | 15 | 6% |
| fpo_v1_it3 | A∧B | 14 | 6% |
| fpo_v2_sel | A∧E_contact | 36 | 16% |
| fpo_v2_sel | A∧E_free | 30 | 13% |
| fpo_v2_sel | A∧E_contact∧B | 22 | 10% |
| fpo_v2_sel | E_contact | 18 | 8% |
| fpo_v2_sel | A∧B | 17 | 8% |
| fpo_v2_sel | E_contact∧B | 11 | 5% |
| official | E_contact∧B | 31 | 15% |
| official | B | 24 | 11% |
| official | A∧E_contact∧B | 22 | 10% |
| official | A∧E_contact | 19 | 9% |
| official | A | 18 | 8% |
| official | A∧C | 12 | 6% |

### Theo task (mọi arm gộp, 3 tổ hợp đầu)

| task | combo | n |
|---|---|---|
| 0 | A∧E_free | 40 |
| 0 | A∧E_contact | 29 |
| 0 | A | 28 |
| 1 | A | 3 |
| 1 | A∧B | 2 |
| 1 | A∧E_contact | 2 |
| 2 | A∧B | 12 |
| 2 | A∧E_contact∧B | 3 |
| 2 | E_contact∧B | 2 |
| 4 | G∧E_free∧B | 6 |
| 4 | E_contact∧B | 4 |
| 4 | E_contact | 3 |
| 5 | E_contact | 2 |
| 5 | G∧E_contact | 2 |
| 5 | F∧E_contact | 1 |
| 6 | A∧E_contact∧B | 11 |
| 6 | E_contact | 4 |
| 6 | G∧E_contact∧B | 4 |
| 7 | A∧E_contact | 7 |
| 7 | E_contact | 5 |
| 7 | A∧E_contact∧B | 3 |
| 8 | A∧E_contact | 72 |
| 8 | E_contact∧B | 59 |
| 8 | B | 33 |
| 9 | (không nhãn) | 1 |
| 9 | E_contact | 1 |
| 9 | E_contact∧E_free | 1 |

## Nhãn cũ (hàng) so với nhãn mới (cột), episode thất bại

| old_primary | A | C | F | G | D | E_contact | E_free | B | O |
|---|---|---|---|---|---|---|---|---|---|
| A | 334 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| B | 4 | 0 | 5 | 0 | 0 | 0 | 0 | 37 | 0 |
| C | 14 | 40 | 6 | 0 | 0 | 6 | 0 | 0 | 0 |
| D | 1 | 0 | 2 | 18 | 5 | 10 | 1 | 2 | 0 |
| E | 43 | 0 | 17 | 14 | 0 | 109 | 11 | 0 | 0 |
| O | 3 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 5 |

## Task 1, 2, 7: nhầm vật (C), seed 1000, 20 init

`n_fail_C_any`: số episode thất bại có nhãn C (đa nhãn). McNemar chính xác theo cặp init so với official.

| task | arm | n_ep | n_fail | n_fail_C_any | n_primary_C | C_objs | only_arm_C | only_official_C | mcnemar_p_vs_official |
|---|---|---|---|---|---|---|---|---|---|
| 1 | official | 20 | 3 | 1 | 1 | milk_1 ×1 | 0 | 0 | 1.00 |
| 1 | fpo_v1_it3 | 20 | 2 | 1 | 1 | milk_1 ×1 | 0 | 0 | 1.00 |
| 1 | fpo_v2_sel | 20 | 8 | 1 | 0 | milk_1 ×1 | 0 | 0 | 1.00 |
| 2 | official | 20 | 4 | 0 | 0 |  | 0 | 0 | 1.00 |
| 2 | fpo_v1_it3 | 20 | 7 | 0 | 0 |  | 0 | 0 | 1.00 |
| 2 | fpo_v2_sel | 20 | 9 | 0 | 0 |  | 0 | 0 | 1.00 |
| 7 | official | 20 | 4 | 0 | 0 |  | 0 | 0 | 1.00 |
| 7 | fpo_v1_it3 | 20 | 13 | 1 | 0 | ketchup_1 ×1 | 1 | 0 | 1.00 |
| 7 | fpo_v2_sel | 20 | 10 | 1 | 0 | tomato_sauce_1 ×1 | 1 | 0 | 1.00 |

## Tâm vùng đích dạng fixture (dùng cho G)

Ước lượng từ log: vị trí xy của vật tại hàng cuối predicate True, trung bình theo init rồi theo các init. Vùng đích là vật di động (giỏ, đĩa) thì dùng vị trí vật đó trong log.

| task | region | object | x | y | std_across_inits_max | n_init | n_episodes |
|---|---|---|---|---|---|---|---|
| 2 | flat_stove_1_cook_region | moka_pot_1 | -0.045 | 0.201 | 0.022 | 18 | 40 |
| 3 | white_cabinet_1_bottom_region | akita_black_bowl_1 | -0.011 | 0.295 | 0.009 | 20 | 60 |
| 5 | desk_caddy_1_back_contain_region | black_book_1 | -0.428 | -0.155 | 0.019 | 20 | 55 |
| 6 | living_room_table_plate_right_region | chocolate_pudding_1 | 0.109 | 0.116 | 0.020 | 16 | 30 |
| 8 | flat_stove_1_cook_region | moka_pot_1 | -0.010 | -0.171 | 0.013 | 43 | 179 |
| 8 | flat_stove_1_cook_region | moka_pot_2 | -0.098 | -0.227 | 0.010 | 49 | 333 |
| 9 | microwave_1_heating_region | white_yellow_mug_1 | -0.009 | 0.335 | 0.016 | 20 | 52 |

## Kiểm chứng bằng tay

`manual_check.csv`: 20 episode thất bại, chọn ngẫu nhiên (seed 20261101) xoay vòng qua các nhóm (arm, nhãn chính). Cột `nhan_tay`, `nhan_script` để trống. Video có sẵn: 0/20; lệnh render: `render_commands.sh` (chưa chạy, cần GPU).
