# Video: v2 θ4 thất bại ở task 1 và 7 (eval quên, seed 1000)

Config: `results/videos/v2_forget/configs/videos_v2_forget.yaml`; script `improve/scripts/phase3_eval_video.py` (replay chính xác, expert fp32, batch 8, 20 episode/task). Kiểm tra tất định: `replay_check_v2.csv`, 0/40 episode replay khác bản gốc `outputs/fpo_v2/eval_s1000/fpo_v2_sel`.

Vật của task 0 = vật mục tiêu của task 0 (`alphabet_soup_1`, `tomato_sauce_1`; bỏ `basket_1` vì dùng chung) mà KHÔNG phải vật mục tiêu của task đang chạy: task 1 (cream cheese + butter → basket): cả hai; task 7 (alphabet soup + cream cheese → basket): chỉ `tomato_sauce_1`.

Bộ phát hiện tự động (`src/reward.py`, tính từ log gốc, CHƯA xem video để xác nhận từng episode): chạm = grasp hoặc tiếp xúc gripper ở bất kỳ bước nào; nhấc = đang giữ và cao hơn vị trí reset ≥ 3 cm; dịch chuyển = đã chạm rồi bị dời > ngưỡng `MOVE_THRESHOLD`. Script: `improve/scripts/v2_forget_wrong_object.py`; bảng đầy đủ: `wrong_object.csv`.

## Tổng hợp (mọi episode, không chỉ thất bại)

| arm | task | episode | thành công | chạm vật task 0 | nhấc vật task 0 | dịch chuyển vật task 0 |
|---|---|---|---|---|---|---|
| official | 1 | 20 | 17/20 | 0/20 | 0/20 | 0/20 |
| official | 7 | 20 | 16/20 | 2/20 | 0/20 | 0/20 |
| v2 | 1 | 20 | 12/20 | 0/20 | 0/20 | 0/20 |
| v2 | 7 | 20 | 10/20 | 4/20 | 1/20 | 1/20 |

Chỉ episode thất bại:

| arm | task | thất bại | chạm vật task 0 | nhấc vật task 0 |
|---|---|---|---|---|
| official | 1 | 3 | 0/3 | 0/3 |
| official | 7 | 4 | 1/4 | 0/4 |
| v2 | 1 | 8 | 0/8 | 0/8 |
| v2 | 7 | 10 | 4/10 | 1/10 |

## Video (v2 θ4, episode thất bại)

| task | init | file | chạm vật task 0 | nhấc vật task 0 | vật đã nhấc (mọi vật) | official cùng episode |
|---|---|---|---|---|---|---|
| 1 | 0 | [v2_t1_init0_s1000_fail.mp4](v2_t1_init0_s1000_fail.mp4) | — | — | — | thành công |
| 1 | 3 | [v2_t1_init3_s1000_fail.mp4](v2_t1_init3_s1000_fail.mp4) | — | — | cream_cheese_1 | thành công |
| 1 | 4 | [v2_t1_init4_s1000_fail.mp4](v2_t1_init4_s1000_fail.mp4) | — | — | — | thất bại |
| 1 | 11 | [v2_t1_init11_s1000_fail.mp4](v2_t1_init11_s1000_fail.mp4) | — | — | cream_cheese_1 | thành công |
| 1 | 12 | [v2_t1_init12_s1000_fail.mp4](v2_t1_init12_s1000_fail.mp4) | — | — | — | thành công |
| 1 | 16 | [v2_t1_init16_s1000_fail.mp4](v2_t1_init16_s1000_fail.mp4) | — | — | cream_cheese_1 | thất bại |
| 1 | 18 | [v2_t1_init18_s1000_fail.mp4](v2_t1_init18_s1000_fail.mp4) | — | — | cream_cheese_1,butter_1,basket_1 | thành công |
| 1 | 19 | [v2_t1_init19_s1000_fail.mp4](v2_t1_init19_s1000_fail.mp4) | — | — | — | thất bại |
| 7 | 2 | [v2_t7_init2_s1000_fail.mp4](v2_t7_init2_s1000_fail.mp4) | — | — | alphabet_soup_1 | thành công |
| 7 | 3 | [v2_t7_init3_s1000_fail.mp4](v2_t7_init3_s1000_fail.mp4) | tomato_sauce_1 | — | — | thành công |
| 7 | 4 | [v2_t7_init4_s1000_fail.mp4](v2_t7_init4_s1000_fail.mp4) | — | — | cream_cheese_1 | thành công |
| 7 | 5 | [v2_t7_init5_s1000_fail.mp4](v2_t7_init5_s1000_fail.mp4) | tomato_sauce_1 | — | — | thất bại |
| 7 | 6 | [v2_t7_init6_s1000_fail.mp4](v2_t7_init6_s1000_fail.mp4) | tomato_sauce_1 | tomato_sauce_1 | tomato_sauce_1 | thành công |
| 7 | 7 | [v2_t7_init7_s1000_fail.mp4](v2_t7_init7_s1000_fail.mp4) | — | — | alphabet_soup_1 | thành công |
| 7 | 11 | [v2_t7_init11_s1000_fail.mp4](v2_t7_init11_s1000_fail.mp4) | — | — | — | thành công |
| 7 | 12 | [v2_t7_init12_s1000_fail.mp4](v2_t7_init12_s1000_fail.mp4) | tomato_sauce_1 | — | — | thất bại |
| 7 | 15 | [v2_t7_init15_s1000_fail.mp4](v2_t7_init15_s1000_fail.mp4) | — | — | — | thành công |
| 7 | 18 | [v2_t7_init18_s1000_fail.mp4](v2_t7_init18_s1000_fail.mp4) | — | — | alphabet_soup_1,basket_1 | thành công |
