# Video replay phần Improve

Config: `_meta/configs/videos_v2.yaml`. Script: `improve/scripts/phase3_eval_video.py`. Mỗi video: agent view | wrist view, 20 fps, header gồm nhánh, task, init state, seed eval (env seed = seed + init), bước, Φ, sự kiện reward_r2 (drop/slip/collision/timeout, hiện màu đỏ 2 s từ bước xảy ra).

## Kiểm tra tính tất định

- **official**: 112 episode chạy lại, 112 có bản gốc; **1 KHÁC bản gốc**.
- **v1t3**: 112 episode chạy lại, 112 có bản gốc; **1 KHÁC bản gốc**.
- **v2**: 112 episode chạy lại, không có bản gốc để so (episode mới).

⚠ Các episode khác bản gốc (success/length gốc → replay; `*` = episode có video):

| nhánh | seed | task | init | gốc | replay |
|---|---|---|---|---|---|
| official | 1000 | 8 | 13 | True/392 | True/437 |
| v1t3 | 1000 | 8 | 2 | True/344 | True/345 |

## Danh sách video

Ghi chú tự động từ nhãn `src/failure_labels.py` và sự kiện `src/reward_r2.py` (sự kiện@bước), chưa xem video.

| loại | seed | task | init | nhánh | file | kết quả | khớp bản gốc | ghi chú |
|---|---|---|---|---|---|---|---|---|
| official thất bại, θ3 thành công | 1000 | 0 | 0 | official | [official_t0_init0_s1000_fail.mp4](official/official_t0_init0_s1000_fail.mp4) | thất bại | có | thất bại, nhãn A: không gắp được vật mục tiêu; subtask 0.0; sự kiện: slip@71 alphabet_soup_1; collision@307 orange_juice_1; timeout@519 |
| official thất bại, θ3 thành công | 1000 | 0 | 0 | v1t3 | [v1t3_t0_init0_s1000_success.mp4](v1t3/v1t3_t0_init0_s1000_success.mp4) | thành công | có | thành công ở bước 268; subtask 1.0; sự kiện: không có |
| official thất bại, θ3 thành công | 1000 | 0 | 0 | v2 | [v2_t0_init0_s1000_fail.mp4](v2/v2_t0_init0_s1000_fail.mp4) | thất bại | không có bản gốc | thất bại, nhãn C: chạm/gắp nhầm vật khác; subtask 0.0; sự kiện: slip@130 alphabet_soup_1; collision@221 orange_juice_1; timeout@519 |
| official thất bại, θ3 thành công | 2000 | 0 | 14 | official | [official_t0_init14_s2000_fail.mp4](official/official_t0_init14_s2000_fail.mp4) | thất bại | có | thất bại, nhãn C: chạm/gắp nhầm vật khác; subtask 0.0; sự kiện: collision@102 orange_juice_1; collision@120 milk_1; timeout@519 |
| official thất bại, θ3 thành công | 2000 | 0 | 14 | v1t3 | [v1t3_t0_init14_s2000_success.mp4](v1t3/v1t3_t0_init14_s2000_success.mp4) | thành công | có | thành công ở bước 324; subtask 1.0; sự kiện: không có |
| official thất bại, θ3 thành công | 2000 | 0 | 14 | v2 | [v2_t0_init14_s2000_fail.mp4](v2/v2_t0_init14_s2000_fail.mp4) | thất bại | không có bản gốc | thất bại, nhãn A: không gắp được vật mục tiêu; subtask 0.0; sự kiện: collision@147 orange_juice_1; timeout@519 |
| official thất bại, θ3 thành công | 2000 | 8 | 1 | official | [official_t8_init1_s2000_fail.mp4](official/official_t8_init1_s2000_fail.mp4) | thất bại | có | thất bại, nhãn E: đứng yên (stall); subtask 0.5; sự kiện: timeout@519 |
| official thất bại, θ3 thành công | 2000 | 8 | 1 | v1t3 | [v1t3_t8_init1_s2000_success.mp4](v1t3/v1t3_t8_init1_s2000_success.mp4) | thành công | có | thành công ở bước 368; subtask 1.0; sự kiện: không có |
| official thất bại, θ3 thành công | 2000 | 8 | 1 | v2 | [v2_t8_init1_s2000_success.mp4](v2/v2_t8_init1_s2000_success.mp4) | thành công | không có bản gốc | thành công ở bước 418; subtask 1.0; sự kiện: không có |
| official thành công, θ3 thất bại | 1000 | 0 | 43 | official | [official_t0_init43_s1000_success.mp4](official/official_t0_init43_s1000_success.mp4) | thành công | có | thành công ở bước 432; subtask 1.0; sự kiện: slip@352 tomato_sauce_1 |
| official thành công, θ3 thất bại | 1000 | 0 | 43 | v1t3 | [v1t3_t0_init43_s1000_fail.mp4](v1t3/v1t3_t0_init43_s1000_fail.mp4) | thất bại | có | thất bại, nhãn C: chạm/gắp nhầm vật khác; subtask 0.0; sự kiện: slip@284 alphabet_soup_1; slip@487 alphabet_soup_1; timeout@519 |
| official thành công, θ3 thất bại | 1000 | 0 | 43 | v2 | [v2_t0_init43_s1000_success.mp4](v2/v2_t0_init43_s1000_success.mp4) | thành công | không có bản gốc | thành công ở bước 249; subtask 1.0; sự kiện: không có |
| official thành công, θ3 thất bại | 1000 | 8 | 29 | official | [official_t8_init29_s1000_success.mp4](official/official_t8_init29_s1000_success.mp4) | thành công | có | thành công ở bước 379; subtask 1.0; sự kiện: không có |
| official thành công, θ3 thất bại | 1000 | 8 | 29 | v1t3 | [v1t3_t8_init29_s1000_fail.mp4](v1t3/v1t3_t8_init29_s1000_fail.mp4) | thất bại | có | thất bại, nhãn E: đứng yên (stall); subtask 0.0; sự kiện: timeout@519 |
| official thành công, θ3 thất bại | 1000 | 8 | 29 | v2 | [v2_t8_init29_s1000_fail.mp4](v2/v2_t8_init29_s1000_fail.mp4) | thất bại | không có bản gốc | thất bại, nhãn A: không gắp được vật mục tiêu; subtask 0.0; sự kiện: slip@139 moka_pot_2; timeout@519 |
| official thành công, θ3 thất bại | 2000 | 0 | 20 | official | [official_t0_init20_s2000_success.mp4](official/official_t0_init20_s2000_success.mp4) | thành công | có | thành công ở bước 256; subtask 1.0; sự kiện: không có |
| official thành công, θ3 thất bại | 2000 | 0 | 20 | v1t3 | [v1t3_t0_init20_s2000_fail.mp4](v1t3/v1t3_t0_init20_s2000_fail.mp4) | thất bại | có | thất bại, nhãn A: không gắp được vật mục tiêu; subtask 0.0; sự kiện: slip@199 alphabet_soup_1; slip@251 alphabet_soup_1; slip@295 alphabet_soup_1; slip@340 alphabet_soup_1; slip@379 alphabet_soup_1; slip@422 alphabet_soup_1; timeout@519 |
| official thành công, θ3 thất bại | 2000 | 0 | 20 | v2 | [v2_t0_init20_s2000_success.mp4](v2/v2_t0_init20_s2000_success.mp4) | thành công | không có bản gốc | thành công ở bước 254; subtask 1.0; sự kiện: không có |
| cả hai thất bại | 1000 | 0 | 18 | official | [official_t0_init18_s1000_fail.mp4](official/official_t0_init18_s1000_fail.mp4) | thất bại | có | thất bại, nhãn C: chạm/gắp nhầm vật khác; subtask 0.5; sự kiện: collision@65 orange_juice_1; collision@107 milk_1; timeout@519 |
| cả hai thất bại | 1000 | 0 | 18 | v1t3 | [v1t3_t0_init18_s1000_fail.mp4](v1t3/v1t3_t0_init18_s1000_fail.mp4) | thất bại | có | thất bại, nhãn C: chạm/gắp nhầm vật khác; subtask 0.0; sự kiện: collision@65 orange_juice_1; collision@104 milk_1; timeout@519 |
| cả hai thất bại | 1000 | 0 | 18 | v2 | [v2_t0_init18_s1000_fail.mp4](v2/v2_t0_init18_s1000_fail.mp4) | thất bại | không có bản gốc | thất bại, nhãn C: chạm/gắp nhầm vật khác; subtask 0.0; sự kiện: collision@60 orange_juice_1; timeout@519 |
| cả hai thất bại | 2000 | 8 | 3 | official | [official_t8_init3_s2000_fail.mp4](official/official_t8_init3_s2000_fail.mp4) | thất bại | có | thất bại, nhãn E: đứng yên (stall); subtask 0.5; sự kiện: timeout@519 |
| cả hai thất bại | 2000 | 8 | 3 | v1t3 | [v1t3_t8_init3_s2000_fail.mp4](v1t3/v1t3_t8_init3_s2000_fail.mp4) | thất bại | có | thất bại, nhãn E: đứng yên (stall); subtask 0.5; sự kiện: timeout@519 |
| cả hai thất bại | 2000 | 8 | 3 | v2 | [v2_t8_init3_s2000_fail.mp4](v2/v2_t8_init3_s2000_fail.mp4) | thất bại | không có bản gốc | thất bại, nhãn E: đứng yên (stall); subtask 0.5; sự kiện: timeout@519 |
| cả hai thành công | 2000 | 8 | 0 | official | [official_t8_init0_s2000_success.mp4](official/official_t8_init0_s2000_success.mp4) | thành công | có | thành công ở bước 350; subtask 1.0; sự kiện: không có |
| cả hai thành công | 2000 | 8 | 0 | v1t3 | [v1t3_t8_init0_s2000_success.mp4](v1t3/v1t3_t8_init0_s2000_success.mp4) | thành công | có | thành công ở bước 344; subtask 1.0; sự kiện: không có |
| cả hai thành công | 2000 | 8 | 0 | v2 | [v2_t8_init0_s2000_success.mp4](v2/v2_t8_init0_s2000_success.mp4) | thành công | không có bản gốc | thành công ở bước 371; subtask 1.0; sự kiện: không có |
