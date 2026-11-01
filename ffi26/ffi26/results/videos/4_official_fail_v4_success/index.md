# SmolVLA gốc thất bại, FPO v4 thành công (cùng cảnh, cùng seed)

42 episode của eval cuối (task 0, 8 × 50 init × seed 3000/4000/5000) mà SmolVLA gốc thất bại còn FPO v4 thành công. Mỗi video replay lại episode gốc (cùng init, seed; kết quả khớp bản gốc ở cả 84 video) (`_meta/replay/replay_check_*.csv`).

Mở `sidebyside/`: trái là SmolVLA gốc, phải là FPO v4, cùng bước thời gian; camera agent, phóng to ×2. Episode kết thúc trước thì giữ khung cuối. Video đầy đủ cả camera cổ tay: `_meta/replay/*.mp4`.

Script: `report/scripts/make_sidebyside_v4wins.py`; config replay: `_meta/configs/videos.yaml`.

Replay không khớp từng bước với bản gốc (cùng kết quả, khác độ dài; khả năng do tính toán GPU không tất định):

- FPO v4, task 8, init 17, seed 5000: replay 390 bước, bản gốc 397 bước (cùng kết quả)

## Task 0 (12 episode)

| video | init | seed | số bước v4 | số bước gốc |
|---|---|---|---|---|
| `t0_init4_s3000.mp4` | 4 | 3000 | 366 | 520 |
| `t0_init4_s5000.mp4` | 4 | 5000 | 498 | 520 |
| `t0_init16_s5000.mp4` | 16 | 5000 | 328 | 520 |
| `t0_init17_s3000.mp4` | 17 | 3000 | 275 | 520 |
| `t0_init20_s4000.mp4` | 20 | 4000 | 255 | 520 |
| `t0_init20_s5000.mp4` | 20 | 5000 | 254 | 520 |
| `t0_init24_s4000.mp4` | 24 | 4000 | 248 | 520 |
| `t0_init29_s4000.mp4` | 29 | 4000 | 275 | 520 |
| `t0_init36_s4000.mp4` | 36 | 4000 | 447 | 520 |
| `t0_init36_s5000.mp4` | 36 | 5000 | 458 | 520 |
| `t0_init37_s5000.mp4` | 37 | 5000 | 307 | 520 |
| `t0_init38_s5000.mp4` | 38 | 5000 | 401 | 520 |

## Task 8 (30 episode)

| video | init | seed | số bước v4 | số bước gốc |
|---|---|---|---|---|
| `t8_init0_s3000.mp4` | 0 | 3000 | 391 | 520 |
| `t8_init3_s4000.mp4` | 3 | 4000 | 453 | 520 |
| `t8_init5_s5000.mp4` | 5 | 5000 | 384 | 520 |
| `t8_init8_s4000.mp4` | 8 | 4000 | 470 | 520 |
| `t8_init8_s5000.mp4` | 8 | 5000 | 399 | 520 |
| `t8_init9_s3000.mp4` | 9 | 3000 | 363 | 520 |
| `t8_init11_s5000.mp4` | 11 | 5000 | 371 | 520 |
| `t8_init12_s3000.mp4` | 12 | 3000 | 432 | 520 |
| `t8_init13_s3000.mp4` | 13 | 3000 | 422 | 520 |
| `t8_init13_s4000.mp4` | 13 | 4000 | 482 | 520 |
| `t8_init14_s3000.mp4` | 14 | 3000 | 379 | 520 |
| `t8_init17_s5000.mp4` | 17 | 5000 | 390 | 520 |
| `t8_init22_s3000.mp4` | 22 | 3000 | 385 | 520 |
| `t8_init22_s5000.mp4` | 22 | 5000 | 374 | 520 |
| `t8_init23_s4000.mp4` | 23 | 4000 | 510 | 520 |
| `t8_init24_s3000.mp4` | 24 | 3000 | 390 | 520 |
| `t8_init25_s5000.mp4` | 25 | 5000 | 371 | 520 |
| `t8_init26_s3000.mp4` | 26 | 3000 | 373 | 520 |
| `t8_init27_s5000.mp4` | 27 | 5000 | 361 | 520 |
| `t8_init30_s3000.mp4` | 30 | 3000 | 429 | 520 |
| `t8_init30_s5000.mp4` | 30 | 5000 | 384 | 520 |
| `t8_init36_s4000.mp4` | 36 | 4000 | 362 | 520 |
| `t8_init38_s3000.mp4` | 38 | 3000 | 364 | 520 |
| `t8_init38_s4000.mp4` | 38 | 4000 | 355 | 520 |
| `t8_init40_s3000.mp4` | 40 | 3000 | 427 | 520 |
| `t8_init40_s4000.mp4` | 40 | 4000 | 391 | 520 |
| `t8_init41_s3000.mp4` | 41 | 3000 | 436 | 520 |
| `t8_init41_s4000.mp4` | 41 | 4000 | 359 | 520 |
| `t8_init42_s3000.mp4` | 42 | 3000 | 441 | 520 |
| `t8_init45_s3000.mp4` | 45 | 3000 | 351 | 520 |
