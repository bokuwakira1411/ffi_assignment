# D_nas1: libero_10, n_action_steps = 1, relative, seed 1000, 50 episode/task

Cấu hình (`cmd.txt`): giống B_main nhưng 50 episode mỗi task, chỉ seed 1000. Chỉ có video cho episode 0–9 của mỗi task
(`lerobot_eval.py:823`), nên bảng nguyên nhân chỉ gồm 44 trên tổng 227 thất bại.

Cảnh ban đầu của episode 0–9 trùng với B_main, nên D_nas1 là một lần lặp nữa của B_main với seed 1000. Các nhãn ở đây
được gán ngày 31/10/2026, chỉ dựa trên khung điển hình (chưa xem timeline).

## Nhận xét

1. Phân bố giống B_main: **F3 (14/44)**, **F2 (13/44)** và **F6 (12/44)** là ba nguyên nhân lớn nhất.
2. Các kiểu theo task lặp lại B_main: task 1 cả 4 thất bại là F6 (cầm hộp cream cheese đi ra khỏi khung, giống B_main
   nhóm A); task 6 tay ôm cốc trên đĩa (3 episode); task 9 cầm cốc trước cửa lò (3/3); task 8 kẹp ấm nằm ngang (3 episode).
3. **Không có F1** (gắp hỏng rồi làm tiếp) trong 44 thất bại có video, khác với D_nas50 (16/48, xem `D_nas50.md`).
4. Hai trường hợp nhầm đích (F5): task 0 `e4` nhấc cả giỏ (giống B_main `t0 e4 s1000`), task 4 `e4` đặt cả hai cốc
   lên cùng một đĩa.

## Tỉ lệ thành công

Nguồn: `per_task[0].n_success` / `n_episodes` trong `eval_info.json` của từng job; seed: 1000.

| Task | Thành công | Tỉ lệ |
|---|---|---|
| 0 | 12/50 | 24% |
| 1 | 29/50 | 58% |
| 2 | 45/50 | 90% |
| 3 | 46/50 | 92% |
| 4 | 20/50 | 40% |
| 5 | 43/50 | 86% |
| 6 | 18/50 | 36% |
| 7 | 23/50 | 46% |
| 8 | 7/50 | 14% |
| 9 | 30/50 | 60% |
| **Tổng** | **273/500** | **54.6%** |

## Nguyên nhân thất bại theo task (chỉ các episode có video)

Mỗi episode có đúng một nhãn nguyên nhân chính (định nghĩa ở `docs/failure_overview.md`). Giai đoạn: 0 = chưa xong bước con thứ nhất; 1 = xong bước con thứ nhất, hỏng ở bước sau; ? = không xác định.

| Task | Thất bại có video | F1 | F2 | F3 | F4 | F5 | F6 | F0 | GĐ 0 | GĐ 1 | GĐ ? |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 9 |  | 3 | 4 |  | 1 | 1 |  | 7 | 2 |  |
| 1 | 4 |  |  |  |  |  | 4 |  |  |  | 4 |
| 2 | 1 |  |  |  |  |  | 1 |  |  | 1 |  |
| 4 | 7 |  | 2 |  |  | 1 | 4 |  | 1 | 1 | 5 |
| 6 | 7 |  | 2 | 5 |  |  |  |  | 2 | 5 |  |
| 7 | 5 |  | 2 |  | 2 |  | 1 |  | 2 | 3 |  |
| 8 | 8 |  | 4 | 2 |  |  | 1 | 1 | 3 | 4 | 1 |
| 9 | 3 |  |  | 3 |  |  |  |  | 3 |  |  |
| **Tổng** | **44** | **0** | **13** | **14** | **2** | **2** | **12** | **1** | **18** | **16** | **10** |

Mức bằng chứng: khung điển hình 44.

## Danh sách từng episode thất bại

Ảnh: dải 4 khung (bước 10 | 1/3 episode | khung điển hình | khung cuối), sinh bởi `docs/failures/tools/extract_frames.py`. Chuyển động = trung bình 100 bước cuối.

### Task 0

| Episode | Seed | Giai đoạn | Nguyên nhân | Bằng chứng | Chuyển động | Ghi chú | Ảnh |
|---|---|---|---|---|---|---|---|
| e0 | 1000 | 0 | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 0.07 | giỏ trống, lơ lửng trên vùng lon | [ảnh](frames/D_nas1/t0_e0_s1000.jpg) |
| e1 | 1000 | 1 | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 0.11 | một lon trong giỏ, lơ lửng | [ảnh](frames/D_nas1/t0_e1_s1000.jpg) |
| e2 | 1000 | 0 | F2 làm đổ / kẹp sai tư thế | khung điển hình | 0.79 | giỏ trống, các hộp bên phải bị xô đổ | [ảnh](frames/D_nas1/t0_e2_s1000.jpg) |
| e3 | 1000 | 0 | F2 làm đổ / kẹp sai tư thế | khung điển hình | 0.56 | lon xanh bị đổ | [ảnh](frames/D_nas1/t0_e3_s1000.jpg) |
| e4 | 1000 | 0 | F5 nhầm vật / nhầm đích | khung điển hình | 1.58 | nhấc cả giỏ | [ảnh](frames/D_nas1/t0_e4_s1000.jpg) |
| e5 | 1000 | 0 | F6 ra khỏi vùng làm việc | khung điển hình | 0.19 | giỏ trống, tay ra mép phải khung | [ảnh](frames/D_nas1/t0_e5_s1000.jpg) |
| e6 | 1000 | 0 | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 0.03 | giỏ trống, đứng yên ở mép phải | [ảnh](frames/D_nas1/t0_e6_s1000.jpg) |
| e8 | 1000 | 0 | F2 làm đổ / kẹp sai tư thế | khung điển hình | 0.21 | giỏ trống, hộp nước cam bị đổ | [ảnh](frames/D_nas1/t0_e8_s1000.jpg) |
| e9 | 1000 | 1 | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 1.85 | một lon trong giỏ, lơ lửng | [ảnh](frames/D_nas1/t0_e9_s1000.jpg) |

### Task 1

| Episode | Seed | Giai đoạn | Nguyên nhân | Bằng chứng | Chuyển động | Ghi chú | Ảnh |
|---|---|---|---|---|---|---|---|
| e2 | 1000 | ? | F6 ra khỏi vùng làm việc | khung điển hình | 0.10 | cầm hộp xanh đi về góc trên trái, ra khỏi khung; butter còn trên bàn (giống B_main nhóm A) | [ảnh](frames/D_nas1/t1_e2_s1000.jpg) |
| e3 | 1000 | ? | F6 ra khỏi vùng làm việc | khung điển hình | 0.17 | cầm hộp xanh đi về góc trên trái, ra khỏi khung; butter còn trên bàn (giống B_main nhóm A) | [ảnh](frames/D_nas1/t1_e3_s1000.jpg) |
| e5 | 1000 | ? | F6 ra khỏi vùng làm việc | khung điển hình | 0.02 | cầm hộp xanh đi về góc trên trái, ra khỏi khung; butter còn trên bàn (giống B_main nhóm A) | [ảnh](frames/D_nas1/t1_e5_s1000.jpg) |
| e8 | 1000 | ? | F6 ra khỏi vùng làm việc | khung điển hình | 0.15 | cầm hộp xanh đi về góc trên trái, ra khỏi khung; butter còn trên bàn (giống B_main nhóm A) | [ảnh](frames/D_nas1/t1_e8_s1000.jpg) |

### Task 2

| Episode | Seed | Giai đoạn | Nguyên nhân | Bằng chứng | Chuyển động | Ghi chú | Ảnh |
|---|---|---|---|---|---|---|---|
| e1 | 1000 | 1 | F6 ra khỏi vùng làm việc | khung điển hình | 1.47 | bếp đã bật, ấm trên bàn, tay ra khỏi khung | [ảnh](frames/D_nas1/t2_e1_s1000.jpg) |

### Task 4

| Episode | Seed | Giai đoạn | Nguyên nhân | Bằng chứng | Chuyển động | Ghi chú | Ảnh |
|---|---|---|---|---|---|---|---|
| e0 | 1000 | 0 | F2 làm đổ / kẹp sai tư thế | khung điển hình | 0.21 | cốc trắng bị úp đổ giữa bàn | [ảnh](frames/D_nas1/t4_e0_s1000.jpg) |
| e2 | 1000 | ? | F6 ra khỏi vùng làm việc | khung điển hình | 0.22 | cốc vàng-trắng chưa đụng tới; tay ra mép phải khung | [ảnh](frames/D_nas1/t4_e2_s1000.jpg) |
| e3 | 1000 | ? | F6 ra khỏi vùng làm việc | khung điển hình | 0.45 | cốc vàng-trắng chưa đụng tới; tay ra mép phải khung | [ảnh](frames/D_nas1/t4_e3_s1000.jpg) |
| e4 | 1000 | ? | F5 nhầm vật / nhầm đích | khung điển hình | 0.01 | cả hai cốc đặt cùng đĩa trái ảnh | [ảnh](frames/D_nas1/t4_e4_s1000.jpg) |
| e6 | 1000 | ? | F6 ra khỏi vùng làm việc | khung điển hình | 0.47 | cốc vàng-trắng chưa đụng tới; tay ra mép phải khung | [ảnh](frames/D_nas1/t4_e6_s1000.jpg) |
| e7 | 1000 | 1 | F2 làm đổ / kẹp sai tư thế | khung điển hình | 0.17 | cốc trắng ở đĩa phải; cốc vàng-trắng bị nghiêng ở mép đĩa trái | [ảnh](frames/D_nas1/t4_e7_s1000.jpg) |
| e9 | 1000 | ? | F6 ra khỏi vùng làm việc | khung điển hình | 0.15 | cốc vàng-trắng chưa đụng tới; tay ở mép phải khung | [ảnh](frames/D_nas1/t4_e9_s1000.jpg) |

### Task 6

| Episode | Seed | Giai đoạn | Nguyên nhân | Bằng chứng | Chuyển động | Ghi chú | Ảnh |
|---|---|---|---|---|---|---|---|
| e0 | 1000 | 1 | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 0.06 | cốc trên đĩa; lơ lửng ở pudding | [ảnh](frames/D_nas1/t6_e0_s1000.jpg) |
| e1 | 1000 | 1 | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 0.53 | cốc trên đĩa; lơ lửng ở pudding | [ảnh](frames/D_nas1/t6_e1_s1000.jpg) |
| e2 | 1000 | 0 | F2 làm đổ / kẹp sai tư thế | khung điển hình | 0.09 | cốc bị cầm nghiêng trên đĩa | [ảnh](frames/D_nas1/t6_e2_s1000.jpg) |
| e3 | 1000 | 1 | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 0.26 | tay vẫn ôm cốc trên đĩa | [ảnh](frames/D_nas1/t6_e3_s1000.jpg) |
| e4 | 1000 | 1 | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 0.16 | tay vẫn ôm cốc trên đĩa | [ảnh](frames/D_nas1/t6_e4_s1000.jpg) |
| e5 | 1000 | 1 | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 0.22 | tay vẫn ôm cốc trên đĩa | [ảnh](frames/D_nas1/t6_e5_s1000.jpg) |
| e8 | 1000 | 0 | F2 làm đổ / kẹp sai tư thế | khung điển hình | 0.24 | cốc đỏ (vật nhiễu) bị đổ | [ảnh](frames/D_nas1/t6_e8_s1000.jpg) |

### Task 7

| Episode | Seed | Giai đoạn | Nguyên nhân | Bằng chứng | Chuyển động | Ghi chú | Ảnh |
|---|---|---|---|---|---|---|---|
| e1 | 1000 | 1 | F6 ra khỏi vùng làm việc | khung điển hình | 0.39 | lon xanh đã rời bàn; tay ra khỏi khung; hộp cream cheese còn trên bàn | [ảnh](frames/D_nas1/t7_e1_s1000.jpg) |
| e3 | 1000 | 0 | F2 làm đổ / kẹp sai tư thế | khung điển hình | 0.57 | lon xanh bị đổ | [ảnh](frames/D_nas1/t7_e3_s1000.jpg) |
| e5 | 1000 | 0 | F2 làm đổ / kẹp sai tư thế | khung điển hình | 0.83 | lon xanh bị nghiêng, tay lơ lửng | [ảnh](frames/D_nas1/t7_e5_s1000.jpg) |
| e8 | 1000 | 1 | F4 đặt không đúng / gần đích | khung điển hình | 0.27 | hộp cream cheese ở mép giỏ | [ảnh](frames/D_nas1/t7_e8_s1000.jpg) |
| e9 | 1000 | 1 | F4 đặt không đúng / gần đích | khung điển hình | 0.29 | hộp cream cheese ở mép giỏ | [ảnh](frames/D_nas1/t7_e9_s1000.jpg) |

### Task 8

| Episode | Seed | Giai đoạn | Nguyên nhân | Bằng chứng | Chuyển động | Ghi chú | Ảnh |
|---|---|---|---|---|---|---|---|
| e0 | 1000 | ? | F0 không xác định | khung điển hình | 3.07 | tay ở bếp; vị trí ấm 2 không rõ | [ảnh](frames/D_nas1/t8_e0_s1000.jpg) |
| e1 | 1000 | 0 | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 0.05 | không ấm nào trên bếp; lơ lửng cạnh ấm | [ảnh](frames/D_nas1/t8_e1_s1000.jpg) |
| e2 | 1000 | 0 | F6 ra khỏi vùng làm việc | khung điển hình | 1.73 | tay ra mép phải khung | [ảnh](frames/D_nas1/t8_e2_s1000.jpg) |
| e3 | 1000 | 1 | F2 làm đổ / kẹp sai tư thế | khung điển hình | 0.77 | một ấm trên bếp; ấm 2 bị kẹp nằm ngang | [ảnh](frames/D_nas1/t8_e3_s1000.jpg) |
| e4 | 1000 | 1 | F2 làm đổ / kẹp sai tư thế | khung điển hình | 0.53 | một ấm trên bếp; ấm 2 bị kẹp nằm ngang | [ảnh](frames/D_nas1/t8_e4_s1000.jpg) |
| e5 | 1000 | 1 | F2 làm đổ / kẹp sai tư thế | khung điển hình | 0.53 | một ấm trên bếp; ấm 2 bị kẹp nằm ngang | [ảnh](frames/D_nas1/t8_e5_s1000.jpg) |
| e6 | 1000 | 1 | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 1.78 | một ấm trên bếp; ấm 2 trên bàn | [ảnh](frames/D_nas1/t8_e6_s1000.jpg) |
| e7 | 1000 | 0 | F2 làm đổ / kẹp sai tư thế | khung điển hình | 0.06 | ấm bị cầm nghiêng/đổ cạnh bếp | [ảnh](frames/D_nas1/t8_e7_s1000.jpg) |

### Task 9

| Episode | Seed | Giai đoạn | Nguyên nhân | Bằng chứng | Chuyển động | Ghi chú | Ảnh |
|---|---|---|---|---|---|---|---|
| e7 | 1000 | 0 | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 0.18 | cầm cốc trước cửa lò, chưa đưa vào | [ảnh](frames/D_nas1/t9_e7_s1000.jpg) |
| e8 | 1000 | 0 | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 0.20 | cầm cốc trước cửa lò, chưa đưa vào | [ảnh](frames/D_nas1/t9_e8_s1000.jpg) |
| e9 | 1000 | 0 | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 0.32 | cầm cốc trước cửa lò, chưa đưa vào | [ảnh](frames/D_nas1/t9_e9_s1000.jpg) |
