# D_nas50: libero_10, n_action_steps = 50, relative, seed 1000, 50 episode/task

Cấu hình (`cmd.txt`): giống D_nas1, chỉ khác `--policy.n_action_steps=50` (giá trị trong config của checkpoint).
Chỉ có video cho episode 0–9 của mỗi task, nên bảng nguyên nhân chỉ gồm 48 trên tổng 261 thất bại.
Phân tích chi tiết, gồm 22 cặp cùng cảnh mà NAS=1 thành công: `failure_analysis.md` mục 6.5. Nhóm N1–N7 ở đó
được gán sang bộ nguyên nhân chung F0–F6 (cột "Ghi chú" giữ tên nhóm cũ).

## Nhận xét

1. **F1 gắp hỏng (16/48) là nguyên nhân lớn nhất**, so với 0/44 ở D_nas1. Gồm: gắp trượt trong chunk đầu rồi vẫn mang
   tay không tới đích (10 episode, N1); lượn trên cụm hộp mà không gắp được lon xanh bị che ở task 0 (4 episode,
   khung điển hình); không nhấc được butter (task 1 `e0`) hoặc ấm (task 2 `e2`).
2. **F4 đặt không đúng / gần đích (11/48)**, so với 2/44 ở D_nas1: ngăn kéo chưa đóng hẳn (task 3), hai cốc ở đĩa
   nhưng không được tính (task 4, 5 episode), sách sai ngăn (task 5), để cốc ngoài lò (task 9).
3. **F6 ra khỏi vùng làm việc = 0/48**, so với 12/44 ở D_nas1 và 27/142 ở B_main. Với NAS=50, robot không đi lạc khỏi
   vùng làm việc.
4. F3 đứng yên / lơ lửng (15/48) vẫn còn, chủ yếu ở task 6 và 8 sau khi xong bước con thứ nhất.

## Tỉ lệ thành công

Nguồn: `per_task[0].n_success` / `n_episodes` trong `eval_info.json` của từng job; seed: 1000.

| Task | Thành công | Tỉ lệ |
|---|---|---|
| 0 | 12/50 | 24% |
| 1 | 21/50 | 42% |
| 2 | 37/50 | 74% |
| 3 | 40/50 | 80% |
| 4 | 17/50 | 34% |
| 5 | 37/50 | 74% |
| 6 | 11/50 | 22% |
| 7 | 18/50 | 36% |
| 8 | 19/50 | 38% |
| 9 | 27/50 | 54% |
| **Tổng** | **239/500** | **47.8%** |

## Nguyên nhân thất bại theo task (chỉ các episode có video)

Mỗi episode có đúng một nhãn nguyên nhân chính (định nghĩa ở `docs/failure_overview.md`). Giai đoạn: 0 = chưa xong bước con thứ nhất; 1 = xong bước con thứ nhất, hỏng ở bước sau; ? = không xác định.

| Task | Thất bại có video | F1 | F2 | F3 | F4 | F5 | F6 | F0 | GĐ 0 | GĐ 1 | GĐ ? |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 8 | 5 | 1 | 2 |  |  |  |  | 4 | 4 |  |
| 1 | 5 | 4 |  |  |  |  |  | 1 | 4 | 1 |  |
| 2 | 3 | 1 | 1 | 1 |  |  |  |  |  | 3 |  |
| 3 | 2 |  |  |  | 2 |  |  |  |  | 2 |  |
| 4 | 6 |  |  | 1 | 5 |  |  |  | 1 | 5 |  |
| 5 | 2 |  |  |  | 2 |  |  |  | 2 |  |  |
| 6 | 7 | 2 |  | 5 |  |  |  |  | 2 | 5 |  |
| 7 | 7 | 4 |  | 1 |  | 2 |  |  | 6 | 1 |  |
| 8 | 6 |  | 1 | 5 |  |  |  |  |  | 6 |  |
| 9 | 2 |  |  |  | 2 |  |  |  | 2 |  |  |
| **Tổng** | **48** | **16** | **3** | **15** | **11** | **2** | **0** | **1** | **21** | **27** | **0** |

Mức bằng chứng: khung điển hình 26, timeline 22.

## Danh sách từng episode thất bại

Ảnh: dải 4 khung (bước 10 | 1/3 episode | khung điển hình | khung cuối), sinh bởi `docs/failures/tools/extract_frames.py`. Chuyển động = trung bình 100 bước cuối.

### Task 0

| Episode | Seed | Giai đoạn | Nguyên nhân | Bằng chứng | Chuyển động | Ghi chú | Ảnh |
|---|---|---|---|---|---|---|---|
| e0 | 1000 | 0 | F1 gắp hỏng | khung điển hình | 2.67 | U→: lượn trên cụm hộp bên phải, không gắp được lon xanh (bị hộp sữa che) | [ảnh](frames/D_nas50/t0_e0_s1000.jpg) |
| e2 | 1000 | 1 | F2 làm đổ / kẹp sai tư thế | khung điển hình | 0.99 | N3: một lon trong giỏ, một lon đổ cạnh giỏ | [ảnh](frames/D_nas50/t0_e2_s1000.jpg) |
| e3 | 1000 | 0 | F1 gắp hỏng | khung điển hình | 1.59 | U→: lượn trên cụm hộp bên phải, không gắp được lon xanh (bị hộp sữa che) | [ảnh](frames/D_nas50/t0_e3_s1000.jpg) |
| e4 | 1000 | 1 | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 1.60 | N3: một lon trong giỏ | [ảnh](frames/D_nas50/t0_e4_s1000.jpg) |
| e6 | 1000 | 1 | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 1.35 | N3: một lon trong giỏ | [ảnh](frames/D_nas50/t0_e6_s1000.jpg) |
| e7 | 1000 | 1 | F1 gắp hỏng | timeline | 0.80 | N1: trượt lon alphabet soup, cho lon khác vào giỏ | [ảnh](frames/D_nas50/t0_e7_s1000.jpg) |
| e8 | 1000 | 0 | F1 gắp hỏng | khung điển hình | 2.27 | U→: lượn trên cụm hộp bên phải, không gắp được lon xanh (bị hộp sữa che) | [ảnh](frames/D_nas50/t0_e8_s1000.jpg) |
| e9 | 1000 | 0 | F1 gắp hỏng | khung điển hình | 2.15 | U→: lượn trên cụm hộp bên phải, không gắp được lon xanh (bị hộp sữa che) | [ảnh](frames/D_nas50/t0_e9_s1000.jpg) |

### Task 1

| Episode | Seed | Giai đoạn | Nguyên nhân | Bằng chứng | Chuyển động | Ghi chú | Ảnh |
|---|---|---|---|---|---|---|---|
| e0 | 1000 | 1 | F1 gắp hỏng | timeline | 3.18 | N3: không gắp được butter | [ảnh](frames/D_nas50/t1_e0_s1000.jpg) |
| e1 | 1000 | 0 | F1 gắp hỏng | timeline | 4.68 | N1: gắp trượt hộp xanh, mang tay không tới giỏ | [ảnh](frames/D_nas50/t1_e1_s1000.jpg) |
| e5 | 1000 | 0 | F0 không xác định | khung điển hình | 2.60 | U | [ảnh](frames/D_nas50/t1_e5_s1000.jpg) |
| e7 | 1000 | 0 | F1 gắp hỏng | timeline | 2.96 | N1: gắp trượt hộp xanh, mang tay không tới giỏ | [ảnh](frames/D_nas50/t1_e7_s1000.jpg) |
| e8 | 1000 | 0 | F1 gắp hỏng | khung điển hình | 3.34 | N1 (khung bước 150 + khung cuối) | [ảnh](frames/D_nas50/t1_e8_s1000.jpg) |

### Task 2

| Episode | Seed | Giai đoạn | Nguyên nhân | Bằng chứng | Chuyển động | Ghi chú | Ảnh |
|---|---|---|---|---|---|---|---|
| e1 | 1000 | 1 | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 0.85 | N3: lơ lửng trên quai ấm | [ảnh](frames/D_nas50/t2_e1_s1000.jpg) |
| e2 | 1000 | 1 | F1 gắp hỏng | timeline | 0.25 | N3: không nhấc được ấm | [ảnh](frames/D_nas50/t2_e2_s1000.jpg) |
| e7 | 1000 | 1 | F2 làm đổ / kẹp sai tư thế | timeline | 1.11 | N3: làm đổ ấm | [ảnh](frames/D_nas50/t2_e7_s1000.jpg) |

### Task 3

| Episode | Seed | Giai đoạn | Nguyên nhân | Bằng chứng | Chuyển động | Ghi chú | Ảnh |
|---|---|---|---|---|---|---|---|
| e0 | 1000 | 1 | F4 đặt không đúng / gần đích | timeline | 0.13 | N3: bát trong ngăn kéo, ngăn kéo chưa đóng hẳn | [ảnh](frames/D_nas50/t3_e0_s1000.jpg) |
| e8 | 1000 | 1 | F4 đặt không đúng / gần đích | timeline | 0.11 | N3: bát trong ngăn kéo, ngăn kéo chưa đóng hẳn | [ảnh](frames/D_nas50/t3_e8_s1000.jpg) |

### Task 4

| Episode | Seed | Giai đoạn | Nguyên nhân | Bằng chứng | Chuyển động | Ghi chú | Ảnh |
|---|---|---|---|---|---|---|---|
| e1 | 1000 | 1 | F4 đặt không đúng / gần đích | timeline | 0.01 | N6: gần đích, đứng yên, không tính | [ảnh](frames/D_nas50/t4_e1_s1000.jpg) |
| e3 | 1000 | 1 | F4 đặt không đúng / gần đích | khung điển hình | 0.01 | N6 | [ảnh](frames/D_nas50/t4_e3_s1000.jpg) |
| e5 | 1000 | 1 | F4 đặt không đúng / gần đích | timeline | 0.01 | N6: gần đích, đứng yên, không tính | [ảnh](frames/D_nas50/t4_e5_s1000.jpg) |
| e6 | 1000 | 1 | F4 đặt không đúng / gần đích | khung điển hình | 0.04 | N6 | [ảnh](frames/D_nas50/t4_e6_s1000.jpg) |
| e7 | 1000 | 1 | F4 đặt không đúng / gần đích | khung điển hình | 0.01 | N6 | [ảnh](frames/D_nas50/t4_e7_s1000.jpg) |
| e8 | 1000 | 0 | F3 đứng yên / lơ lửng không tiến triển | timeline | 2.15 | U: cầm cốc trắng lơ lửng cạnh đĩa | [ảnh](frames/D_nas50/t4_e8_s1000.jpg) |

### Task 5

| Episode | Seed | Giai đoạn | Nguyên nhân | Bằng chứng | Chuyển động | Ghi chú | Ảnh |
|---|---|---|---|---|---|---|---|
| e5 | 1000 | 0 | F4 đặt không đúng / gần đích | timeline | 0.21 | N7: sách sai ngăn | [ảnh](frames/D_nas50/t5_e5_s1000.jpg) |
| e8 | 1000 | 0 | F4 đặt không đúng / gần đích | timeline | 0.03 | N7: sách sai ngăn | [ảnh](frames/D_nas50/t5_e8_s1000.jpg) |

### Task 6

| Episode | Seed | Giai đoạn | Nguyên nhân | Bằng chứng | Chuyển động | Ghi chú | Ảnh |
|---|---|---|---|---|---|---|---|
| e1 | 1000 | 1 | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 0.84 | N3 | [ảnh](frames/D_nas50/t6_e1_s1000.jpg) |
| e4 | 1000 | 1 | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 0.79 | N3 | [ảnh](frames/D_nas50/t6_e4_s1000.jpg) |
| e5 | 1000 | 1 | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 0.61 | N3: tay vẫn ôm cốc | [ảnh](frames/D_nas50/t6_e5_s1000.jpg) |
| e6 | 1000 | 0 | F1 gắp hỏng | timeline | 3.07 | N1: bỏ qua cốc trắng, chuyển sang pudding | [ảnh](frames/D_nas50/t6_e6_s1000.jpg) |
| e7 | 1000 | 1 | F3 đứng yên / lơ lửng không tiến triển | timeline | 0.42 | N3: cầm pudding cạnh đĩa | [ảnh](frames/D_nas50/t6_e7_s1000.jpg) |
| e8 | 1000 | 0 | F1 gắp hỏng | khung điển hình | 2.37 | N1 | [ảnh](frames/D_nas50/t6_e8_s1000.jpg) |
| e9 | 1000 | 1 | F3 đứng yên / lơ lửng không tiến triển | timeline | 0.61 | N3: cầm pudding cạnh đĩa | [ảnh](frames/D_nas50/t6_e9_s1000.jpg) |

### Task 7

| Episode | Seed | Giai đoạn | Nguyên nhân | Bằng chứng | Chuyển động | Ghi chú | Ảnh |
|---|---|---|---|---|---|---|---|
| e0 | 1000 | 0 | F1 gắp hỏng | timeline | 2.16 | N1: gắp trượt, rồi nhấc cả giỏ | [ảnh](frames/D_nas50/t7_e0_s1000.jpg) |
| e2 | 1000 | 1 | F3 đứng yên / lơ lửng không tiến triển | timeline | 1.19 | N3: lơ lửng trên hộp cream cheese | [ảnh](frames/D_nas50/t7_e2_s1000.jpg) |
| e3 | 1000 | 0 | F5 nhầm vật / nhầm đích | khung điển hình | 2.93 | N5: nhấc cả giỏ | [ảnh](frames/D_nas50/t7_e3_s1000.jpg) |
| e4 | 1000 | 0 | F1 gắp hỏng | timeline | 1.18 | N1: tay không tới giỏ | [ảnh](frames/D_nas50/t7_e4_s1000.jpg) |
| e5 | 1000 | 0 | F1 gắp hỏng | khung điển hình | 1.61 | N1 | [ảnh](frames/D_nas50/t7_e5_s1000.jpg) |
| e6 | 1000 | 0 | F1 gắp hỏng | timeline | 1.61 | N1: tay không tới giỏ | [ảnh](frames/D_nas50/t7_e6_s1000.jpg) |
| e8 | 1000 | 0 | F5 nhầm vật / nhầm đích | khung điển hình | 2.20 | N5: nhấc cả giỏ | [ảnh](frames/D_nas50/t7_e8_s1000.jpg) |

### Task 8

| Episode | Seed | Giai đoạn | Nguyên nhân | Bằng chứng | Chuyển động | Ghi chú | Ảnh |
|---|---|---|---|---|---|---|---|
| e1 | 1000 | 1 | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 1.93 | N3: ấm 2 chưa lên bếp | [ảnh](frames/D_nas50/t8_e1_s1000.jpg) |
| e2 | 1000 | 1 | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 2.40 | N3: ấm 2 chưa lên bếp | [ảnh](frames/D_nas50/t8_e2_s1000.jpg) |
| e3 | 1000 | 1 | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 1.88 | N3: tay vẫn kẹp ấm thứ nhất trên bếp; ấm 2 trên bàn | [ảnh](frames/D_nas50/t8_e3_s1000.jpg) |
| e4 | 1000 | 1 | F2 làm đổ / kẹp sai tư thế | khung điển hình | 0.95 | N3: ấm 2 bị kẹp nằm ngang | [ảnh](frames/D_nas50/t8_e4_s1000.jpg) |
| e5 | 1000 | 1 | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 2.34 | N3: ấm 2 chưa lên bếp | [ảnh](frames/D_nas50/t8_e5_s1000.jpg) |
| e7 | 1000 | 1 | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 2.81 | N3: ấm 2 chưa lên bếp | [ảnh](frames/D_nas50/t8_e7_s1000.jpg) |

### Task 9

| Episode | Seed | Giai đoạn | Nguyên nhân | Bằng chứng | Chuyển động | Ghi chú | Ảnh |
|---|---|---|---|---|---|---|---|
| e0 | 1000 | 0 | F4 đặt không đúng / gần đích | timeline | 0.97 | N2: để cốc ngoài lò rồi thao tác cửa lò | [ảnh](frames/D_nas50/t9_e0_s1000.jpg) |
| e2 | 1000 | 0 | F4 đặt không đúng / gần đích | timeline | 0.41 | N2: để cốc ngoài lò rồi thao tác cửa lò | [ảnh](frames/D_nas50/t9_e2_s1000.jpg) |
