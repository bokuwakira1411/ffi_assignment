# D_nas10: libero_10, n_action_steps = 10, relative, seed 1000–1002, 50 episode/task

Chạy ngày 31/10/2026 bằng `TAG=D_nas10 NAS=10 CM=relative N_EP=50 SEEDS=1000 bash run_queue.sh`, giống hệt D_nas1
và D_nas50 trừ `--policy.n_action_steps=10` (`cmd.txt`). Thời gian thực: 1597 s (tổng `seconds=` trong `status.txt`).
Lý do chọn 10: `lerobot/docs/source/libero.mdx:205` dùng `n_action_steps=10` cho **Pi0.5** ("matching the original
OpenPI implementation"); tài liệu không khuyến nghị giá trị này cho SmolVLA, nên đây là một phép thử thêm.
Seed 1001, 1002 chạy thêm ngày 31/10/2026 bằng cùng lệnh với `SEEDS="1001 1002"` (2366 s và 2392 s theo `timing.csv`;
chậm hơn seed 1000, nguyên nhân không xác định). Bảng tỉ lệ thành công bên dưới gộp cả 3 seed. **Phân loại nguyên nhân
chỉ làm cho seed 1000**: có video cho episode 0–9 mỗi task, 28 trên tổng 132 thất bại; nhãn chỉ dựa trên khung điển hình.
Video episode 0–9 của seed 1001, 1002 có sẵn nhưng chưa trích khung và chưa gán nhãn.

## Nhận xét

1. **NAS=10 tốt nhất trong ba giá trị**: 368/500 = 73,6% (Wilson 95%: 69,6–77,3), so với NAS=1 54,6% (50,2–58,9) và
   NAS=50 47,8% (43,5–52,2). Hai khoảng tin cậy NAS=10 và NAS=1 không chồng nhau. NAS=10 tốt hơn NAS=1 ở **cả 10 task**.
2. **So từng cảnh** (cùng task, cùng episode, tức cùng init state, 500 cặp): NAS=1 thất bại còn NAS=10 thành công ở
   118 cặp; ngược lại chỉ 23 cặp.
3. Trong 44 thất bại có video của D_nas1, NAS=10 thành công ở 20 cảnh: F6 ra khỏi vùng làm việc 7/12, F3 đứng yên 5/14,
   F2 làm đổ 4/13. Trong 48 thất bại có video của D_nas50, NAS=10 thành công ở 28 cảnh: **F1 gắp hỏng 10/16**,
   F4 đặt không đúng 8/11, F3 7/15.
   Cách hiểu **[suy luận, chưa kiểm chứng]**: quan sát lại sau 10 bước đủ để sửa lần gắp trượt (vấn đề của NAS=50), nhưng
   mỗi chunk 10 action vẫn đủ dài để chuyển động mạch lạc, không dừng hay đi lạc như NAS=1.
4. Phân bố nguyên nhân của 28 thất bại còn lại: F2 làm đổ / kẹp sai tư thế 9, F6 7, F4 5, F3 4. Kiểu "kẹp ấm thứ hai
   nằm ngang" ở task 8 (3 episode) và "cầm pudding cạnh đĩa" ở task 6 (4 episode) vẫn còn.
5. **Lặp lại với 3 seed**: 368 / 360 / 364 trên 500 (seed 1000 / 1001 / 1002), gộp 1092/1500 = 72,8% (Wilson 95%:
   70,5–75,0). Theo task (gộp, /150): 60, 136, 138, 146, 102, 135, 81, 104, 71, 119. Seed không đổi cảnh, chỉ đổi nhiễu
   lấy mẫu. Trên 500 cảnh: 289 thành công 3/3, 83 là 2/3, 59 là 1/3, 69 thất bại 3/3 (nhiều nhất task 0: 22, task 8: 14,
   task 6: 12). Nguồn: `per_task[0].metrics.successes` trong `eval_info.json`.

## Tỉ lệ thành công

Nguồn: `per_task[0].n_success` / `n_episodes` trong `eval_info.json` của từng job; seed: 1000, 1001, 1002.

| Task | Thành công | Tỉ lệ |
|---|---|---|
| 0 | 60/150 | 40% |
| 1 | 136/150 | 91% |
| 2 | 138/150 | 92% |
| 3 | 146/150 | 97% |
| 4 | 102/150 | 68% |
| 5 | 135/150 | 90% |
| 6 | 81/150 | 54% |
| 7 | 104/150 | 69% |
| 8 | 71/150 | 47% |
| 9 | 119/150 | 79% |
| **Tổng** | **1092/1500** | **72.8%** |

## Nguyên nhân thất bại theo task (chỉ các episode có video)

Mỗi episode có đúng một nhãn nguyên nhân chính (định nghĩa ở `docs/failure_overview.md`). Giai đoạn: 0 = chưa xong bước con thứ nhất; 1 = xong bước con thứ nhất, hỏng ở bước sau; ? = không xác định.

| Task | Thất bại có video | F1 | F2 | F3 | F4 | F5 | F6 | F0 | GĐ 0 | GĐ 1 | GĐ ? |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 9 |  | 3 |  |  | 1 | 4 | 1 | 7 | 1 | 1 |
| 2 | 1 |  | 1 |  |  |  |  |  |  | 1 |  |
| 4 | 4 |  | 1 |  | 2 |  | 1 |  |  | 3 | 1 |
| 5 | 1 |  |  |  | 1 |  |  |  | 1 |  |  |
| 6 | 5 |  |  | 4 |  |  | 1 |  | 1 | 4 |  |
| 7 | 2 | 1 | 1 |  |  |  |  |  | 1 | 1 |  |
| 8 | 4 |  | 3 |  |  |  | 1 |  |  | 4 |  |
| 9 | 2 |  |  |  | 2 |  |  |  | 1 | 1 |  |
| **Tổng** | **28** | **1** | **9** | **4** | **5** | **1** | **7** | **1** | **11** | **15** | **2** |

Mức bằng chứng: khung điển hình 28.

## Danh sách từng episode thất bại

Ảnh: dải 4 khung (bước 10 | 1/3 episode | khung điển hình | khung cuối), sinh bởi `docs/failures/tools/extract_frames.py`. Chuyển động = trung bình 100 bước cuối.

### Task 0

| Episode | Seed | Giai đoạn | Nguyên nhân | Bằng chứng | Chuyển động | Ghi chú | Ảnh |
|---|---|---|---|---|---|---|---|
| e0 | 1000 | 0 | F5 nhầm vật / nhầm đích | khung điển hình | 0.47 | cho hộp nước cam (vật không thuộc mục tiêu) vào giỏ | [ảnh](frames/D_nas10/t0_e0_s1000.jpg) |
| e1 | 1000 | 1 | F2 làm đổ / kẹp sai tư thế | khung điển hình | 0.15 | một lon trong giỏ; lon thứ hai bị kẹp nghiêng, lơ lửng | [ảnh](frames/D_nas10/t0_e1_s1000.jpg) |
| e2 | 1000 | 0 | F6 ra khỏi vùng làm việc | khung điển hình | 1.36 | giỏ trống; tay ở mép phải khung | [ảnh](frames/D_nas10/t0_e2_s1000.jpg) |
| e4 | 1000 | ? | F0 không xác định | khung điển hình | 2.24 | tay ở trong giỏ; lon đỏ không còn trên bàn; vì sao không tính: không xác định | [ảnh](frames/D_nas10/t0_e4_s1000.jpg) |
| e5 | 1000 | 0 | F6 ra khỏi vùng làm việc | khung điển hình | 0.73 | giỏ trống; tay ở mép phải khung | [ảnh](frames/D_nas10/t0_e5_s1000.jpg) |
| e6 | 1000 | 0 | F6 ra khỏi vùng làm việc | khung điển hình | 0.10 | giỏ trống; tay ra khỏi khung bên phải | [ảnh](frames/D_nas10/t0_e6_s1000.jpg) |
| e7 | 1000 | 0 | F6 ra khỏi vùng làm việc | khung điển hình | 0.32 | giỏ trống; tay ở mép phải khung | [ảnh](frames/D_nas10/t0_e7_s1000.jpg) |
| e8 | 1000 | 0 | F2 làm đổ / kẹp sai tư thế | khung điển hình | 2.08 | hộp nước cam bị làm rơi cạnh giỏ; giỏ trống | [ảnh](frames/D_nas10/t0_e8_s1000.jpg) |
| e9 | 1000 | 0 | F2 làm đổ / kẹp sai tư thế | khung điển hình | 0.59 | lon xanh bị đổ nằm ngang | [ảnh](frames/D_nas10/t0_e9_s1000.jpg) |

### Task 2

| Episode | Seed | Giai đoạn | Nguyên nhân | Bằng chứng | Chuyển động | Ghi chú | Ảnh |
|---|---|---|---|---|---|---|---|
| e1 | 1000 | 1 | F2 làm đổ / kẹp sai tư thế | khung điển hình | 1.49 | bếp đã bật; ấm moka bị làm đổ nằm cạnh bếp | [ảnh](frames/D_nas10/t2_e1_s1000.jpg) |

### Task 4

| Episode | Seed | Giai đoạn | Nguyên nhân | Bằng chứng | Chuyển động | Ghi chú | Ảnh |
|---|---|---|---|---|---|---|---|
| e0 | 1000 | 1 | F2 làm đổ / kẹp sai tư thế | khung điển hình | 0.17 | cốc trắng ở đĩa phải; cốc vàng-trắng bị nghiêng ở mép đĩa trái | [ảnh](frames/D_nas10/t4_e0_s1000.jpg) |
| e2 | 1000 | ? | F6 ra khỏi vùng làm việc | khung điển hình | 0.49 | cốc vàng-trắng chưa đụng tới; cánh tay vươn ra khỏi khung bên phải | [ảnh](frames/D_nas10/t4_e2_s1000.jpg) |
| e6 | 1000 | 1 | F4 đặt không đúng / gần đích | khung điển hình | 0.04 | hai cốc có vẻ ở hai đĩa, robot đứng yên, không được tính | [ảnh](frames/D_nas10/t4_e6_s1000.jpg) |
| e7 | 1000 | 1 | F4 đặt không đúng / gần đích | khung điển hình | 0.12 | hai cốc có vẻ ở hai đĩa, không được tính | [ảnh](frames/D_nas10/t4_e7_s1000.jpg) |

### Task 5

| Episode | Seed | Giai đoạn | Nguyên nhân | Bằng chứng | Chuyển động | Ghi chú | Ảnh |
|---|---|---|---|---|---|---|---|
| e5 | 1000 | 0 | F4 đặt không đúng / gần đích | khung điển hình | 0.09 | sách bị thả nằm nghiêng ở ngăn phải | [ảnh](frames/D_nas10/t5_e5_s1000.jpg) |

### Task 6

| Episode | Seed | Giai đoạn | Nguyên nhân | Bằng chứng | Chuyển động | Ghi chú | Ảnh |
|---|---|---|---|---|---|---|---|
| e0 | 1000 | 1 | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 0.80 | cốc trên đĩa; cầm pudding cạnh đĩa | [ảnh](frames/D_nas10/t6_e0_s1000.jpg) |
| e1 | 1000 | 1 | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 0.22 | cốc trên đĩa; cầm pudding cạnh đĩa | [ảnh](frames/D_nas10/t6_e1_s1000.jpg) |
| e4 | 1000 | 1 | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 0.41 | cốc trên đĩa; cầm pudding cạnh đĩa | [ảnh](frames/D_nas10/t6_e4_s1000.jpg) |
| e8 | 1000 | 0 | F6 ra khỏi vùng làm việc | khung điển hình | 0.52 | cốc trắng chưa đụng tới; tay ra mép phải khung | [ảnh](frames/D_nas10/t6_e8_s1000.jpg) |
| e9 | 1000 | 1 | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 0.06 | cốc trên đĩa; cầm pudding cạnh đĩa, đứng yên | [ảnh](frames/D_nas10/t6_e9_s1000.jpg) |

### Task 7

| Episode | Seed | Giai đoạn | Nguyên nhân | Bằng chứng | Chuyển động | Ghi chú | Ảnh |
|---|---|---|---|---|---|---|---|
| e3 | 1000 | 0 | F1 gắp hỏng | khung điển hình | 0.26 | kẹp lon xanh tại chỗ nhưng không nhấc lên | [ảnh](frames/D_nas10/t7_e3_s1000.jpg) |
| e5 | 1000 | 1 | F2 làm đổ / kẹp sai tư thế | khung điển hình | 0.95 | hộp cream cheese trong giỏ; lon xanh bị đổ nằm ngang trên bàn | [ảnh](frames/D_nas10/t7_e5_s1000.jpg) |

### Task 8

| Episode | Seed | Giai đoạn | Nguyên nhân | Bằng chứng | Chuyển động | Ghi chú | Ảnh |
|---|---|---|---|---|---|---|---|
| e2 | 1000 | 1 | F2 làm đổ / kẹp sai tư thế | khung điển hình | 0.50 | một ấm trên bếp; ấm 2 bị kẹp nằm ngang | [ảnh](frames/D_nas10/t8_e2_s1000.jpg) |
| e3 | 1000 | 1 | F2 làm đổ / kẹp sai tư thế | khung điển hình | 0.31 | một ấm trên bếp; ấm 2 bị kẹp nằm ngang | [ảnh](frames/D_nas10/t8_e3_s1000.jpg) |
| e4 | 1000 | 1 | F6 ra khỏi vùng làm việc | khung điển hình | 0.27 | một ấm trên bếp; ấm 2 trên bàn; tay ra mép trên trái khung | [ảnh](frames/D_nas10/t8_e4_s1000.jpg) |
| e9 | 1000 | 1 | F2 làm đổ / kẹp sai tư thế | khung điển hình | 0.81 | một ấm trên bếp; ấm 2 bị kẹp nghiêng | [ảnh](frames/D_nas10/t8_e9_s1000.jpg) |

### Task 9

| Episode | Seed | Giai đoạn | Nguyên nhân | Bằng chứng | Chuyển động | Ghi chú | Ảnh |
|---|---|---|---|---|---|---|---|
| e7 | 1000 | 0 | F4 đặt không đúng / gần đích | khung điển hình | 0.98 | cốc ở mép cửa lò, chưa vào trong | [ảnh](frames/D_nas10/t9_e7_s1000.jpg) |
| e9 | 1000 | 1 | F4 đặt không đúng / gần đích | khung điển hình | 0.21 | cốc không còn trên bàn (có vẻ đã vào lò); cửa lò chưa đóng | [ảnh](frames/D_nas10/t9_e9_s1000.jpg) |
