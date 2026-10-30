# B_main: libero_10, n_action_steps = 1, relative, 3 seed

Cấu hình (`cmd.txt`): `lerobot/smolvla_libero`, `--policy.n_action_steps=1`, `--env.control_mode=relative`,
10 episode mỗi job, seed 1000/1001/1002. Episode i dùng init state i, nên 3 seed dùng cùng 10 cảnh ban đầu
(`differences.md` mục 3). Phân tích chi tiết từng nhóm: `failure_analysis.md` mục 2–4.

Bản này gán lại 142 thất bại vào bộ nguyên nhân chung F0–F6. Các nhãn lấy từ mục 3 của `failure_analysis.md`.
31 episode trước đây là "không xác định" đã được xem lại bằng khung điển hình ngày 31/10/2026; hiện chỉ còn 2 episode F0.

## Nhận xét

1. **Nguyên nhân lớn nhất là F3 (53/142)**: robot dừng hoặc lơ lửng, không thực hiện thao tác tiếp theo. Tập trung ở
   task 6 (16/21: cốc đã lên đĩa, robot ôm cốc hoặc lơ lửng ở hộp pudding), task 8 (16/29) và task 9 (7/7: cầm cốc
   trước cửa lò vi sóng).
2. **F2 (39/142)**: làm đổ hoặc kẹp sai tư thế. Ở task 0, 12 episode **với vào cụm hộp bên phải** (lon alphabet soup
   nằm sau hộp sữa) và làm đổ lon hoặc hộp nước cam (`e2 s1000` [timeline], 11 episode còn lại [khung điển hình]). Ở task 8, 13 episode kẹp ấm nằm ngang hoặc làm đổ ấm.
3. **F6 (27/142)**: cánh tay đi ra khỏi vùng làm việc và ở đó đến hết giờ. Task 1 (11/14: cầm hộp cream cheese đi
   về góc trên trái) và task 4 (8/23: mang cốc trắng sang phải, cánh tay vươn ra khỏi khung).
4. **F1 kiểu "tay không"** cũng xuất hiện với NAS=1: ở task 0 `e0 s1001` và `e0 s1002`, tay tới giỏ nhưng lon xanh vẫn
   nằm sau hộp sữa **[khung điển hình]**. Như vậy kiểu thất bại N1 của D_nas50 (`failure_analysis.md` mục 6.5) không
   chỉ có ở NAS=50, nhưng với NAS=1 mới thấy 2/142 so với 10/48 ở NAS=50.
5. Giai đoạn: 59 episode hỏng trước khi xong bước con thứ nhất, 51 hỏng ở bước sau, 32 không xác định (chủ yếu task 1
   và 4, vì camera ngoài không thấy vật thứ nhất đã tới đích hay chưa).

## Tỉ lệ thành công

Nguồn: `per_task[0].n_success` / `n_episodes` trong `eval_info.json` của từng job; seed: 1000, 1001, 1002.

| Task | Thành công | Tỉ lệ |
|---|---|---|
| 0 | 1/30 | 3% |
| 1 | 16/30 | 53% |
| 2 | 28/30 | 93% |
| 3 | 30/30 | 100% |
| 4 | 7/30 | 23% |
| 5 | 27/30 | 90% |
| 6 | 9/30 | 30% |
| 7 | 16/30 | 53% |
| 8 | 1/30 | 3% |
| 9 | 23/30 | 77% |
| **Tổng** | **158/300** | **52.7%** |

## Nguyên nhân thất bại theo task (chỉ các episode có video)

Mỗi episode có đúng một nhãn nguyên nhân chính (định nghĩa ở `docs/failure_overview.md`). Giai đoạn: 0 = chưa xong bước con thứ nhất; 1 = xong bước con thứ nhất, hỏng ở bước sau; ? = không xác định.

| Task | Thất bại có video | F1 | F2 | F3 | F4 | F5 | F6 | F0 | GĐ 0 | GĐ 1 | GĐ ? |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 29 | 2 | 12 | 8 |  | 1 | 5 | 1 | 20 | 9 |  |
| 1 | 14 | 3 |  |  |  |  | 11 |  | 3 |  | 11 |
| 2 | 2 |  |  |  | 1 |  | 1 |  |  | 2 |  |
| 4 | 23 |  | 5 | 6 | 3 |  | 8 | 1 |  | 4 | 19 |
| 5 | 3 |  |  |  | 3 |  |  |  | 3 |  |  |
| 6 | 21 |  | 4 | 16 | 1 |  |  |  | 5 | 16 |  |
| 7 | 14 | 3 | 5 |  | 4 |  | 2 |  | 8 | 4 | 2 |
| 8 | 29 |  | 13 | 16 |  |  |  |  | 13 | 16 |  |
| 9 | 7 |  |  | 7 |  |  |  |  | 7 |  |  |
| **Tổng** | **142** | **8** | **39** | **53** | **12** | **1** | **27** | **2** | **59** | **51** | **32** |

Mức bằng chứng: khung điển hình 136, timeline 6.

## Danh sách từng episode thất bại

Ảnh: dải 4 khung (bước 10 | 1/3 episode | khung điển hình | khung cuối), sinh bởi `docs/failures/tools/extract_frames.py`. Chuyển động = trung bình 100 bước cuối.

### Task 0

| Episode | Seed | Giai đoạn | Nguyên nhân | Bằng chứng | Chuyển động | Ghi chú | Ảnh |
|---|---|---|---|---|---|---|---|
| e0 | 1000 | 0 | F3 đứng yên / lơ lửng không tiến triển | timeline | 0.07 | B: không lấy được lon xanh, lơ lửng giữa bàn | [ảnh](frames/B_main/t0_e0_s1000.jpg) |
| e1 | 1000 | 1 | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 0.11 | A1: lon xanh trong giỏ; cầm lon đỏ lơ lửng (e1 s1000: T) | [ảnh](frames/B_main/t0_e1_s1000.jpg) |
| e2 | 1000 | 0 | F2 làm đổ / kẹp sai tư thế | timeline | 0.79 | B: tay len giữa các hộp, làm đổ hộp và lon | [ảnh](frames/B_main/t0_e2_s1000.jpg) |
| e3 | 1000 | 0 | F2 làm đổ / kẹp sai tư thế | khung điển hình | 0.56 | B: khung điển hình: với vào cụm hộp bên phải, làm đổ lon xanh hoặc hộp nước cam | [ảnh](frames/B_main/t0_e3_s1000.jpg) |
| e4 | 1000 | 1 | F5 nhầm vật / nhầm đích | timeline | 1.58 | A4: nhấc cả giỏ | [ảnh](frames/B_main/t0_e4_s1000.jpg) |
| e5 | 1000 | 0 | F2 làm đổ / kẹp sai tư thế | khung điển hình | 0.19 | B: khung điển hình: với vào cụm hộp bên phải, làm đổ lon xanh hoặc hộp nước cam | [ảnh](frames/B_main/t0_e5_s1000.jpg) |
| e6 | 1000 | 0 | F2 làm đổ / kẹp sai tư thế | khung điển hình | 0.03 | B: khung điển hình: với vào cụm hộp bên phải, làm đổ lon xanh hoặc hộp nước cam | [ảnh](frames/B_main/t0_e6_s1000.jpg) |
| e8 | 1000 | 0 | F2 làm đổ / kẹp sai tư thế | khung điển hình | 0.21 | B: khung điển hình: với vào cụm hộp bên phải, làm đổ lon xanh hoặc hộp nước cam | [ảnh](frames/B_main/t0_e8_s1000.jpg) |
| e9 | 1000 | 1 | F0 không xác định | khung điển hình | 1.85 | A2: lon xanh trong giỏ, lon đỏ trên bàn, tay trên giỏ | [ảnh](frames/B_main/t0_e9_s1000.jpg) |
| e0 | 1001 | 0 | F1 gắp hỏng | khung điển hình | 1.16 | B: tay tới giỏ nhưng lon xanh vẫn ở chỗ cũ sau hộp sữa (tay không) | [ảnh](frames/B_main/t0_e0_s1001.jpg) |
| e1 | 1001 | 1 | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 0.20 | A1: lon xanh trong giỏ; cầm lon đỏ lơ lửng (e1 s1000: T) | [ảnh](frames/B_main/t0_e1_s1001.jpg) |
| e2 | 1001 | 0 | F2 làm đổ / kẹp sai tư thế | khung điển hình | 0.78 | B: khung điển hình: với vào cụm hộp bên phải, làm đổ lon xanh hoặc hộp nước cam | [ảnh](frames/B_main/t0_e2_s1001.jpg) |
| e3 | 1001 | 0 | F2 làm đổ / kẹp sai tư thế | khung điển hình | 0.50 | B: khung điển hình: với vào cụm hộp bên phải, làm đổ lon xanh hoặc hộp nước cam | [ảnh](frames/B_main/t0_e3_s1001.jpg) |
| e4 | 1001 | 1 | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 0.03 | A3: lơ lửng đứng yên trên lon đỏ (s1001: T) | [ảnh](frames/B_main/t0_e4_s1001.jpg) |
| e5 | 1001 | 0 | F2 làm đổ / kẹp sai tư thế | khung điển hình | 0.20 | B: khung điển hình: với vào cụm hộp bên phải, làm đổ lon xanh hoặc hộp nước cam | [ảnh](frames/B_main/t0_e5_s1001.jpg) |
| e6 | 1001 | 0 | F6 ra khỏi vùng làm việc | khung điển hình | 0.14 | B: tay ở mép phải khung, ra khỏi khung một phần | [ảnh](frames/B_main/t0_e6_s1001.jpg) |
| e7 | 1001 | 0 | F2 làm đổ / kẹp sai tư thế | khung điển hình | 0.04 | B: khung điển hình: với vào cụm hộp bên phải, làm đổ lon xanh hoặc hộp nước cam | [ảnh](frames/B_main/t0_e7_s1001.jpg) |
| e8 | 1001 | 0 | F6 ra khỏi vùng làm việc | khung điển hình | 0.11 | B: tay ở mép phải khung, ra khỏi khung một phần | [ảnh](frames/B_main/t0_e8_s1001.jpg) |
| e9 | 1001 | 1 | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 0.30 | A1 | [ảnh](frames/B_main/t0_e9_s1001.jpg) |
| e0 | 1002 | 0 | F1 gắp hỏng | khung điển hình | 2.02 | B: tay tới giỏ nhưng lon xanh vẫn ở chỗ cũ sau hộp sữa (tay không) | [ảnh](frames/B_main/t0_e0_s1002.jpg) |
| e1 | 1002 | 1 | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 0.26 | A1: lon xanh trong giỏ; cầm lon đỏ lơ lửng (e1 s1000: T) | [ảnh](frames/B_main/t0_e1_s1002.jpg) |
| e2 | 1002 | 0 | F6 ra khỏi vùng làm việc | khung điển hình | 1.07 | B: tay ở mép phải khung, ra khỏi khung một phần | [ảnh](frames/B_main/t0_e2_s1002.jpg) |
| e3 | 1002 | 0 | F2 làm đổ / kẹp sai tư thế | khung điển hình | 0.87 | B: khung điển hình: với vào cụm hộp bên phải, làm đổ lon xanh hoặc hộp nước cam | [ảnh](frames/B_main/t0_e3_s1002.jpg) |
| e4 | 1002 | 1 | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 0.03 | A3: lơ lửng đứng yên trên lon đỏ (s1001: T) | [ảnh](frames/B_main/t0_e4_s1002.jpg) |
| e5 | 1002 | 0 | F2 làm đổ / kẹp sai tư thế | khung điển hình | 0.22 | B: khung điển hình: với vào cụm hộp bên phải, làm đổ lon xanh hoặc hộp nước cam | [ảnh](frames/B_main/t0_e5_s1002.jpg) |
| e6 | 1002 | 0 | F6 ra khỏi vùng làm việc | khung điển hình | 0.25 | B: tay ở mép phải khung, ra khỏi khung một phần | [ảnh](frames/B_main/t0_e6_s1002.jpg) |
| e7 | 1002 | 0 | F2 làm đổ / kẹp sai tư thế | khung điển hình | 0.04 | B: khung điển hình: với vào cụm hộp bên phải, làm đổ lon xanh hoặc hộp nước cam | [ảnh](frames/B_main/t0_e7_s1002.jpg) |
| e8 | 1002 | 0 | F6 ra khỏi vùng làm việc | khung điển hình | 0.07 | B: tay ở mép phải khung, ra khỏi khung một phần | [ảnh](frames/B_main/t0_e8_s1002.jpg) |
| e9 | 1002 | 1 | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 0.44 | A1 | [ảnh](frames/B_main/t0_e9_s1002.jpg) |

### Task 1

| Episode | Seed | Giai đoạn | Nguyên nhân | Bằng chứng | Chuyển động | Ghi chú | Ảnh |
|---|---|---|---|---|---|---|---|
| e2 | 1000 | ? | F6 ra khỏi vùng làm việc | khung điển hình | 0.10 | A: cầm hộp xanh đi về góc trên trái, ra khỏi khung (e3 s1000: T) | [ảnh](frames/B_main/t1_e2_s1000.jpg) |
| e3 | 1000 | ? | F6 ra khỏi vùng làm việc | khung điển hình | 0.17 | A: cầm hộp xanh đi về góc trên trái, ra khỏi khung (e3 s1000: T) | [ảnh](frames/B_main/t1_e3_s1000.jpg) |
| e5 | 1000 | ? | F6 ra khỏi vùng làm việc | khung điển hình | 0.02 | A (e5 s1000: T) | [ảnh](frames/B_main/t1_e5_s1000.jpg) |
| e8 | 1000 | ? | F6 ra khỏi vùng làm việc | khung điển hình | 0.15 | A: cầm hộp xanh đi về góc trên trái, ra khỏi khung (e3 s1000: T) | [ảnh](frames/B_main/t1_e8_s1000.jpg) |
| e0 | 1001 | 0 | F1 gắp hỏng | khung điển hình | 0.58 | B: kẹt ở hộp xanh, không nhấc lên được (e0 s1002: T) | [ảnh](frames/B_main/t1_e0_s1001.jpg) |
| e2 | 1001 | ? | F6 ra khỏi vùng làm việc | khung điển hình | 0.15 | A: cầm hộp xanh đi về góc trên trái, ra khỏi khung (e3 s1000: T) | [ảnh](frames/B_main/t1_e2_s1001.jpg) |
| e3 | 1001 | ? | F6 ra khỏi vùng làm việc | khung điển hình | 0.10 | A: cầm hộp xanh đi về góc trên trái, ra khỏi khung (e3 s1000: T) | [ảnh](frames/B_main/t1_e3_s1001.jpg) |
| e5 | 1001 | ? | F6 ra khỏi vùng làm việc | khung điển hình | 0.06 | A (e5 s1000: T) | [ảnh](frames/B_main/t1_e5_s1001.jpg) |
| e8 | 1001 | ? | F6 ra khỏi vùng làm việc | khung điển hình | 0.13 | A: cầm hộp xanh đi về góc trên trái, ra khỏi khung (e3 s1000: T) | [ảnh](frames/B_main/t1_e8_s1001.jpg) |
| e0 | 1002 | 0 | F1 gắp hỏng | khung điển hình | 0.10 | B: kẹt ở hộp xanh, không nhấc lên được (e0 s1002: T) | [ảnh](frames/B_main/t1_e0_s1002.jpg) |
| e1 | 1002 | 0 | F1 gắp hỏng | khung điển hình | 0.02 | B | [ảnh](frames/B_main/t1_e1_s1002.jpg) |
| e2 | 1002 | ? | F6 ra khỏi vùng làm việc | khung điển hình | 0.23 | A: cầm hộp xanh đi về góc trên trái, ra khỏi khung (e3 s1000: T) | [ảnh](frames/B_main/t1_e2_s1002.jpg) |
| e3 | 1002 | ? | F6 ra khỏi vùng làm việc | khung điển hình | 0.20 | A: cầm hộp xanh đi về góc trên trái, ra khỏi khung (e3 s1000: T) | [ảnh](frames/B_main/t1_e3_s1002.jpg) |
| e8 | 1002 | ? | F6 ra khỏi vùng làm việc | khung điển hình | 0.11 | A: cầm hộp xanh đi về góc trên trái, ra khỏi khung (e3 s1000: T) | [ảnh](frames/B_main/t1_e8_s1002.jpg) |

### Task 2

| Episode | Seed | Giai đoạn | Nguyên nhân | Bằng chứng | Chuyển động | Ghi chú | Ảnh |
|---|---|---|---|---|---|---|---|
| e1 | 1000 | 1 | F6 ra khỏi vùng làm việc | khung điển hình | 1.47 | bếp đã bật, ấm trên bàn; tay ra khỏi khung | [ảnh](frames/B_main/t2_e1_s1000.jpg) |
| e5 | 1002 | 1 | F4 đặt không đúng / gần đích | khung điển hình | 0.49 | bếp đã bật, cầm ấm ở mép bếp | [ảnh](frames/B_main/t2_e5_s1002.jpg) |

### Task 4

| Episode | Seed | Giai đoạn | Nguyên nhân | Bằng chứng | Chuyển động | Ghi chú | Ảnh |
|---|---|---|---|---|---|---|---|
| e0 | 1000 | ? | F2 làm đổ / kẹp sai tư thế | khung điển hình | 0.21 | B: cốc đổ / cầm nghiêng | [ảnh](frames/B_main/t4_e0_s1000.jpg) |
| e2 | 1000 | ? | F6 ra khỏi vùng làm việc | timeline | 0.22 | A: mang cốc trắng ra khỏi khung hình | [ảnh](frames/B_main/t4_e2_s1000.jpg) |
| e3 | 1000 | ? | F6 ra khỏi vùng làm việc | khung điển hình | 0.45 | A: mang cốc trắng sang phải, cánh tay vươn ra khỏi khung; cốc vàng-trắng chưa đụng tới | [ảnh](frames/B_main/t4_e3_s1000.jpg) |
| e4 | 1000 | 1 | F4 đặt không đúng / gần đích | timeline | 0.01 | C | [ảnh](frames/B_main/t4_e4_s1000.jpg) |
| e6 | 1000 | ? | F6 ra khỏi vùng làm việc | khung điển hình | 0.47 | A: mang cốc trắng sang phải, cánh tay vươn ra khỏi khung; cốc vàng-trắng chưa đụng tới | [ảnh](frames/B_main/t4_e6_s1000.jpg) |
| e7 | 1000 | ? | F2 làm đổ / kẹp sai tư thế | khung điển hình | 0.17 | B | [ảnh](frames/B_main/t4_e7_s1000.jpg) |
| e9 | 1000 | 1 | F3 đứng yên / lơ lửng không tiến triển | timeline | 0.15 | A: đặt cốc trắng rồi lơ lửng tại đĩa | [ảnh](frames/B_main/t4_e9_s1000.jpg) |
| e0 | 1001 | 1 | F4 đặt không đúng / gần đích | khung điển hình | 0.01 | C: gần đích, không tính | [ảnh](frames/B_main/t4_e0_s1001.jpg) |
| e2 | 1001 | ? | F6 ra khỏi vùng làm việc | khung điển hình | 0.28 | A: mang cốc trắng sang phải, cánh tay vươn ra khỏi khung; cốc vàng-trắng chưa đụng tới | [ảnh](frames/B_main/t4_e2_s1001.jpg) |
| e3 | 1001 | ? | F2 làm đổ / kẹp sai tư thế | khung điển hình | 0.13 | B | [ảnh](frames/B_main/t4_e3_s1001.jpg) |
| e4 | 1001 | ? | F2 làm đổ / kẹp sai tư thế | khung điển hình | 0.05 | B | [ảnh](frames/B_main/t4_e4_s1001.jpg) |
| e6 | 1001 | ? | F6 ra khỏi vùng làm việc | khung điển hình | 1.21 | A: mang cốc trắng sang phải, cánh tay vươn ra khỏi khung; cốc vàng-trắng chưa đụng tới | [ảnh](frames/B_main/t4_e6_s1001.jpg) |
| e7 | 1001 | ? | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 0.10 | A: lơ lửng/cầm cốc trắng ở đĩa phải đến hết giờ; cốc vàng-trắng chưa đụng tới | [ảnh](frames/B_main/t4_e7_s1001.jpg) |
| e9 | 1001 | ? | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 0.19 | A: lơ lửng/cầm cốc trắng ở đĩa phải đến hết giờ; cốc vàng-trắng chưa đụng tới | [ảnh](frames/B_main/t4_e9_s1001.jpg) |
| e0 | 1002 | ? | F2 làm đổ / kẹp sai tư thế | khung điển hình | 0.42 | B: cốc đổ / cầm nghiêng | [ảnh](frames/B_main/t4_e0_s1002.jpg) |
| e2 | 1002 | ? | F6 ra khỏi vùng làm việc | khung điển hình | 0.88 | A: mang cốc trắng sang phải, cánh tay vươn ra khỏi khung; cốc vàng-trắng chưa đụng tới | [ảnh](frames/B_main/t4_e2_s1002.jpg) |
| e3 | 1002 | ? | F6 ra khỏi vùng làm việc | khung điển hình | 0.50 | A: mang cốc trắng sang phải, cánh tay vươn ra khỏi khung; cốc vàng-trắng chưa đụng tới | [ảnh](frames/B_main/t4_e3_s1002.jpg) |
| e4 | 1002 | 1 | F4 đặt không đúng / gần đích | khung điển hình | 0.01 | C | [ảnh](frames/B_main/t4_e4_s1002.jpg) |
| e5 | 1002 | ? | F0 không xác định | khung điển hình | 0.03 | không phân loại được | [ảnh](frames/B_main/t4_e5_s1002.jpg) |
| e6 | 1002 | ? | F6 ra khỏi vùng làm việc | khung điển hình | 1.08 | A: mang cốc trắng sang phải, cánh tay vươn ra khỏi khung; cốc vàng-trắng chưa đụng tới | [ảnh](frames/B_main/t4_e6_s1002.jpg) |
| e7 | 1002 | ? | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 0.11 | A: lơ lửng/cầm cốc trắng ở đĩa phải đến hết giờ; cốc vàng-trắng chưa đụng tới | [ảnh](frames/B_main/t4_e7_s1002.jpg) |
| e8 | 1002 | ? | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 0.17 | A: lơ lửng/cầm cốc trắng ở đĩa phải đến hết giờ; cốc vàng-trắng chưa đụng tới | [ảnh](frames/B_main/t4_e8_s1002.jpg) |
| e9 | 1002 | ? | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 0.04 | A: lơ lửng/cầm cốc trắng ở đĩa phải đến hết giờ; cốc vàng-trắng chưa đụng tới | [ảnh](frames/B_main/t4_e9_s1002.jpg) |

### Task 5

| Episode | Seed | Giai đoạn | Nguyên nhân | Bằng chứng | Chuyển động | Ghi chú | Ảnh |
|---|---|---|---|---|---|---|---|
| e1 | 1001 | 0 | F4 đặt không đúng / gần đích | khung điển hình | 0.06 | sách ở ngăn phải | [ảnh](frames/B_main/t5_e1_s1001.jpg) |
| e1 | 1002 | 0 | F4 đặt không đúng / gần đích | khung điển hình | 0.07 | sách ở ngăn phải | [ảnh](frames/B_main/t5_e1_s1002.jpg) |
| e7 | 1002 | 0 | F4 đặt không đúng / gần đích | khung điển hình | 0.21 | cầm sách ở ngăn phải | [ảnh](frames/B_main/t5_e7_s1002.jpg) |

### Task 6

| Episode | Seed | Giai đoạn | Nguyên nhân | Bằng chứng | Chuyển động | Ghi chú | Ảnh |
|---|---|---|---|---|---|---|---|
| e0 | 1000 | 1 | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 0.06 | A: cốc trên đĩa; lơ lửng/cầm ở pudding | [ảnh](frames/B_main/t6_e0_s1000.jpg) |
| e1 | 1000 | 0 | F4 đặt không đúng / gần đích | khung điển hình | 0.53 | D: cốc trước đĩa | [ảnh](frames/B_main/t6_e1_s1000.jpg) |
| e2 | 1000 | 0 | F2 làm đổ / kẹp sai tư thế | khung điển hình | 0.09 | D: cốc bị cầm nghiêng | [ảnh](frames/B_main/t6_e2_s1000.jpg) |
| e3 | 1000 | 1 | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 0.26 | B: tay vẫn ôm cốc trên đĩa | [ảnh](frames/B_main/t6_e3_s1000.jpg) |
| e4 | 1000 | 1 | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 0.16 | B | [ảnh](frames/B_main/t6_e4_s1000.jpg) |
| e5 | 1000 | 1 | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 0.22 | B: tay vẫn ôm cốc trên đĩa | [ảnh](frames/B_main/t6_e5_s1000.jpg) |
| e8 | 1000 | 0 | F2 làm đổ / kẹp sai tư thế | khung điển hình | 0.24 | C: làm đổ cốc đỏ | [ảnh](frames/B_main/t6_e8_s1000.jpg) |
| e0 | 1001 | 1 | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 0.02 | A: cốc trên đĩa; lơ lửng/cầm ở pudding | [ảnh](frames/B_main/t6_e0_s1001.jpg) |
| e1 | 1001 | 1 | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 0.16 | A | [ảnh](frames/B_main/t6_e1_s1001.jpg) |
| e2 | 1001 | 1 | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 0.15 | A | [ảnh](frames/B_main/t6_e2_s1001.jpg) |
| e3 | 1001 | 1 | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 0.36 | B: tay vẫn ôm cốc trên đĩa | [ảnh](frames/B_main/t6_e3_s1001.jpg) |
| e4 | 1001 | 1 | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 0.07 | A | [ảnh](frames/B_main/t6_e4_s1001.jpg) |
| e5 | 1001 | 1 | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 0.23 | B: tay vẫn ôm cốc trên đĩa | [ảnh](frames/B_main/t6_e5_s1001.jpg) |
| e8 | 1001 | 0 | F2 làm đổ / kẹp sai tư thế | khung điển hình | 0.23 | C: làm đổ cốc đỏ | [ảnh](frames/B_main/t6_e8_s1001.jpg) |
| e9 | 1001 | 1 | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 0.27 | A | [ảnh](frames/B_main/t6_e9_s1001.jpg) |
| e0 | 1002 | 1 | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 0.09 | A: cốc trên đĩa; lơ lửng/cầm ở pudding | [ảnh](frames/B_main/t6_e0_s1002.jpg) |
| e3 | 1002 | 1 | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 0.24 | B: tay vẫn ôm cốc trên đĩa | [ảnh](frames/B_main/t6_e3_s1002.jpg) |
| e4 | 1002 | 1 | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 0.04 | A | [ảnh](frames/B_main/t6_e4_s1002.jpg) |
| e5 | 1002 | 1 | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 0.25 | B: tay vẫn ôm cốc trên đĩa | [ảnh](frames/B_main/t6_e5_s1002.jpg) |
| e8 | 1002 | 0 | F2 làm đổ / kẹp sai tư thế | khung điển hình | 0.87 | C: làm đổ cốc đỏ | [ảnh](frames/B_main/t6_e8_s1002.jpg) |
| e9 | 1002 | 1 | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 0.17 | A | [ảnh](frames/B_main/t6_e9_s1002.jpg) |

### Task 7

| Episode | Seed | Giai đoạn | Nguyên nhân | Bằng chứng | Chuyển động | Ghi chú | Ảnh |
|---|---|---|---|---|---|---|---|
| e1 | 1000 | ? | F6 ra khỏi vùng làm việc | khung điển hình | 0.39 | tay/vật ra khỏi khung | [ảnh](frames/B_main/t7_e1_s1000.jpg) |
| e3 | 1000 | 0 | F2 làm đổ / kẹp sai tư thế | khung điển hình | 0.57 | A1 | [ảnh](frames/B_main/t7_e3_s1000.jpg) |
| e5 | 1000 | 0 | F1 gắp hỏng | khung điển hình | 0.83 | A2 | [ảnh](frames/B_main/t7_e5_s1000.jpg) |
| e8 | 1000 | 1 | F4 đặt không đúng / gần đích | khung điển hình | 0.27 | B: hộp cream cheese ở mép giỏ | [ảnh](frames/B_main/t7_e8_s1000.jpg) |
| e9 | 1000 | 1 | F4 đặt không đúng / gần đích | khung điển hình | 0.29 | B | [ảnh](frames/B_main/t7_e9_s1000.jpg) |
| e3 | 1001 | 0 | F2 làm đổ / kẹp sai tư thế | khung điển hình | 2.09 | A3: lon bị cầm nằm ngang ở mép giỏ | [ảnh](frames/B_main/t7_e3_s1001.jpg) |
| e5 | 1001 | 0 | F2 làm đổ / kẹp sai tư thế | khung điển hình | 0.93 | A1 | [ảnh](frames/B_main/t7_e5_s1001.jpg) |
| e7 | 1001 | ? | F6 ra khỏi vùng làm việc | khung điển hình | 0.20 | tay/vật ra khỏi khung | [ảnh](frames/B_main/t7_e7_s1001.jpg) |
| e8 | 1001 | 1 | F4 đặt không đúng / gần đích | khung điển hình | 1.22 | B: hộp cream cheese ở mép giỏ | [ảnh](frames/B_main/t7_e8_s1001.jpg) |
| e0 | 1002 | 0 | F2 làm đổ / kẹp sai tư thế | khung điển hình | 0.20 | A1: lon xanh đổ / cầm nghiêng | [ảnh](frames/B_main/t7_e0_s1002.jpg) |
| e3 | 1002 | 0 | F1 gắp hỏng | khung điển hình | 0.20 | A2: lon xanh vẫn đứng, tay gần/đang kẹp | [ảnh](frames/B_main/t7_e3_s1002.jpg) |
| e5 | 1002 | 0 | F1 gắp hỏng | khung điển hình | 1.26 | A2 | [ảnh](frames/B_main/t7_e5_s1002.jpg) |
| e6 | 1002 | 0 | F2 làm đổ / kẹp sai tư thế | khung điển hình | 1.16 | A1 | [ảnh](frames/B_main/t7_e6_s1002.jpg) |
| e8 | 1002 | 1 | F4 đặt không đúng / gần đích | khung điển hình | 1.48 | B: hộp cream cheese ở mép giỏ | [ảnh](frames/B_main/t7_e8_s1002.jpg) |

### Task 8

| Episode | Seed | Giai đoạn | Nguyên nhân | Bằng chứng | Chuyển động | Ghi chú | Ảnh |
|---|---|---|---|---|---|---|---|
| e0 | 1000 | 1 | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 2.33 | A4: tay vẫn kẹp ấm trên bếp | [ảnh](frames/B_main/t8_e0_s1000.jpg) |
| e1 | 1000 | 0 | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 0.05 | B2: cầm ấm thẳng đứng giữa bàn | [ảnh](frames/B_main/t8_e1_s1000.jpg) |
| e2 | 1000 | 0 | F2 làm đổ / kẹp sai tư thế | khung điển hình | 1.75 | B1 | [ảnh](frames/B_main/t8_e2_s1000.jpg) |
| e3 | 1000 | 1 | F2 làm đổ / kẹp sai tư thế | khung điển hình | 0.70 | A1: ấm 2 bị kẹp nằm ngang | [ảnh](frames/B_main/t8_e3_s1000.jpg) |
| e4 | 1000 | 1 | F2 làm đổ / kẹp sai tư thế | khung điển hình | 0.53 | A1 | [ảnh](frames/B_main/t8_e4_s1000.jpg) |
| e5 | 1000 | 1 | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 0.12 | A2 | [ảnh](frames/B_main/t8_e5_s1000.jpg) |
| e6 | 1000 | 1 | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 1.69 | A3: ấm 2 vẫn trên bàn | [ảnh](frames/B_main/t8_e6_s1000.jpg) |
| e7 | 1000 | 0 | F2 làm đổ / kẹp sai tư thế | khung điển hình | 0.05 | B1 | [ảnh](frames/B_main/t8_e7_s1000.jpg) |
| e8 | 1000 | 1 | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 0.13 | A2 | [ảnh](frames/B_main/t8_e8_s1000.jpg) |
| e0 | 1001 | 0 | F2 làm đổ / kẹp sai tư thế | khung điển hình | 0.10 | B1: ấm đổ / cầm nghiêng | [ảnh](frames/B_main/t8_e0_s1001.jpg) |
| e1 | 1001 | 1 | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 0.06 | A2: cầm ấm 2 thẳng đứng, không đặt | [ảnh](frames/B_main/t8_e1_s1001.jpg) |
| e2 | 1001 | 0 | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 0.07 | B2 | [ảnh](frames/B_main/t8_e2_s1001.jpg) |
| e3 | 1001 | 1 | F2 làm đổ / kẹp sai tư thế | khung điển hình | 0.28 | A1: ấm 2 bị kẹp nằm ngang | [ảnh](frames/B_main/t8_e3_s1001.jpg) |
| e4 | 1001 | 0 | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 0.09 | B2 | [ảnh](frames/B_main/t8_e4_s1001.jpg) |
| e5 | 1001 | 1 | F2 làm đổ / kẹp sai tư thế | khung điển hình | 0.48 | A1 | [ảnh](frames/B_main/t8_e5_s1001.jpg) |
| e6 | 1001 | 1 | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 1.74 | A2 | [ảnh](frames/B_main/t8_e6_s1001.jpg) |
| e7 | 1001 | 0 | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 0.16 | B2 | [ảnh](frames/B_main/t8_e7_s1001.jpg) |
| e8 | 1001 | 1 | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 2.35 | A3 | [ảnh](frames/B_main/t8_e8_s1001.jpg) |
| e9 | 1001 | 1 | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 2.72 | A3 | [ảnh](frames/B_main/t8_e9_s1001.jpg) |
| e0 | 1002 | 0 | F2 làm đổ / kẹp sai tư thế | khung điển hình | 1.56 | B1: ấm đổ / cầm nghiêng | [ảnh](frames/B_main/t8_e0_s1002.jpg) |
| e1 | 1002 | 0 | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 0.04 | B2: cầm ấm thẳng đứng giữa bàn | [ảnh](frames/B_main/t8_e1_s1002.jpg) |
| e2 | 1002 | 0 | F2 làm đổ / kẹp sai tư thế | khung điển hình | 0.65 | B1 | [ảnh](frames/B_main/t8_e2_s1002.jpg) |
| e3 | 1002 | 1 | F2 làm đổ / kẹp sai tư thế | khung điển hình | 0.30 | A1: ấm 2 bị kẹp nằm ngang | [ảnh](frames/B_main/t8_e3_s1002.jpg) |
| e4 | 1002 | 1 | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 0.18 | A2 | [ảnh](frames/B_main/t8_e4_s1002.jpg) |
| e5 | 1002 | 1 | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 0.09 | A2 | [ảnh](frames/B_main/t8_e5_s1002.jpg) |
| e6 | 1002 | 0 | F2 làm đổ / kẹp sai tư thế | khung điển hình | 0.59 | B1 | [ảnh](frames/B_main/t8_e6_s1002.jpg) |
| e7 | 1002 | 0 | F2 làm đổ / kẹp sai tư thế | khung điển hình | 0.58 | B1 | [ảnh](frames/B_main/t8_e7_s1002.jpg) |
| e8 | 1002 | 1 | F2 làm đổ / kẹp sai tư thế | khung điển hình | 0.48 | A1 | [ảnh](frames/B_main/t8_e8_s1002.jpg) |
| e9 | 1002 | 0 | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 0.06 | B2 | [ảnh](frames/B_main/t8_e9_s1002.jpg) |

### Task 9

| Episode | Seed | Giai đoạn | Nguyên nhân | Bằng chứng | Chuyển động | Ghi chú | Ảnh |
|---|---|---|---|---|---|---|---|
| e7 | 1000 | 0 | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 0.18 | cầm cốc trước cửa lò, chưa đưa vào | [ảnh](frames/B_main/t9_e7_s1000.jpg) |
| e8 | 1000 | 0 | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 0.20 |  | [ảnh](frames/B_main/t9_e8_s1000.jpg) |
| e9 | 1000 | 0 | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 0.32 |  | [ảnh](frames/B_main/t9_e9_s1000.jpg) |
| e7 | 1001 | 0 | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 0.16 | cầm cốc trước cửa lò, chưa đưa vào | [ảnh](frames/B_main/t9_e7_s1001.jpg) |
| e9 | 1001 | 0 | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 0.24 |  | [ảnh](frames/B_main/t9_e9_s1001.jpg) |
| e7 | 1002 | 0 | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 0.29 | cầm cốc trước cửa lò, chưa đưa vào | [ảnh](frames/B_main/t9_e7_s1002.jpg) |
| e8 | 1002 | 0 | F3 đứng yên / lơ lửng không tiến triển | khung điển hình | 0.42 |  | [ảnh](frames/B_main/t9_e8_s1002.jpg) |
