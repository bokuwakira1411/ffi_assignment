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
