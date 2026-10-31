# D_nas50: libero_10, n_action_steps = 50, relative, seed 1000, 50 episode/task

> `failure_analysis.md` đã xoá khỏi repo ngày 31/10/2026; bản cuối: `git show e852bb7:failure_analysis.md`.

Cấu hình (`cmd.txt`): giống D_nas1, chỉ khác `--policy.n_action_steps=50` (giá trị trong config của checkpoint).
Chỉ có video cho episode 0–9 của mỗi task, nên bảng nguyên nhân chỉ gồm 48 trên tổng 261 thất bại.
Phân tích chi tiết, gồm 22 cặp cùng cảnh mà NAS=1 thành công: `e852bb7:failure_analysis.md` mục 6.5. Nhóm N1–N7 ở đó
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
