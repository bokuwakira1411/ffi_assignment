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
