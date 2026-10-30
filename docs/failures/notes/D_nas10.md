# D_nas10: libero_10, n_action_steps = 10, relative, seed 1000, 50 episode/task

Chạy ngày 31/10/2026 bằng `TAG=D_nas10 NAS=10 CM=relative N_EP=50 SEEDS=1000 bash run_queue.sh`, giống hệt D_nas1
và D_nas50 trừ `--policy.n_action_steps=10` (`cmd.txt`). Thời gian thực: 1597 s (tổng `seconds=` trong `status.txt`).
Lý do chọn 10: `lerobot/docs/source/libero.mdx:205` dùng `n_action_steps=10` cho **Pi0.5** ("matching the original
OpenPI implementation"); tài liệu không khuyến nghị giá trị này cho SmolVLA, nên đây là một phép thử thêm.
Chỉ có video cho episode 0–9 mỗi task: 28 trên tổng 132 thất bại. Nhãn chỉ dựa trên khung điển hình.

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
