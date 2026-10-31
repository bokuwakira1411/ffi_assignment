# LIBERO-Spatial: chỉ kết quả cuối

> `failure_analysis.md` đã xoá khỏi repo ngày 31/10/2026; bản cuối: `git show e852bb7:failure_analysis.md`.

Theo yêu cầu (31/10/2026), LIBERO-Spatial chỉ dùng để kiểm tra checkpoint có chạy đúng hay không, nên ở đây
chỉ báo tỉ lệ thành công, không phân tích thất bại. Phân tích thất bại đã làm trước đó vẫn nằm ở
`e852bb7:failure_analysis.md` mục 5 và 6.3–6.4.

Nguồn: `per_task[0].n_success` trong `eval_info.json` của từng job, cộng theo task. Mọi run dùng
`lerobot/smolvla_libero`, giới hạn 280 bước/episode.

| Task | F_spatial (NAS=1, relative, 3 seed x 10) | F_spatial_nas50 (NAS=50, relative, seed 1000 x 10) | F_spatial_abs (NAS=1, absolute, seed 1000 x 10) | F_spatial_nas10 (NAS=10, relative, seed 1000 x 10) |
|---|---|---|---|---|
| 0 | 29/30 | 10/10 | 1/10 | 10/10 |
| 1 | 24/30 | 5/10 | 1/10 | 9/10 |
| 2 | 30/30 | 7/10 | 0/10 | 9/10 |
| 3 | 28/30 | 7/10 | 0/10 | 10/10 |
| 4 | 20/30 | 4/10 | 0/10 | 7/10 |
| 5 | 0/30 | 0/10 | 0/10 | 1/10 |
| 6 | 29/30 | 6/10 | 0/10 | 10/10 |
| 7 | 26/30 | 6/10 | 0/10 | 9/10 |
| 8 | 23/30 | 5/10 | 0/10 | 7/10 |
| 9 | 28/30 | 8/10 | 0/10 | 10/10 |
| **Tổng** | **237/300 = 79,0%** | **58/100** | **2/100** | **82/100** |

So với số của paper SmolVLA 0.45B cho Spatial, 90% [CHƯA KIỂM CHỨNG: lấy từ `differences.md` mục 3, PDF không có
trên máy]: F_spatial thấp hơn 11 điểm. Riêng task 5 ("pick up the black bowl on the ramekin...") thất bại 30/30;
nếu bỏ task 5, 9 task còn lại đạt 237/270 = 87,8%. Kết luận về checkpoint: chạy đúng với cấu hình NAS=1,
relative; cấu hình absolute không dùng được (2/100, `e852bb7:failure_analysis.md` mục 6.4).

F_spatial_nas10 chạy ngày 31/10/2026 (424 s). 82/100 so với 78/100 của F_spatial cùng seed 1000
(`differences.md` mục 3b): chênh 4 điểm trên 100 episode, nằm trong sai số (Wilson 95% của 82/100 là 73,3–88,3).
Task 5 vẫn gần như thất bại hoàn toàn (1/10).
