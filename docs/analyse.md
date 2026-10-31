# FFI26 – Analyse: điểm mạnh và giới hạn của SmolVLA (`lerobot/smolvla_libero`)

> `failure_analysis.md` đã xoá khỏi repo ngày 31/10/2026; bản cuối: `git show e852bb7:failure_analysis.md`.

Cấu hình chính: libero_10, NAS = 10, relative (`docs/reproduce.md` §4). Nguồn dữ liệu:

| Viết tắt | Dữ liệu | Cách gán nhãn |
|---|---|---|
| **D10** | D_nas10: 10 task × 50 init × 3 seed (`docs/failure_overview.md` §1) | tỉ lệ thành công; nhãn F0–F6 bằng mắt cho 28 thất bại có video (seed 1000, ep 0–9) |
| **D1 / D50 / B** | D_nas1, D_nas50, B_main (NAS = 1, 50, 1) (`docs/failure_overview.md`, `e852bb7:failure_analysis.md`) | nhãn F0–F6 bằng mắt (44 / 48 / 142 thất bại) |
| **LOG** | checkpoint chính thức, NAS = 10, có log predicate BDDL và vị trí vật mỗi bước: task 0, 8 × 50 ep; 8 task khác × 20 ep (`ffi26/ffi26/results/round2/episodes.csv`) | nhãn tự động A–E, O (`improve/src/failure_labels.py`); sự kiện slip/collision (`improve/src/reward_r2.py`) |

Nhãn bằng mắt do Claude gán, một người, phần lớn chỉ dựa trên khung điển hình (`docs/failure_overview.md` §5).
Nhãn tự động dùng ngưỡng tự chọn. Detector collision đã được kiểm bằng ảnh (`results/round2/detector_check/`); detector slip chưa.

## 1. Điểm mạnh

| # | Điểm mạnh | Bằng chứng |
|---|---|---|
| S1 | **Task một bước hoặc có vật khớp nối làm rất tốt** | D10 (/150): task 3 "bát vào ngăn kéo rồi đóng" 146, task 2 "bật bếp + đặt ấm" 138, task 1 136, task 5 135. Spatial: 9/10 task đạt 237/270 = 87,8% khi bỏ task 5 (`docs/failures/spatial_results.md`) |
| S2 | **Hiểu đúng câu lệnh: hiếm khi nhầm vật hay nhầm đích** | Nhầm vật/đích (F5) chỉ 1–2 thất bại mỗi run: B 1/142, D10 1/28. LOG: nhãn C (nhầm vật) là nhãn chính ở 4/90 thất bại |
| S3 | **Action chunk + quan sát lại mỗi 10 bước: vừa mượt vừa sửa được sai** | NAS=10 đạt 73,6%, hơn NAS=1 (54,6%) ở cả 10 task và hơn NAS=50 (47,8%) (`docs/failure_overview.md` §1). Cùng cảnh: NAS=10 cứu 10/16 lần gắp hỏng (F1) của NAS=50 và 7/12 lần ra khỏi vùng làm việc (F6) của NAS=1. LOG: 14/170 episode thành công có ≥ 1 lần trượt vật (slip) rồi gắp lại thành công, ví dụ task 0 init 31 (2 lần) |
| S4 | **Nhỏ và nhanh** | 450M tham số, 96,5 ms / chunk trên RTX 5070 Ti (`reproduce.md` §2.3). Một chunk 50 action dùng được cho nhiều bước: 10 job D_nas1 mất 6.309 s, D_nas10 1.597 s, D_nas50 1.336 s (`differences.md` §3b, `docs/failures/D_nas10.md`). Train/eval/RL đều chạy được trên 1 GPU 16 GB |

## 2. Giới hạn

| # | Giới hạn | Bằng chứng |
|---|---|---|
| L1 | **Task dài nhiều bước: xong bước con thứ nhất rồi không làm tiếp** | Hai task yếu nhất là task hai vật: task 0 60/150, task 8 71/150 (D10). LOG task 8: 19/27 thất bại xong đúng một nửa goal (một ấm trên bếp). LOG task 0: 9/29 thất bại xong một lon. B: 47 thất bại "xong bước 1, hỏng bước 2"; 91/142 thất bại có chuyển động 100 bước cuối < 0,3, tức robot dừng hẳn từ trước khi hết giờ (`e852bb7:failure_analysis.md` §2.3, §4) |
| L2 | **Thao tác tinh kém: gắp trượt, làm đổ, kẹp sai tư thế, đặt chưa đủ chính xác** | LOG task 0: 25/29 thất bại là A (gắp hỏng); 20/29 có slip, so với 5/21 episode thành công. Collision (xô lệch vật khác) xảy ra 12 lần trong thất bại task 0, 0 lần trong thành công. D10: F2 (làm đổ / kẹp sai) là nguyên nhân lớn nhất, 9/28; task 8 hay kẹp ấm thứ hai nằm ngang (B: 6 ep). Spatial task 5 (bát chồng trên ramekin) 30/30 thất bại: 25 lần gạt bát rơi rồi dừng (`e852bb7:failure_analysis.md` §5.2). Đặt gần đích mà env không tính (F4) 5/28; predicate `On` cần tâm hai vật lệch < 3 cm (`docs/failure_overview.md` §4.1) |
| L3 | **Rất nhạy với cách thực thi, mà giá trị mặc định lại tệ nhất** | Cùng checkpoint, NAS 1 / 10 / 50: 54,6% / 73,6% / 47,8%; NAS = 50 là giá trị trong config. absolute thay relative: 2/100, và 2 lần "thành công" là do va chạm (`e852bb7:failure_analysis.md` §6.4). Hai kiểu lỗi ngược nhau: NAS=1 hay dừng hoặc đi lạc (F3 + F6 = 26/44 ở D1), NAS=50 gắp trượt rồi vẫn làm tiếp (F1 16/48) |
| L4 | **Kết quả do cảnh quyết định, ít đa dạng khi lấy mẫu** | 500 cảnh × 3 seed: 289 cảnh thành công cả 3, 69 thất bại cả 3, chỉ 142 cho kết quả lẫn lộn (D10). B: 111/142 thất bại thuộc 37 cảnh thất bại cả 3 seed (`e852bb7:failure_analysis.md` §2.2). Đổi nhiễu ε chỉ đổi chunk 0,044 (trung bình, đơn vị chuẩn hoá; độ lệch chuẩn của chunk 0,77) (`improve/outputs/fpo/smoke/verify.json`). Hệ quả cho RL: thử lại ít khi ra kết quả khác; khi thu 8 rollout / cảnh, 8/32 cảnh thất bại cả 8 lần (`improve/outputs/r2/round1/groups.csv`) |
| L5 | **Khó fine-tune thêm bằng recipe mặc định** | Action expert được nạp ở bf16. Khi fine-tune bằng `lerobot-train` với lr 1e-5, bước cập nhật nhỏ hơn độ phân giải bf16: sau 6.000 step chỉ 5,5% phần tử của expert (96,6M) từng thay đổi, phần fp32 thay đổi 97–100% (`results/round2/fpo_plan.md` §9). Không biến thể fine-tune nào ở round 1 tốt hơn checkpoint gốc (`results/round2/summary.md`) |

## 3. Quỹ đạo thành công và thất bại

| | Thành công | Thất bại |
|---|---|---|
| Độ dài | B: 151 / 245 / 456 bước (min / trung vị / max). LOG task 0: 249 / 266 / 432; task 8: 350 / 391 / 478 | luôn đúng giới hạn 520 bước (hết giờ), không có thất bại sớm (`e852bb7:failure_analysis.md` §2.1) |
| Hành vi | các bước con nối tiếp không dừng. Ví dụ D1 task 1 ep 1: nhấc hộp cream cheese ở bước ~50–74, vào giỏ ~150–170. D50 task 8 ep 0: ấm 1 lên bếp ~159, ấm 2 ~372 | dừng hoặc lơ lửng sau bước con đầu. Ví dụ B task 0 ep 1 seed 1000: lon xanh vào giỏ ~222, sau đó cầm lon đỏ lơ lửng từ ~297 tới 519 (`docs/failures/frames/B_main/t0_e1_s1000.jpg`) |
| Phục hồi | gắp lại sau khi trượt (LOG: 14 episode) | có thử lại nhưng không xong: Spatial task 9 ep 5 seed 1001 làm rơi bát cạnh đĩa rồi gắp lại, bát vẫn nghiêng ở mép đĩa (`e852bb7:failure_analysis.md` §5.1) |
| Lỗi thao tác | – | kẹp ấm thứ hai nằm ngang (`frames/B_main/t8_e3_s1000.jpg`); ôm cốc trên đĩa tới hết giờ (`frames/B_main/t6_e3_s1000.jpg`); gắp trượt rồi mang tay không tới giỏ (D50 task 1 ep 1, `e852bb7:failure_analysis.md` §6.5) |

Kết luận: điểm yếu chính nằm ở chỗ **chuyển sang bước con tiếp theo** và **độ chính xác khi gắp/đặt**, không nằm ở
hiểu câu lệnh. Đổi NAS sửa được phần lỗi do cách thực thi (S3), còn L1 và L2 thì không. Vì vậy phần Improve nhắm vào
task 0 và task 8.
