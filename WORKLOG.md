# Nhật ký làm việc của Claude

Mỗi lần Claude chạy code hoặc lệnh trên repo này (`~/Lan_K68AI/temp`) hoặc thư mục kết quả (`~/ffi26`), Claude
ghi thêm một mục vào file này **trước khi** báo lại cho bạn. Mỗi mục ghi: mục tiêu, kế hoạch, từng lệnh và lý do,
file đã đọc, file đã ghi hoặc sửa, và kết quả.

Chú thích cột "Tác động":
- **chỉ đọc**: chỉ đọc file, không thay đổi gì.
- **scratch**: chỉ ghi vào thư mục nháp của phiên làm việc
  `/tmp/claude-1003/-home-student-Lan-K68AI-temp/92c565a4-1d53-4fbd-9f85-01bf079683ec/scratchpad/` (nằm ngoài repo).
- **SỬA REPO**: thay đổi một file trong repo này hoặc trong `~/ffi26`.

---

## Phiên 30/10/2026

### Việc 1: điền `differences.md` từ các file kết quả (khoảng 17:30–17:50)

Kế hoạch báo trước cho bạn: không có. Đây là lỗi của Claude. Từ nay kế hoạch được báo trước (xem mục "Quy tắc" ở cuối).

| # | Lệnh / thao tác | Lý do | Tác động |
|---|---|---|---|
| 1 | `ls`, `find` trong `~/ffi26` | xem cấu trúc thư mục | chỉ đọc |
| 2 | `cat differences.md readme.md DAY1.md run_steps train_more.sh` | hiểu "differences" là gì (là template có các ô `[ĐIỀN]`) | chỉ đọc |
| 3 | `cat results/versions.txt env_report.json`, `grep` trong `pip_freeze.txt` | phiên bản LeRobot, torch, driver, GPU | chỉ đọc |
| 4 | `cat` các file `runs/*/summary_*.csv`, `timing.csv`, `timing.csv.bak` | các bảng tổng hợp và thời gian chạy đã có | chỉ đọc |
| 5 | `cat run_queue.sh collect_results.py run_D.sh smoke_test.sh` | ý nghĩa các cột CSV và các cờ đã dùng | chỉ đọc |
| 6 | lệnh `python3 -I` một dòng, in cấu trúc của một file `eval_info.json` | tìm hiểu cấu trúc JSON | chỉ đọc |
| 7 | viết `scratchpad/agg.py`, chạy `python3 -I agg.py ~/ffi26/results/runs` | tính lại tỉ lệ thành công của mọi job từ `eval_info.json` thô (hóa ra các file summary đã cũ) | scratch (script); chỉ đọc với thư mục kết quả |
| 8 | `cat`/`python3 -I` trên `smoke_*`, `demo_*`, `ckpt_files/*.json`, `probe_report.json`, `logs/D_full.log` | kết quả smoke/demo, config, kết quả probe | chỉ đọc |
| 9 | `grep` trong các file `eval.log` | giới hạn bước, config do lerobot-eval in ra | chỉ đọc |
| 10 | đọc `config.json` của `lerobot/smolvla_libero` và `lerobot/smolvla_base` trong `~/.cache/huggingface/hub` | kiểm chứng các trường của recipe và lỗi action 6 chiều so với 7 chiều | chỉ đọc |
| 11 | `grep`/`sed -n` trên `~/ffi26/lerobot/src/lerobot/envs/libero.py` và `scripts/lerobot_eval.py`; `git rev-parse HEAD` | giới hạn bước, cách kiểm tra thành công, cách gán init state, giới hạn số video | chỉ đọc (chỉ đọc mã nguồn, không chạy) |
| 12 | script `python3 -I` so sánh từng episode (B_main với D_nas1, giữa các seed) | kiểm tra tính tất định và việc các seed dùng chung init state | chỉ đọc |
| 13 | script `python3 -I` tính CI Wilson và kiểm định McNemar | số liệu thống kê cho báo cáo | chỉ đọc |
| 14 | `conda activate lerobot` + `python -c "from libero.libero import benchmark ..."` (chạy trong thư mục scratchpad) | lấy tên task của libero_10 / libero_spatial | chỉ đọc. Lệnh này **chạy code của gói `hf_libero` đã cài**, không phải code của repo này |
| 15 | **Ghi `differences.md`** (viết lại toàn bộ template) | sản phẩm bạn yêu cầu | **SỬA REPO**. Bản cũ còn trong commit git `3d7a55f` (`git diff differences.md` cho thấy thay đổi) |
| 16 | 3 lần Edit + 2 lần `sed -i` trên `differences.md` | sửa lỗi của chính Claude: 98→99/100 episode trùng, một số dòng code trích dẫn, 110→100 job, câu chữ về smoke test | **SỬA REPO** |

Không script nào của repo này (`run_queue.sh`, `collect_results.py`, `smoke_test.sh`, `train_more.sh`, ...) được chạy.
Không file nào trong `~/ffi26` bị sửa.

### Việc 2: phân tích các ca thất bại từ video (khoảng 18:20–18:31, bạn đã dừng giữa chừng)

Kế hoạch báo trước cho bạn: không có (cùng lỗi trên). Bạn đã dừng công việc. Kế hoạch được báo sau đó và **đang chờ bạn duyệt**.

| # | Lệnh / thao tác | Lý do | Tác động |
|---|---|---|---|
| 1 | `ffprobe` trên một video, `which ffmpeg` | định dạng video: 360×360, mỗi bước env một khung hình | chỉ đọc |
| 2 | viết `scratchpad/vidindex.py`, chạy → `scratchpad/vids.csv` | liệt kê toàn bộ 1.000 video kèm cờ thành công và số khung hình | scratch; chỉ đọc với thư mục kết quả |
| 3 | viết `scratchpad/sheet.py` (dòng thời gian) và `scratchpad/grid.py` (lưới khung hình cuối, có tùy chọn cắt ảnh) | tạo ảnh để xem | scratch |
| 4 | viết `scratchpad/motion.py`, chạy → `scratchpad/motion.csv` | chỉ số chuyển động cuối episode (phát hiện robot đứng yên lơ lửng) | scratch; chỉ đọc với thư mục kết quả |
| 5 | viết `scratchpad/mk.py`, chạy `grid.py`/`sheet.py` nhiều lần → `scratchpad/*.png` (B0…B9, F5) | ảnh tổng hợp cho task 0–9 của libero_10 và task 5 của spatial | scratch; chỉ đọc với thư mục kết quả |
| 6 | xem các file PNG | phân loại thất bại bằng mắt | chỉ đọc |

Toàn bộ Python ở trên chạy bằng `~/.conda/envs/lerobot/bin/python` (numpy, PIL) hoặc `python3 -I`, cùng `ffmpeg` của hệ thống.
Không file nào trong repo này hay trong `~/ffi26` bị sửa trong Việc 2.

Các phát hiện tạm thời nằm trong câu trả lời trên chat. Chưa ghi gì vào repo.

### Việc 3: viết nhật ký này (khoảng 18:35)

| # | Lệnh / thao tác | Lý do | Tác động |
|---|---|---|---|
| 1 | `ls -l` trên scratchpad và `differences.md`, `git status` | lấy thời điểm thật cho nhật ký | chỉ đọc |
| 2 | **Ghi `WORKLOG.md`** (file này, bản tiếng Anh) | bạn yêu cầu | **SỬA REPO** (file mới) |

### Việc 4: dịch nhật ký sang tiếng Việt

| # | Lệnh / thao tác | Lý do | Tác động |
|---|---|---|---|
| 1 | **Ghi lại `WORKLOG.md`** bằng tiếng Việt, nội dung giữ nguyên | bạn yêu cầu các file báo cáo viết bằng tiếng Việt | **SỬA REPO** |

### Việc 5: viết bản nháp `failure_analysis.md` (phương án 1 bạn đã chọn)

Kế hoạch đã báo và được bạn duyệt: viết ngay với các phát hiện đã có, không phân tích video mới. Chỉ tạo `failure_analysis.md`.

| # | Lệnh / thao tác | Lý do | Tác động |
|---|---|---|---|
| 1 | script `python3 -I` đọc `scratchpad/vids.csv` và `scratchpad/motion.csv` | đếm số cảnh ban đầu thất bại ở cả 3 seed (37/100, chiếm 111/142 thất bại) và số thất bại có chuyển động < 0,3 (91/142) | chỉ đọc (file trong scratchpad) |
| 2 | **Ghi `failure_analysis.md`** (file mới, tiếng Việt) | bạn muốn kiểm tra luận lý trước | **SỬA REPO** (file mới) |

Kết quả: bản nháp gồm phương pháp, các phát hiện đo được, phân tích 10 task libero_10, quan sát một phần của spatial task 5
(ghi rõ là chưa kết luận) và danh sách việc còn lại. Không chạy video mới, không sửa file nào trong `~/ffi26`.

### Việc 6: phân tích phần video còn lại và cập nhật `failure_analysis.md`

Kế hoạch đã báo và được bạn duyệt ("you can working on the rest videos"): (1) kiểm chứng giả thuyết spatial task 5,
(2) phân tích các task spatial còn lại, (3) so sánh video ablation. Chỉ cập nhật `failure_analysis.md` và file này.

| # | Lệnh / thao tác | Lý do | Tác động |
|---|---|---|---|
| 1 | viết lại `scratchpad/sheet.py` (thêm cắt vùng `CROP` và khoảng bước), tạo dòng thời gian 16 khung cho spatial task 5 (`e4 s1001`, `e0 s1000`, `e3 s1001`) và khung đầu ở độ phân giải đầy đủ | kiểm chứng giả thuyết "gắp nhầm ramekin / nhấc đĩa" | scratch; chỉ đọc video |
| 2 | `grid.py` cắt vùng cho 30 thất bại của spatial task 5 | đếm các nhóm | scratch |
| 3 | viết `scratchpad/pairs.py` (khung đầu cạnh khung cuối), chạy cho 33 thất bại spatial còn lại | phân loại theo vị trí bát mục tiêu | scratch |
| 4 | `sheet.py` cho 6 episode spatial đại diện | kiểm chứng cách đọc khung cuối | scratch |
| 5 | script `python3 -I` đọc `vids.csv`/`motion.csv`: tìm các cặp episode mà hai cấu hình ablation cho kết quả khác nhau; trung vị chuyển động theo run | chọn cặp để xem, số đo khách quan | chỉ đọc |
| 6 | `sheet.py` cho 2 cặp D task 8 (NAS 1/50), 2 cặp spatial (NAS 1/50), 4 episode absolute | so sánh hành vi | scratch |
| 7 | ghép mục 1–4 cũ (không đổi) với mục 5–9 mới thành `failure_analysis.md` (qua `head`/`cat` trong scratchpad); trước đó Edit phần đầu file (bảng trạng thái) | cập nhật báo cáo | **SỬA REPO** |

Kết quả chính: spatial task 5 thì 25/30 làm xáo trộn chồng bát–ramekin, 2/30 gắp nhầm ramekin (1 xác nhận bằng timeline),
1/30 gắp đúng bát nhưng không thả. Spatial còn lại: nhóm lớn nhất là mang bát tới đĩa nhưng không đặt được (14/63).
Với NAS=50, thất bại gần như không kết thúc bằng trạng thái đứng yên. Hai "thành công" của absolute là do va chạm
(25 và 60 bước, trong khi thành công ngắn nhất với relative là 71 bước). Đã bỏ một câu suy đoán về absolute vì không có
bằng chứng. Không sửa file nào trong `~/ffi26`, không chạy script nào của repo.

---

## Quy tắc Claude tuân theo từ nay trong repo này

1. Trước khi chạy bất cứ thứ gì: nêu mục tiêu và kế hoạch (các bước, file sẽ đọc, file sẽ ghi). Nếu không chỉ là đọc nhanh thì chờ bạn duyệt.
2. Sau khi chạy: ghi thêm một mục vào file này, gồm các lệnh, lý do, tác động (chỉ đọc / scratch / SỬA REPO) và kết quả.
3. Không sửa kết quả trong `~/ffi26` và không chạy các script của repo (`run_queue.sh`, `train_more.sh`, `collect_results.py`, ...) khi chưa được bạn cho phép rõ ràng.
4. Chat với bạn bằng tiếng Anh; mọi file báo cáo (md trong repo) viết bằng tiếng Việt.
