# Nhật ký làm việc của Claude

Mỗi lần Claude chạy code hoặc lệnh trên repo này (`~/Lan_K68AI/temp`) hoặc thư mục kết quả (`~/ffi26`), Claude
ghi thêm một mục vào file này **trước khi** báo lại cho bạn. Mỗi mục ghi: mục tiêu, kế hoạch, từng lệnh và lý do,
file đã đọc, file đã ghi hoặc sửa, và kết quả.

Chú thích cột "Tác động":
- **chỉ đọc**: chỉ đọc file, không thay đổi gì.
- **scratch**: chỉ ghi vào thư mục nháp của phiên làm việc
  `/tmp/claude-1003/-home-student-Lan-K68AI-temp/92c565a4-1d53-4fbd-9f85-01bf079683ec/scratchpad/` (nằm ngoài repo).
- **SỬA REPO**: thay đổi một file trong repo này hoặc trong `~/ffi26`.

## ☀ TÓM TẮT SÁNG 01/11/2026 (FPO v2 + video)

**Trạng thái:** mọi thứ chạy xong, không có lỗi. Loop `fpo_v2` kết thúc sạch lúc 03:27:25 (`loop.log` có `=== [done]`, 0 dòng Traceback/Error; đủ `selection.json` và 5 `summary.md`). Watcher thấy loop dừng lúc 03:28:25 (kiểm tra mỗi 2 phút), chạy video 03:28–03:50, mọi bước exit 0. Bảng chi tiết: mục "01/11/2026 — FPO v2: kết quả vòng lặp đầy đủ" ở cuối file (một phiên Claude khác viết; tôi đã tính lại các số chính từ `results/round2/fpo_v2/*/episodes.csv` và khớp).

**Kiểm định chính, v2 vs official** (task 0 + 8, seed 3000/4000/5000 gộp, 300 cặp, expert fp32 cho mọi nhánh, McNemar chính xác):
- v2 θ4 126/300 = 42.0%, official 121/300 = 40.3%. Chỉ v2 thành công 49, chỉ official 44, **p = 0.68 → không có khác biệt**.
- Theo task, cùng 300 cặp: task 8 tăng 76 so với 60 (38/22, p = 0.052), task 0 giảm 50 so với 61 (11/22, p = 0.080). Tổng không đổi nhưng có đánh đổi giữa hai task; cả hai đều chưa đạt p < 0.05.
- Held-out (204 cặp): 86 so với 84, p = 0.90.
- Tham chiếu: v1 θ3 98/300 = 32.7%, kém official (40/63, p = 0.030).

**Checkpoint được chọn:** θ4 = `outputs/fpo_v2/iter3/train/pretrained_model`. Lý do (`selection.json`): không có lần giảm nào > 15 điểm khi thu (giảm lớn nhất 42.6% → 36.7%, iteration 3 → 4, 5.9 điểm). Lưu ý: θ4 có tỉ lệ thu thấp nhất trong 5 checkpoint (94/256), nhưng quy tắc đăng ký trước vẫn chọn nó.

**Theo iteration** (`outputs/fpo_v2/iter_summary.csv`; mọi iteration dừng sớm vì clip_frac 10 bước cuối > 0.2):

| iter | tỉ lệ thu của θ_k | số bước train | KL proxy so với θ0 (cuối) |
|---|---|---|---|
| 0 | 97/256 = 37.9% | 23 | 0.032 |
| 1 | 106/256 = 41.4% | 27 | 0.046 |
| 2 | 101/256 = 39.5% | 16 | 0.183 |
| 3 | 109/256 = 42.6% | 18 | 0.199 |
| 4 (θ4, chỉ thu) | 94/256 = 36.7% | — | — |

Tổng cộng chỉ 84 bước train (0.19–0.32 epoch mỗi iteration). KL tăng mạnh ở iteration 2.

**Quên (8 task khác × 20, seed 1000, cặp so với official):** official 127/160 = 79.4%; v2 108/160 = 67.5% (9/28, **p = 0.003**); v1 θ3 111/160 = 69.4% (11/27, p = 0.014). Giảm nhiều nhất ở task 7 (16 → 10), task 1 (17 → 12) và task 2 (16 → 11).

**Video** (`results/videos/index.md`, 27 file mp4, 28 MB): 9 cặp held-out × 3 nhánh (official, v1 θ3, v2 θ4).
- Tính tất định: chạy lại 112 episode/nhánh cho official và v1 θ3 (mọi batch đứng trước episode được chọn). **222/224 khớp bản gốc cả success lẫn length.** 2 episode chỉ khác length, success giống nhau, và không nằm trong 9 episode có video: official s1000 task 8 init 13 (392 → 437 bước), v1 θ3 s1000 task 8 init 2 (344 → 345). Cả 18 video official/θ3 có kết quả khớp `episodes.csv`.
- v2 trên 9 cặp: 4/9 thành công (s2000 t8 init 1; s1000 t0 init 43; s2000 t0 init 20; s2000 t8 init 0). Đây là episode mới, không có bản gốc để so. v2 chạy với expert fp32, còn official/θ3 chạy bf16 như eval gốc, nên 9 episode này không so sánh trực tiếp được với hai nhánh kia.
- Cột "ghi chú" trong `index.md` sinh tự động từ nhãn thất bại và sự kiện, **chưa có ai xem video để kiểm tra**.

**Những điểm cần biết:**
- Không có bước nào thất bại.
- 2 episode lệch length (ở trên) cho thấy replay không hoàn toàn tất định: length có thể lệch, success thì không lệch trong 224 lần.
- tmux `fpo_v2_gpu` bị watcher dừng lúc 03:28:25. Các dòng `outputs/fpo_v2/gpu.csv` sau thời điểm này không có (monitor đã dừng trước khi chạy video).
- Hai bộ eval dùng độ chính xác khác nhau: các số v1 θ3 trong eval v2 (expert fp32) khác với eval v1 cũ (bf16).

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

### Việc 7: chuyển thư mục thật `ffi26` vào repo (thay cho symlink)

Bạn hỏi vì sao `ffi26/ffi26` hiện như shortcut: đó là symlink (tạo ngày 29/10 lúc 10:53) trỏ tới `/home/student/ffi26`.
Kế hoạch đã báo và được bạn duyệt ("yes"): kiểm tra đường dẫn cứng, chuyển thư mục, tạo symlink ngược, cập nhật `.gitignore`.

| # | Lệnh / thao tác | Lý do | Tác động |
|---|---|---|---|
| 1 | `ls -la`, `readlink -f`, `du`, `df`, `git ls-files`, `ps aux` | xác nhận là symlink, kích thước 604 MB, cùng ổ đĩa, không có tiến trình nào đang dùng | chỉ đọc |
| 2 | `grep -rIl "/home/student/ffi26"`; tìm `direct_url.json`/`*.pth` trong site-packages | tìm chỗ phụ thuộc đường dẫn cũ | chỉ đọc |
| 3 | `rm ffi26/ffi26` (chỉ xoá symlink); `mv /home/student/ffi26 ffi26/ffi26`; `ln -s .../temp/ffi26/ffi26 /home/student/ffi26` | đưa dữ liệu thật vào repo; giữ đường dẫn cũ hoạt động | **SỬA REPO** và `~/ffi26` (đã được duyệt) |
| 4 | tạo `.gitignore` (bỏ qua `ffi26/ffi26/lerobot/` và `ffi26/ffi26/ckpt_files/.cache/`) | `lerobot` là bản clone upstream có `.git` riêng (421 MB) | **SỬA REPO** |

Lý do cần symlink ngược: môi trường conda `lerobot` cài `lerobot` dạng editable, trỏ tới `/home/student/ffi26/lerobot`
(`~/.conda/envs/lerobot/lib/python3.12/site-packages/__editable__.lerobot-0.6.2.pth`). Ngoài ra đường dẫn cũ còn nằm trong
`cmd.txt`, `eval.log`, `eval_info.json` và `logs/*.log` (chỉ là ghi chép lịch sử, không sửa). Đã kiểm tra:
`import lerobot` vẫn chạy, file nằm tại `.../temp/ffi26/ffi26/lerobot/src/lerobot/__init__.py`.
Sau đó, theo yêu cầu của bạn ("yes"), đã commit `81bc9f4` (1607 file, gồm 1023 video mp4). Chưa push.

### Việc 8: cập nhật `readme.md`

Theo yêu cầu "update the readme file to reflect what has been done so far and what is planned next".

| # | Lệnh / thao tác | Lý do | Tác động |
|---|---|---|---|
| 1 | `cat readme.md DAY1.md run_steps`, `head train_more.sh`, `ls results/runs logs`, đọc `differences.md` mục 3–5 và `failure_analysis.md` mục 7 | lấy thông tin đã làm / chưa làm | chỉ đọc |
| 2 | `grep` các biến mặc định trong `run_queue.sh`, `collect_results.py`, `train_more.sh` | kiểm tra lệnh chạy lại ghi trong README | chỉ đọc |
| 3 | viết lại `readme.md`: tình trạng, kết quả, cấu trúc repo, cách chạy lại, kế hoạch tiếp | cập nhật README | **SỬA REPO** |

Bản README cũ (kế hoạch 29/10) vẫn còn trong git (`3d7a55f`). Phần "đề xuất của Claude" trong README đã ghi rõ là chưa
được duyệt. Chưa commit.

### Việc 9: tạo template `study.md` cho phần Study

Theo yêu cầu "create a file template for Study Part".

| # | Lệnh / thao tác | Lý do | Tác động |
|---|---|---|---|
| 1 | tạo `study.md`: quy tắc điền, literature matrix (cột theo `DAY1.md` mục E, 8 bài theo mục D), mẫu ghi chú từng bài, dàn ý diffusion vs flow matching, bảng nối với kết quả, checklist | template cho phần Study | **SỬA REPO** (file mới) |

Không điền nội dung bài báo nào: chỉ có tên bài và mã arXiv lấy từ `DAY1.md`. Chưa commit.

### Việc 10: phân tích thất bại của D_nas50 (libero_10)

Bạn hỏi phân tích thất bại của D_nas50 nằm ở đâu. Kiểm tra cho thấy `failure_analysis.md` mục 6 chỉ có số đo chuyển
động (6.1) và task 8, nơi NAS=50 tốt hơn (6.2). Chưa có episode thất bại nào của D_nas50 được xem bằng mắt.
Kế hoạch đã báo và được bạn duyệt ("yes").

| # | Lệnh / thao tác | Lý do | Tác động |
|---|---|---|---|
| 1 | đọc `n_success` trong `eval_info.json` của D_nas1 và D_nas50 (10 task, seed 1000) | so sánh theo task | chỉ đọc |
| 2 | `idx.py`: cờ thành công của episode 0–9 cả hai run, số khung, chuyển động | tìm 48 thất bại có video và 22 cặp NAS=1 thành công / NAS=50 thất bại | chỉ đọc + scratch |
| 3 | `pair.py` cho 22 cặp; `grid.py` cho 26 thất bại còn lại; `crop.py` cho ngăn kéo (task 3), đĩa (task 4), giỏ (task 0) | xem bằng mắt | chỉ đọc + scratch |
| 4 | đọc lại `motion.csv` của phiên trước | kiểm tra số 1,27 ở mục 6.1 | chỉ đọc |
| 5 | thêm mục 6.5 vào `failure_analysis.md`; sửa bảng phạm vi (đầu file), mục 7 ý 6, mục 8, mục 9 | ghi kết quả | **SỬA REPO** |

Kết quả: tổng D_nas1 273/500, D_nas50 239/500; NAS=50 kém hơn ở 8/10 task. 48 thất bại có video, tất cả 520 bước.
Kiểu thất bại mới: gắp trượt rồi vẫn làm tiếp với tay không (10/48). Kiểu lớn nhất vẫn là hỏng ở bước con thứ hai (21/48).
Kiểm tra số liệu: chỉ số chuyển động tính lại bằng OpenCV cho trung vị 1,51, khác 1,27 của mục 6.1. Nguyên nhân là
cách thu nhỏ ảnh khác nhau. Chạy lại theo `motion.csv` gốc (ffmpeg) thì ra đúng 1,27, nên giữ 1,27. Số thất bại
< 0,3 khớp ở cả hai cách (10). Thất bại của D_nas1 vẫn chưa được xem bằng mắt. Không sửa gì trong
`ffi26/ffi26/results`, không chạy script nào của repo. Chưa commit.

### Việc 11: ước lượng thời gian cho NAS=10 và phân tích thất bại theo từng run

Theo yêu cầu "Evaluate total runtime before starting". Chưa chạy eval, chưa sửa file kết quả.

| # | Lệnh / thao tác | Lý do | Tác động |
|---|---|---|---|
| 1 | `grep n_action_steps` trong `lerobot/docs`; WebFetch trang model card `lerobot/smolvla_libero` | kiểm tra khuyến nghị NAS=10 | chỉ đọc |
| 2 | `cat run_D.sh`; tìm `n_action_steps=10` trong `cmd.txt`; `nvidia-smi` | xem đã có run NAS=10 chưa, GPU có rảnh không | chỉ đọc |
| 3 | cộng `seconds=` trong `status.txt` của mọi job | ước lượng thời gian | chỉ đọc |

Kết quả: khuyến nghị `n_action_steps=10` trong `lerobot/docs/source/libero.mdx:205` là cho **Pi0.5** (theo OpenPI), không
phải SmolVLA; model card không nhắc tới. Chưa có run NAS=10 nào. GPU rảnh (0 %). Thời gian thực tế (tổng `seconds=`):
D_nas1 6309 s, D_nas50 1336 s; F_spatial seed 1000 khoảng 774 s, F_spatial_nas50 382 s.
Ước lượng đã báo: D_nas10 khoảng 30 phút, F_spatial_nas10 khoảng 7 phút. Bạn duyệt: chạy cả hai; Spatial chỉ báo kết quả
cuối; reward chỉ viết thiết kế; commit ảnh cùng các file khác.

### Việc 12: chạy NAS=10, trích khung thất bại, phân loại theo run, file tổng hợp và thiết kế reward

| # | Lệnh / thao tác | Lý do | Tác động |
|---|---|---|---|
| 1 | `TAG=D_nas10 NAS=10 CM=relative N_EP=50 SEEDS=1000 bash run_queue.sh`, rồi `TAG=F_spatial_nas10 SUITE=libero_spatial NAS=10 CM=relative N_EP=10 SEEDS=1000 bash run_queue.sh` (log: `ffi26/ffi26/logs/nas10_*.log`) | phép thử NAS=10 | **GPU**, thư mục kết quả mới `results/runs/D_nas10`, `F_spatial_nas10`. 20/20 job rc=0; 1597 s và 424 s |
| 2 | tạo `docs/failures/tools/extract_frames.py`, chạy cho B_main, D_nas1, D_nas50, D_nas10 | ảnh điển hình + chỉ số chuyển động cho mọi thất bại có video | **SỬA REPO**: `docs/failures/frames/` (262 ảnh + index.csv) |
| 3 | xem trang ảnh (`sheet.py`), gán nhãn F0–F6 + giai đoạn | phân loại | **SỬA REPO**: `docs/failures/labels/*.csv` |
| 4 | `make_run_report.py` + `docs/failures/notes/*.md` | file theo từng run | **SỬA REPO**: `docs/failures/{B_main,D_nas1,D_nas10,D_nas50}.md`, `spatial_results.md` |
| 5 | đọc BDDL goal của libero_10, `base_predicates.py`, `base_object_states.py` | căn cứ cho thiết kế reward | chỉ đọc |
| 6 | viết `docs/failure_overview.md` | so sánh cấu hình, nguyên nhân chung, reward | **SỬA REPO** |
| 7 | sửa `failure_analysis.md`: trung vị chuyển động B_main 0,23 → 0,21 (mục 5); câu "N1 chưa thấy ở B_main" (mục 6.5.3); bảng phạm vi; mục 8 | sửa sai và cập nhật | **SỬA REPO** |

Kiểm tra số liệu: `extract_frames.py` tái lập đúng trung vị của `motion.csv` cũ (D_nas50 1,27; D_nas1 0,23; B_main 0,213)
và số thất bại < 0,3 (10, 26, 91). Số 0,23 ghi cho libero_10 ở mục 5 của `failure_analysis.md` là của D_nas1, không
phải B_main, nên đã sửa. Ước lượng thời gian: D_nas10 dự kiến ~1740 s, thực tế 1597 s; F_spatial_nas10 dự kiến ~418 s, thực tế 424 s.

Kết quả chính: D_nas10 368/500 = 73,6% (Wilson 69,6–77,3), so với D_nas1 54,6% và D_nas50 47,8%; tốt hơn NAS=1 ở cả 10 task.
F_spatial_nas10 82/100.

Đã commit `5839ccf` (590 file, chỉ các file của việc này và cặp chuyển `WORKLOG.md` → `docs/WORKLOG.md`). Chưa push.
Không đưa vào commit: các file kết quả cũ đang bị xoá trong thư mục làm việc (smoke_*, runs/run, demo_spatial_*), `DAY1.md`,
`train_more.sh`, `readme.md` đã sửa, `docs/DAY1.md`, `docs/study.md`, vì đó là thay đổi không do việc này tạo ra.

### Việc 13: chọn NAS=10 làm cấu hình chính cho Analyse/Improve

Theo yêu cầu "consider the best successful nas case for the analyse/improve work". Ghi quyết định vào
`docs/failure_overview.md` mục 1, ý 4 (**SỬA REPO**). Không chạy gì. Từ nay Claude không commit; bạn tự kiểm tra và commit.
Lưu ý: commit `5839ccf` đã được tạo trước khi có yêu cầu này.

### Việc 14: chạy NAS=10 thêm seed 1001, 1002 (libero_10)

Lệnh (bạn đã đồng ý): `TAG=D_nas10 NAS=10 CM=relative N_EP=50 SEEDS="1001 1002" bash run_queue.sh`, log
`ffi26/ffi26/logs/nas10_seeds_20261031_012247.log`, kết quả trong `ffi26/ffi26/results/runs/D_nas10/` (20 job mới,
`timing.csv` thêm 20 dòng). Thời gian: 2366 s (seed 1001) + 2392 s (seed 1002).

Kết quả (nguồn `per_task[0].n_success` trong `eval_info.json`):

| Seed | Task 0–9 | Tổng | Wilson 95% |
|---|---|---|---|
| 1000 | 16, 44, 48, 49, 36, 45, 27, 36, 26, 41 | 368/500 = 73,6% | 69,6–77,3 |
| 1001 | 23, 46, 45, 48, 33, 46, 27, 32, 24, 36 | 360/500 = 72,0% | 67,9–75,8 |
| 1002 | 21, 46, 45, 49, 33, 44, 27, 36, 21, 42 | 364/500 = 72,8% | 68,7–76,5 |
| Gộp | 60, 136, 138, 146, 102, 135, 81, 104, 71, 119 | 1092/1500 = 72,8% | 70,5–75,0 |

Theo cảnh (500 cảnh x 3 seed): 289 thành công 3/3, 83 là 2/3, 59 là 1/3, 69 thất bại 3/3.

Sửa repo (**SỬA REPO**): `docs/failure_overview.md` (cột 3 seed, ý 5–6 mục 1, mục 5 Giới hạn),
`docs/failures/notes/D_nas10.md`, sinh lại `docs/failures/D_nas10.md`. Chưa trích khung / gán nhãn cho seed 1001, 1002.
Không commit.

---

### Việc 15: viết `docs/study.md` (RL cho VLA, diffusion vs flow matching)

Đọc (chỉ đọc): bản HTML arXiv của 2509.09674, 2505.22094, 2510.25889, 2511.14759v2, 2505.17016, 2505.18719, 2506.01844,
2410.24164; trang abs 2210.02747; PDF 2303.04137 (chuyển sang text bằng `pdftotext` trong scratch). Đọc thêm
`ffi26/ffi26/results/probe_report.json`.

Sửa repo (**SỬA REPO**): viết lại toàn bộ `docs/study.md` từ template thành bản hoàn chỉnh, gọn: 6 nguồn RL, bảng so sánh,
câu hỏi mở, diffusion vs flow. Bỏ mục "Nối với kết quả của mình" và checklist. Không đụng `literature_matrix.xlsx`. Không commit.

Sửa tiếp (**SỬA REPO**): bỏ ReinFlow (không phải VLA) khỏi danh sách và mọi trích dẫn; còn 5 bài VLA + RL, đánh số lại 1–5.

---

## Quy tắc Claude tuân theo từ nay trong repo này

1. Trước khi chạy bất cứ thứ gì: nêu mục tiêu và kế hoạch (các bước, file sẽ đọc, file sẽ ghi). Nếu không chỉ là đọc nhanh thì chờ bạn duyệt.
2. Sau khi chạy: ghi thêm một mục vào file này, gồm các lệnh, lý do, tác động (chỉ đọc / scratch / SỬA REPO) và kết quả.
3. Không sửa kết quả trong `~/ffi26` và không chạy các script của repo (`run_queue.sh`, `train_more.sh`, `collect_results.py`, ...) khi chưa được bạn cho phép rõ ràng.
4. Chat với bạn bằng tiếng Anh; mọi file báo cáo (md trong repo) viết bằng tiếng Việt.


## Phiên 31/10/2026 tối: docs/reproduce.md, docs/analyse.md, dọn repo (theo yêu cầu)

| # | Thao tác | Tác động |
|---|---|---|
| 1 | Viết `docs/reproduce.md`, `docs/analyse.md` (số lấy từ `probe_report.json`, `differences.md`, `docs/failure_overview.md`, `results/round2/episodes.csv`) | **SỬA REPO** (file mới) |
| 2 | Viết lại `readme.md` | **SỬA REPO** |
| 3 | Xoá (bạn duyệt): `failure_analysis.md`, `docs/DAY1.md`, `ffi26/lerobot/` (clone thứ hai, gitlink), `ffi26/logs/`, `ffi26/results/`, `ffi26/ffi26/results/{smoke_20261030_110519,smoke_20261030_110908,demo_spatial,runs/run}`, `ffi26/ffi26/ckpt_files/`, `runs/{B_main,F_spatial}/summary_per_*.csv`. Giữ `demo_tasks`. Bản cũ (trừ `runs/run`) còn ở commit `e852bb7` | **SỬA REPO** |
| 4 | Sửa tham chiếu tới file đã xoá → `e852bb7:failure_analysis.md` / `e852bb7:docs/DAY1.md` trong `docs/failure_overview.md`, `docs/analyse.md`, `docs/study.md`, `docs/failures/{B_main,D_nas50,spatial_results}.md`, `docs/failures/notes/{B_main,D_nas50}.md`; thêm ghi chú file đã xoá vào `differences.md` | **SỬA REPO** |


## Phiên 31/10/2026 tối: kiểm tra nhiễu của tỉ lệ FPO r (theo yêu cầu)

Câu hỏi: clip_frac của FPO tăng nhanh là do policy đổi thật hay do r bị nhiễu?

| # | Thao tác | Tác động |
|---|---|---|
| 1 | Đọc `improve/scripts/fpo_train.py`, `improve/outputs/fpo/iter{0..3}/train/{train_log.csv,train_info.json}`. Dòng step 1 của log được tính trước `opt.step()` (θ_new = θ_old): clip_frac 0, \|log r\| ~1e-8 ở cả 4 iter | chỉ đọc |
| 2 | Viết script chẩn đoán `ratio_diag.py` trong scratch (`/tmp/claude-1003/-home-student-Lan-K68AI-temp/89ed5cfe-2f87-46c1-91f0-493cd4bd0a7c/scratchpad/ratio_diag.py`). Import `fpo_train.py` (không sửa) | scratch |
| 3 | Chạy (bạn đã duyệt): `conda activate lerobot; cd ~/ffi26/improve; MUJOCO_GL=egl python <scratch>/ratio_diag.py --out <scratch>/ratio_diag` (181 s, 1 GPU). θ_old = `outputs/fpo/iter2/train/pretrained_model` (input của iter3), dữ liệu `outputs/fpo/iter3`, c = 20.178 (`ratio_scale_used` của iter3). Lặp lại 5 batch train đầu của iter3 (cùng chuỗi rng; loss step 1 = 0.379590, khớp log). Mỗi lr ∈ {5e-9, 5e-8, 5e-7} (cố định, không warmup, AdamW mới): đo r trên 512 decision giữ lại (256/task, task 0 và 8, 1 cặp (τ, ε)/decision) sau 1 và 5 bước, l_new tính theo (a) bf16 autocast như lúc train, (b) bản copy fp32 hoàn toàn, không autocast. Không lưu checkpoint, không ghi vào `~/ffi26` | scratch (`ratio_diag.csv`, `ratio_diag.json`) |

Kết quả (`ratio_diag.csv`; std log r với log r = clamp(c·Δl, −5, 5), Δl = l_old − l_new, l_old cùng chế độ tính):

| lr | bước | std log r bf16 | std log r fp32 | clip_frac bf16 / fp32 | tỉ lệ phần tử bf16 đổi giá trị |
|---|---|---|---|---|---|
| 0 (θ_old, tính lại với batch xáo) | 0 | 7.4e-8 | 7.2e-8 | 0 / 0 | 0 |
| 5e-9 | 1 | 0.0189 | 0.00028 | 0 / 0 | 5.9e-5 |
| 5e-8 | 1 | 0.0276 | 0.0028 | 0 / 0 | 4.7e-4 |
| 5e-7 | 1 | 0.0289 | 0.0279 | 0.006 / 0.004 | 6.7e-3 |
| 5e-9 | 5 | 0.0178 | 0.00092 | 0 / 0 | 1.4e-4 |
| 5e-8 | 5 | 0.0264 | 0.0096 | 0 / 0 | 1.2e-3 |
| 5e-7 | 5 | 0.0324 | 0.0901 | 0.002 / 0.041 | 1.2e-2 |

Ở θ_old, l_bf16 so với l_fp32: std(c·Δl) = 0.054, \|Δl\| trung bình 1.2e-3 (mean l_old 0.0479 trên tập eval).

Kết luận:
- fp32: std log r sau bước 1 tỉ lệ với lr (×10 lr → ×10 std): đây là thay đổi thật của policy.
- bf16: std log r sau bước 1 gần như không đổi (0.019 → 0.029) khi lr tăng 100 lần, và lớn hơn fp32 67 lần ở lr 5e-9: **độ dao động của r sau bước 1 trong log (ví dụ [0.89, 1.08] ở iter0, lr 5e-8) là do lượng tử hoá bf16**. Sàn nhiễu ≈ 0.02–0.03 trên log r, tức σ(Δl) = std log r / c ≈ 0.0189/20.18 – 0.0289/20.18 = 9.4e-4 – 1.4e-3 (2.0–3.0% của mean l_old 0.0479).
- Sàn nhiễu này còn cách ngưỡng clip xa (ln 1.2 = 0.18 ≈ 6–9σ), nên riêng nó gần như không gây clip (clip_frac 0 ở lr ≤ 5e-8). Ở lr 5e-7 sau 5 bước, tín hiệu fp32 (0.090) đã vượt sàn và bf16 lại *thấp hơn* (0.032): phần lớn bước cập nhật nhỏ hơn độ phân giải bf16 nên không đi vào forward bf16. Việc clip_frac tăng tới 0.2 ở step 43–100 (lr ~1e-6) có vẻ do policy đổi thật, nhưng điều này **chưa đo trực tiếp** (chỉ chạy tới 5 bước).

c và ngưỡng clip theo Δl:
- c = 1 / mean(l_old), lấy trung bình trên mọi decision × cặp (τ, ε) của iteration, tính ở bf16 trước khi update (`fpo_train.py`, `ratio_scale: auto`). Lý do (docstring, kiểm chứng V4): với c = 1, r của action tốt không thể vượt exp(l_old) ≈ 1.04 nên không bao giờ bị clip.
- c ≈ 20–25 vì CFM loss trung bình (MSE vận tốc trên các bước được thực thi) của SmolVLA trên chunk do chính nó lấy mẫu là ~0.04–0.05: iter0 1/0.03914 = 25.55, iter3 1/0.04956 = 20.18 (`train_info.json`). l_old tăng dần qua các iter nên c giảm dần.
- Clip 0.2 với c = 20.18 (iter3): r > 1.2 ⇔ Δl > ln(1.2)/c = 0.0090; r < 0.8 ⇔ Δl < ln(0.8)/c = −0.0111. Tức loss CFM giảm khoảng 18% hoặc tăng khoảng 22% so với mean l_old.


## Phiên 31/10/2026 tối: trạng thái đầu dùng để train có trùng tập eval không (theo yêu cầu)

| # | Thao tác | Tác động |
|---|---|---|
| 1 | Đọc `results/round2/configs/{fpo,round1,eval_fpo,eval_seed2000,report_seed2000}.yaml`, `improve/scripts/{phase3_eval,r2_report}.py`, `improve/scripts/fpo_confirm.sh` | chỉ đọc |
| 2 | Python (env lerobot, chỉ đọc) trên `improve/outputs/fpo/iter{0..3}/{rollouts.csv,collect_info.json}`, `results/round2/episodes.csv` (seed 1000), `results/round2/seed2000/episodes.csv`: tách thành công theo init state dùng để thu rollout và init state giữ lại | chỉ đọc |

Kết quả:
- **Có trùng.** Eval 50 ep/task chạy episode k với init state k (`phase3_eval.py`: `init_ids=ks`), tức đủ 50 init state, gồm cả 16 init state/task dùng để thu rollout (`fpo.yaml` `collect.tasks.init_ids`). Vậy 32/100 ep eval trên task 0, 8 nằm trên init state đã train. Seed env không trùng: thu rollout dùng 200000–531007, eval dùng 1000+k và 2000+k.
- Báo cáo đã tách phần này: `r2_report.py` lấy held-out = 50 − init_ids của `round1.yaml`. Danh sách đó giống hệt `fpo.yaml`, và cờ `heldout` trong cả hai `episodes.csv` khớp với phép tách tính lại.
- Mapping: `fpo_itK` = θ_K = `iter{K-1}/train`; lần thu iter k dùng θ_k (`collect_info.json`). Vậy lần thu iter3 (133/256) và eval `fpo_it3` cùng một checkpoint.

| | init đã train: thu rollout (16 init × 8, seed mới) | init đã train: eval s1000 / s2000 / gộp | held-out: eval s1000 / s2000 / gộp |
|---|---|---|---|
| official (θ0) | 96/256 = 37.5% | 14/32 / 16/32 / 30/64 = 46.9% | 30/68 / 29/68 / 59/136 = 43.4% |
| fpo_it3 (θ3) | 133/256 = 52.0% | 18/32 / 15/32 / 33/64 = 51.6% | 39/68 / 31/68 / 70/136 = 51.5% |

Nhận xét:
- Trên init state đã train, eval của θ3 (51.6%) khớp với tỉ lệ khi thu (52.0%). Mức tăng 96 → 133 khi thu không lặp lại ở eval chủ yếu vì official đạt 46.9% trên chính các init state này khi eval, so với 37.5% khi thu. Nguyên nhân của chênh lệch 9.4 điểm này **chưa kiểm tra** (có thể do nhiễu: n = 64, SE ≈ 6 điểm; hoặc do khác biệt giữa pipeline thu và pipeline eval).
- Vì vậy phần trùng không làm phồng kết quả eval. Kết quả eval trên init state đã train cũng không tăng rõ (+3/64 khi gộp; n quá nhỏ để kết luận).


## Phiên 31/10/2026 tối: so sánh pipeline thu rollout (fpo_collect.py) và eval (phase3_eval.py), checkpoint chính thức (theo yêu cầu, chỉ đọc)

| # | Thao tác | Tác động |
|---|---|---|
| 1 | Đọc `improve/scripts/{fpo_collect,phase3_eval,r2_collect}.py`, `improve/src/{rollout,fpo_capture,logged_env,reward}.py`, `lerobot/src/lerobot/envs/{libero,configs,utils}.py`, `lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py`, gymnasium 1.4.0 `vector/async_vector_env.py`, `core.py`; `git log`/`git show` cho các file trên | chỉ đọc |
| 2 | Python (chỉ đọc) kiểm tra `init_state_id`/`seed` ghi trong `*.meta.json`: 1024 rollout FPO, `outputs/eval_baseline` (100), `r2/eval/fpo_it3/task{0,8}` (100), `r2/eval_s2000` (200) → 0 sai khác so với giá trị mong đợi | chỉ đọc |
| 3 | So `r2/round1/rollouts.csv` (official, `r2_collect.py`, seed 100000) với `fpo/iter0/rollouts.csv`; kiểm định Fisher (scipy); thời gian từ `collect_info.json`, `fpo/confirm.log` | chỉ đọc |

Giống nhau (cả hai gọi chung `load_policy`, `make_vec_env`, `run_batch` trong `src/rollout.py`):
- Sinh action: ODE Euler tất định khi đã có nhiễu, `num_steps` = 10 (config checkpoint), không có temperature, `noise_scale` = 1.0 (`modeling_smolvla.py:777-830`). Chunk 50, thực thi 10 (`n_action_steps` = 10 ở `fpo.yaml` và các config eval), lấy chunk mới khi hàng đợi rỗng, không RTC (`modeling_smolvla.py:256-285`).
- `max_steps` = 520 (`rollout.py:38,144`), kết thúc khi `check_success()` (`libero.py:376-377`), success = cờ của probe hoặc của env (`rollout.py:185`), `num_steps_wait` = 10 (`libero.py:127,351`).
- Ảnh 256×256 (mặc định của class `LiberoEnv`, `libero.py:119-120`; `make_vec_env` không ghi đè, `rollout.py:73-88`), lật `[::-1, ::-1]` (`libero.py:284`). Pre/post-processor và dtype cùng `load_policy` (`rollout.py:41-71`), không autocast trong `run_batch`.
- Init state đặt qua `next_init_state_id` (`logged_env.py:49-52`; gymnasium `set_wrapper_attr` truyền xuống env bên trong, đã kiểm bằng meta ở bước 2). `hard_reset=True` (`libero.py:131,274`). `log_sim_state=False` ở cả hai (`fpo_collect.py:77`, `phase3_eval.py:57`). `LiberoProbe.read()` chỉ đọc, không gọi step/render (`reward.py:69-108`, tìm bằng grep).

Khác nhau:

| # | Mục | fpo_collect.py | phase3_eval.py | Hướng tác động dự kiến |
|---|---|---|---|---|
| D1 | Nguồn nhiễu sampler | Generator CPU riêng mỗi lần gọi chunk, seed = `seed0*100 + 1000g + k` (`fpo_capture.py:39-54`, `fpo_collect.py:80`) | RNG torch toàn cục, `set_seed(seed)` mỗi (model, task) (`phase3_eval.py:52`; `modeling_smolvla.py:791-793`) | Không có tác động hệ thống (cùng N(0,1), float32). Bằng chứng: `r2_collect.py` dùng RNG toàn cục như eval (`r2_collect.py:69`) được 93/256, `fpo_collect` được 96/256, Fisher p = 0.85 |
| D2 | Thành phần batch | 8 sub-env cùng 1 init state (`fpo_collect.py:82`) | 8 init state liên tiếp, batch cuối 2 (`phase3_eval.py:55-60`) | Không dự kiến (các mẫu không tương tác, cùng câu lệnh nên không có padding khác nhau). Chưa đo |
| D3 | Vòng đời vec env | 1 vec env/task dùng lại cho 16 nhóm (`fpo_collect.py:74-77`) | vec env mới mỗi batch (`phase3_eval.py:57,61`) | Không dự kiến vì `hard_reset=True` dựng lại sim mỗi lần reset |
| D4 | Seed env | 200000 + 100000k + 1000g + j (`fpo_collect.py:64,79`) | 1000 + k / 2000 + k (`phase3_eval.py:59`) | Không dự kiến; chưa kiểm seed ảnh hưởng gì sau khi đã đặt init state |
| D5 | Nguồn số official s1000 | — | `outputs/eval_baseline` (`report.yaml:11`), tạo 04:20 31/10 bằng `phase3_eval.py` + `configs/phase3_r1.yaml` (đã xoá, chưa từng commit), trước commit đầu (15:10) → không xác minh được phiên bản code. Nhật ký `improve_2026-10-31.md:162,290`: seed 1000+k, NAS = 10 | Không xác minh được |
| D6 | Chọn init state (không phải pipeline) | — | Task 0: init state "lẫn lộn" qua lerobot-eval D_nas10 seed 1000–1002; task 8: 16 trong 23 init state lẫn lộn gần 50% nhất khi gộp D_nas10 + `outputs/eval_baseline` (seed 1000) + Phase 1 (`round1.yaml` dòng 15-22, `improve_2026-10-31.md:233-234`) | Kết quả official s1000 trên init state đã train không độc lập với tiêu chí chọn (bị kéo về gần 50%) → có thể ước tính cao hơn tỉ lệ thật. Eval s2000 và mọi lần thu thì độc lập |

Số liệu, official trên 32 init state đã train:
- Thu rollout: `r2_collect` 93/256 = 36.3%, `fpo_collect` 96/256 = 37.5%, gộp 189/512 = 36.9%.
- Eval: s1000 14/32 = 43.8% (Fisher so với 189/512: p = 0.455), s2000 16/32 = 50.0% (p = 0.187), gộp 30/64 = 46.9% (p = 0.134).

Kết luận: không tìm thấy khác biệt pipeline đáng kể. Chênh lệch giữa thu rollout (36.9%) và eval (46.9%) không có ý nghĩa thống kê (p = 0.13), và tương thích với nhiễu do n = 32/seed, cộng với sai lệch do chọn init state (D6) ở s1000.

Đề xuất (CHƯA CHẠY, chờ duyệt):
- (a) Theo yêu cầu: chạy lại official qua `fpo_collect.py` trên 16 init state × 8 seed mới × task 0, 8 (256 rollout). Ước tính khoảng 16 phút (4 lần thu FPO: 949–967 s, `fpo/iter*/collect_info.json`). Lưu ý: việc này đã có hai lần độc lập (93/256, 96/256), nên chạy lần ba khó cho thêm thông tin.
- (b) Phép thử phân biệt được: eval official và fpo_it3 bằng `phase3_eval.py` trên đúng 32 init state đã train × 8 seed mới (256 ep/model, 512 ep tổng). Cần sửa nhỏ để `phase3_eval.py` nhận danh sách init state/seed (hiện chỉ chạy init 0..n−1). Ước tính khoảng 4.7–5.5 s/ep (`fpo/confirm.log`) → khoảng 40–47 phút.


## Phiên 31/10/2026 tối: kiểm tra n_action_steps và nguồn gốc checkpoint (theo yêu cầu, chỉ đọc)

| # | Thao tác | Tác động |
|---|---|---|
| 1 | Đọc `results/round2/configs/*.yaml`, `improve/src/rollout.py`, `run_queue.sh`, `results/runs/D_nas10/lerobot_smolvla_libero/*/{cmd.txt,eval.log,eval_info.json,checkpoint_revision.txt}`, `lerobot/src/lerobot/envs/{libero,configs}.py`, `lerobot/src/lerobot/configs/policies.py`, `smolvlm_with_expert.py`, `differences.md` | chỉ đọc |
| 2 | Đọc `config.json` và header safetensors (đếm dtype) của `~/.cache/huggingface/hub/models--{lerobot,HuggingFaceVLA}--smolvla_libero`; `sha256sum` hai file `model.safetensors` | chỉ đọc |

Bảng cấu hình hiệu lực:

| Run | n_action_steps | chunk / bước flow | max_steps | Ảnh env | dtype inference | Nguồn |
|---|---|---|---|---|---|---|
| phase3_eval: round 1 (`eval_round1*.yaml`), round 2 s1000 (`eval_fpo.yaml`, `eval_official.yaml`, `eval_fpo_it3_other.yaml`), s2000 (`eval_seed2000.yaml`) | 10 | 50 / 10 | 520 | 256×256 | trọng số bf16 (474/500 tensor) + fp32 (26), không autocast | key `n_action_steps: 10` dòng 3–5 của mỗi config → `rollout.py:50` ghi đè `config.json` (50); chunk_size/num_steps từ `config.json`; `rollout.py:38`; `libero.py:119-120` (mặc định của class, `make_vec_env` `rollout.py:73-88` không truyền) |
| phase3_eval: `outputs/eval_baseline` (official s1000 task 0, 8) | 10 [CHƯA KIỂM CHỨNG: `configs/phase3_r1.yaml` đã xoá, chưa commit; chỉ có nhật ký `improve_2026-10-31.md:162,290`] | 50 / 10 | 520 | 256×256 | như trên | như trên |
| fpo_collect (iter 0–3), r2_collect (round 1) | 10 | 50 / 10 | 520 | 256×256 | như trên | `fpo.yaml:4`, `round1.yaml:5` → `fpo_collect.py:66` / `load_policy` |
| D_nas10 (~72%): `lerobot-eval` qua `run_queue.sh` | 10 | 50 / 10 | 520 (`episode_length` None → `TASK_SUITE_MAX_STEPS`) | **360×360** | như trên, `use_amp: False` | `cmd.txt` `--policy.n_action_steps=10`; `eval.log` dòng 45, 61, 85, 91, 118; `configs.py:357-358,443-444` |

D_nas10: 10 task libero_10 × 50 init state × 3 seed (1000, 1001, 1002), batch 5, relative. 368/500 (73.6%), 360/500 (72.0%), 364/500 (72.8%), gộp 1092/1500 = 72.8%.

Kết luận bước 1:
- Thu rollout và eval dùng **cùng** n_action_steps = 10 (và cùng chunk 50, 10 bước flow, 520 bước, 256×256), nên NAS không giải thích được 37.5% so với 46.9%.
- Khác biệt mới tìm thấy: pipeline của mình (`rollout.py`) render ảnh 256×256, còn `lerobot-eval` (run 72%) render 360×360. Các tham số env khác giống nhau (control_freq 20, num_steps_wait 10, hard_reset True). Trên task 0, 8 chưa thấy ảnh hưởng: D_nas10 (360) task 0 60/150 = 40.0%, task 8 71/150 = 47.3%; pipeline của mình (256), official s1000 + s2000: task 0 42/100 = 42.0%, task 8 47/100 = 47.0%. Chưa so trên 8 task còn lại.

Bước 2 (nguồn gốc checkpoint):
- "Official" = `lerobot/smolvla_libero` @ `a24f1f380461b665158d52acd140152c79e8b36d` (`D_nas10/.../checkpoint_revision.txt`; `fpo.yaml` `init_policy`). `config.json`: n_action_steps 50, chunk_size 50, num_steps 10, VLM `SmolVLM2-500M-Video-Instruct`, 16 lớp VLM, `expert_width_multiplier` 0.75.
- `HuggingFaceVLA/smolvla_libero` @ `6721902b…` là checkpoint khác (sha256 file trọng số khác; `config.json`: n_action_steps 1, `SmolVLM2-500M-Instruct`, `expert_width_multiplier` 0.5). Chỉ dùng ở các smoke/demo cũ (`differences.md:27`).
- Khác giao thức sim của paper: NAS 10 (paper 1); 50 init state × 3 seed/task (paper 10 lần thử/task); ảnh 360 (lerobot-eval) hoặc 256 (pipeline mình), độ phân giải trong paper [CHƯA KIỂM CHỨNG]. Bước flow giống (10). Với checkpoint này, NAS 1 qua lerobot-eval cho 54.6% (D_nas1) và 52.7% (B_main), thấp hơn NAS 10 (`differences.md:78`, `docs/failure_overview.md`).


## Phiên 31/10/2026 tối: FPO cải thiện kiểu thất bại nào (theo yêu cầu, chỉ đọc)

| # | Thao tác | Tác động |
|---|---|---|
| 1 | Đọc `results/round2/{summary.md,failure_types.csv}`, `results/round2/seed2000/summary.md`, `improve/improve_2026-10-31.md` (mục FPO). Không có chỗ nào diễn giải FPO sửa được kiểu thất bại nào; chỉ có bảng đếm thô | chỉ đọc |
| 2 | Python (chỉ đọc): đếm nhãn chính (`primary`, từ `src/failure_labels.py`) và sự kiện theo vòng thu (`improve/outputs/fpo/iter{0..3}/rollouts.csv`); ghép cặp official và fpo_it3 trên cùng (task, init, seed) trong `results/round2/episodes.csv` (s1000) và `results/round2/seed2000/episodes.csv`, đếm nhãn của các episode đổi kết quả | chỉ đọc |

Thu rollout (init state đã train, seed mới, 128 rollout/task; iter0 = official, iter3 = θ3):

| Task | Vòng | S | A gắp hỏng | B kẹt chuyển subtask | C nhầm vật | D làm hỏng tiến độ | E đứng yên | drop / slip / collision (sự kiện) |
|---|---|---|---|---|---|---|---|---|
| 0 | iter0 | 55 | 53 | 5 | 2 | 0 | 12 | 7 / 92 / 7 |
| 0 | iter3 | 75 | 41 | 3 | 2 | 1 | 5 | 12 / 101 / 3 |
| 8 | iter0 | 41 | 27 | 11 | 0 | 4 | 45 | 11 / 34 / 0 |
| 8 | iter3 | 58 | 17 | 2 | 0 | 8 | 42 | 6 / 46 / 0 |

(O: iter0 task 0 = 1, iter3 task 0 = 1, task 8 = 0 / 1.)

Eval ghép cặp official → fpo_it3 (task 0, 8; nhãn của official ở ca FPO sửa được, nhãn của FPO ở ca FPO làm hỏng):
- s1000: sửa 19 (task 0: A 10; task 8: E 4, B 3, A 2), hỏng 6 (task 0: A 1, C 1; task 8: E 3, A 1) → +13.
- s2000: sửa 18 (task 0: A 4, C 1, E 1; task 8: E 8, A 3, D 1), hỏng 17 (task 0: A 6, C 1, E 1; task 8: E 5, A 3, D 1) → +1.

Nhận xét:
- Khi thu rollout: số thất bại A (gắp hỏng) giảm ở cả hai task (task 0: −12, task 8: −10), B giảm ở task 8 (−9), E giảm ở task 0 (−7). E ở task 8 (kiểu thất bại lớn nhất, 45/87) gần như không đổi (42); D task 8 tăng 4 → 8; sự kiện slip tăng nhẹ.
- Ở eval: mức giảm A trên task 0 thấy ở s1000 (net +9) nhưng đảo chiều ở s2000 (net −2). Như vậy chỉ thấy cải thiện ổn định ở phân phối thu rollout, chưa thấy ổn định ở eval.
- Lưu ý: nhãn là nhãn tự động (`failure_labels.py`), chưa kiểm tra lại bằng mắt cho các episode này. Official s1000 có sai lệch chọn init state (xem mục so sánh pipeline ở trên).


## Phiên 31/10/2026 tối: đề xuất cải thiện reward từ phân tích thất bại (theo yêu cầu, chỉ đọc)

| # | Thao tác | Tác động |
|---|---|---|
| 1 | Đọc `docs/failure_overview.md` §1–5, `docs/failures/D_nas10.md`, `improve/src/{reward,reward_r2,failure_labels}.py` | chỉ đọc |
| 2 | Python (chỉ đọc) trên `improve/outputs/fpo/iter{0,3}/logs/*.npz`, `rollouts.csv`, `groups.csv`: trạng thái predicate cuối của các thất bại task 8, predicate đúng ở t = 0, vật đã từng được gắp, tương quan độ lệch return trong nhóm với độ lệch success | chỉ đọc |

Số liệu mới:
- Task 8: predicate `turnon flat_stove_1` đúng ngay từ t = 0 ở 128/128 rollout iter0 (Φ0 = 1/3); task 0: Φ0 = 0.
- Thất bại task 8: `moka_pot_1` gần như không bao giờ lên bếp (chỉ 2/70 ở iter3, 0/87 ở iter0). Ở iter3, trong 42 thất bại nhãn E, 21 ca chưa từng gắp được ấm nào, 17 ca đã đặt `moka_pot_2` lên bếp nhưng chưa từng gắp `moka_pot_1`.
- Trong nhóm, độ lệch return so với trung bình nhóm tương quan 0.956 (task 0) và 0.965 (task 8) với độ lệch success (iter3): advantage gần như chỉ phản ánh thành công/thất bại.
- Mỗi task có 2–3 nhóm/vòng mà cả 8 rollout đều thất bại; ở các nhóm này chỉ Φ và chi phí phân biệt được các rollout.


## Phiên 31/10/2026 tối: số liệu cho fpo_v2.yaml (dữ liệu dùng, lọc nhóm, nhiễu gradient) (theo yêu cầu)

| # | Thao tác | Tác động |
|---|---|---|
| 1 | Python (chỉ đọc) trên `improve/outputs/fpo/iter{0..3}/{collect_info.json,groups.csv,decisions/g*.npz,train/train_info.json,train/train_log.csv}` | chỉ đọc |
| 2 | Viết `grad_noise.py` trong scratch (`/tmp/claude-1003/-home-student-Lan-K68AI-temp/89ed5cfe-2f87-46c1-91f0-493cd4bd0a7c/scratchpad/grad_noise.py`, import `fpo_train.py`, không sửa repo) | scratch |
| 3 | Chạy (theo yêu cầu (c)): `conda activate lerobot; cd ~/ffi26/improve; MUJOCO_GL=egl python <scratch>/grad_noise.py --out <scratch>/grad_noise` (117 s, đỉnh 8.5 GB). Gradient của loss FPO tại θ_old (r = 1) cho θ0 (input iter0) và θ3 (input iter3), task 0 và 8, các batch rời nhau: 6 cặp batch 32 và 4 cặp batch 128 (128 = cộng dồn 4 micro-batch 32) mỗi (iter, task). Không cập nhật trọng số, không lưu checkpoint | scratch (`grad_noise.csv`, `grad_noise.json`) |

(a) Quyết định thu được và quyết định thực sự dùng để cập nhật:

| Iter | Thu được | Dùng được (không bị loại, adv ≠ 0) | Bước × 32 | % dùng được / % thu được | Phần epoch đã đi trước khi dừng sớm |
|---|---|---|---|---|---|
| 0 | 11746 | 10914 | 100 × 32 = 3200 | 29.3% / 27.2% | 0.29 |
| 1 | 11259 | 10414 | 69 × 32 = 2208 | 21.2% / 19.6% | 0.21 |
| 2 | 11163 | 10747 | 50 × 32 = 1600 | 14.9% / 14.3% | 0.15 |
| 3 | 11130 | 10261 | 43 × 32 = 1376 | 13.4% / 12.4% | 0.13 |

(b) Lọc nhóm theo độ chênh return 0.1: bị loại 2, 2, 1, 2 trên 32 nhóm (6.2%, 6.2%, 3.1%, 6.2%); nhóm dùng được 30, 30, 31, 30. Mọi nhóm bị loại đều có 0/8 thành công. Nhóm toàn thành công không bị loại (2, 4, 5, 4 nhóm).

(c) Cosine giữa gradient của hai batch rời nhau (trung bình ± SE):

| θ | B = 32 (12 cặp) | B = 128 (8 cặp) | Cosine giữa task 0 và task 8 (trung bình 8 × 128 mỗi task) |
|---|---|---|---|
| θ0 | −0.031 ± 0.052 | 0.082 ± 0.036 | 0.070 |
| θ3 | 0.121 ± 0.065 | 0.260 ± 0.055 | 0.522 |

Theo task: θ0 task 0: −0.022 (32) / 0.070 (128); θ0 task 8: −0.039 / 0.093; θ3 task 0: 0.009 / 0.242; θ3 task 8: 0.232 / 0.277.
Chuẩn gradient trước khi clip trong train v1: trung vị 19.6–23.5; số bước bị clip (> 10): 83/100, 61/69, 47/50, 39/43 (`train_log.csv`).
Thời gian một bước B = 32 trong v1: khoảng 0.45 s (iter3, `time_s` từ bước 1 đến bước 19: 231.7 → 239.8 s).

Nhận xét:
- Ở B = 32, hai batch rời nhau cho gradient gần như không cùng hướng (cosine ≈ 0 ở θ0): bước cập nhật v1 chủ yếu là nhiễu. B = 128 cho cosine dương rõ hơn ở cả hai θ.
- v1 chỉ dùng 12–27% số quyết định đã thu trước khi dừng sớm.
- Grad clip 10 gần như luôn bật. Với AdamW, clip chủ yếu đổi tỉ lệ giữa các bước, không phải giới hạn an toàn thật sự [suy luận].


## Phiên 31/10–1/11/2026: FPO v2 — chọn batch, code, dry-run, chạy đầy đủ (bạn đã duyệt)

| # | Thao tác | Tác động |
|---|---|---|
| 1 | `grad_noise.py` (scratch) thêm tham số `--sizes`; chạy `--iters 0 --sizes 256:4,512:4` (184 s, đỉnh 8.5 GB) | scratch (`grad_noise_big/`) |
| 2 | Kiểm tra (chỉ đọc + nạp model trên GPU): `load_policy` nạp checkpoint v1 θ3 → 129/145 tensor expert fp32 trong file bị đưa vào module bf16 (98.3% phần tử khác giá trị fp32). So file: phần tử expert bf16 khác θ0: θ1 8.47%, θ2 10.12%, θ3 10.97%, θ4 11.37%; trung bình \|fp32 − θ0\| ≈ 8e-6 ở cả 4 checkpoint (không tích luỹ qua các vòng) | chỉ đọc |
| 3 | Viết `improve/scripts/fpo_train_v2.py`, `improve/scripts/fpo_v2_loop.sh`, `results/round2/configs/{fpo_v2,eval_fpo_v2_s3000,eval_fpo_v2_s4000,eval_fpo_v2_s5000,eval_fpo_v2_other,report_fpo_v2_s3000,report_fpo_v2_s4000,report_fpo_v2_s5000,report_fpo_v2_other}.yaml`. Không sửa script/config/kết quả v1 | **SỬA REPO** (file mới) |
| 4 | Dry-run: `fpo_collect.py` với config v2 (seed 1200000), 1 nhóm → 8/8, 26 s; `fpo_train_v2.py` trên dữ liệu v1 iter0 (512 decision/task), 6 bước, `lambda_bc=1`, `--grad_parts`; rồi kiểm tra ranh giới vòng: thu 1 nhóm bằng checkpoint dry-run + train 1 bước từ nó | `outputs/fpo_v2/dryrun/` (mới) |
| 5 | Chạy đầy đủ: tmux `fpo_v2` (`bash scripts/fpo_v2_loop.sh`, log `outputs/fpo_v2/loop.log`), tmux `fpo_v2_gpu` (`outputs/fpo_v2/gpu.csv`). Bắt đầu 31/10 23:29 | **GPU**, `outputs/fpo_v2/` (mới) |

Bước 1, chọn batch (cosine gradient hai batch rời nhau tại θ0, 4 cặp/task): B 256: task 0 0.063, task 8 0.071, gộp 0.067; B 512: 0.127, 0.089, gộp 0.108 (B 128 trước đó: 0.082 ± 0.036). Theo quy tắc của bạn (512 nếu ≥ 0.2; 256 nếu rõ ràng hơn 128; nếu không thì 128) → **batch 128**.

Dry-run (`outputs/fpo_v2/dryrun/train/{train_log,kl_log}.csv`, `train_info.json`):
- Bước 1 (fp32, θ_new = θ_old): r ∈ [1.0, 1.0000017], std log r 1.5e-7, clip_frac 0.
- Bước 2–6 (lr warmup 2e-7 → 1e-6): std log r 0.019, 0.016, 0.029, 0.056, 0.096; clip_frac 0, 0, 0, 0.016, 0.039. KL proxy so với θ0: 4.7e-4 (bước 2), 2.9e-3 (4), 1.1e-2 (6).
- Thời gian: pass loss cũ 56 s cho 1024 decision × 2 cặp; một bước B = 128 khoảng 5.2 s (không có KL), KL probe khoảng 7 s. Đỉnh bộ nhớ torch 11.84 GB.
- λ_bc: chuẩn gradient PPO (trung vị 6 bước) 18.8, BC với λ = 1 là 0.91 → λ = 0.25 · 18.8 / 0.91 = 5.2 → **λ_bc = 5** (quy tắc nêu trước khi đo: gradient BC ≈ 25% gradient PPO).
- Ranh giới vòng: checkpoint v2 lưu expert fp32 145/145, VLM bf16 như v1; thu rollout nạp được (8/8); vòng sau khôi phục 155 tensor fp32 từ file; r ở bước 1 vẫn bằng 1 (std log r 5.7e-8).

Quyết định (ghi rõ để bạn kiểm tra): khi thu rollout và eval vẫn nạp checkpoint như v1 (expert trong module bf16), để so sánh được với mọi eval trước; train v2 khôi phục trọng số fp32 từ file nên master fp32 được giữ qua các vòng. Hệ quả: policy thu rollout là bản làm tròn bf16 của master fp32 (giống RL mixed precision thông thường).

Ước tính thời gian: thu 4 × khoảng 16 phút; train mỗi vòng khoảng 12 phút (nạp model + pass loss cũ fp32 khoảng 10 phút) cộng tối đa 3 epoch × khoảng 82 bước × 5.2 s (khoảng 21 phút, ít hơn nếu dừng sớm); eval 1060 ep × khoảng 5 s ≈ 90 phút. Tổng khoảng 3.3–4.9 h → xong khoảng 02:50–04:25 ngày 1/11.

Còn mở: tmux `fpo_gpu` (monitor của v1) vẫn đang ghi vào `outputs/fpo/gpu.csv` (file kết quả v1); chưa dừng, chờ bạn quyết định. Quy tắc collapse đã đăng ký trước không có trong phiên này; loop chỉ eval θ4 của v2.


## Phiên 31/10/2026 23:33–23:44: FPO v2 chạy lại với expert fp32 khi suy luận (theo yêu cầu)

**Ghi cho báo cáo (v1):** trong v1, trọng số master fp32 bị làm tròn về bf16 ở mỗi ranh giới vòng (`load_policy` đưa 129/145 tensor expert vào module bf16, cả khi thu rollout lẫn khi vòng train sau nạp checkpoint). Vì vậy cập nhật không tích luỹ qua các vòng: trung bình \|fp32 − θ0\| của expert ≈ 8e-6 ở cả θ1–θ4 (8.58e-6, 8.12e-6, 8.28e-6, 8.24e-6). Phần cập nhật còn lại sau làm tròn: 8.5–11.4% phần tử expert bf16 khác θ0. Nguồn: mục "FPO v2 — chọn batch…" ở trên.

**`outputs/fpo/gpu.csv`:** tmux `fpo_gpu` (monitor của v1) đã dừng lúc 23:33:52 (unix 1793464432; dòng cuối cùng của file có thời điểm này). Hoạt động v1 cuối cùng là báo cáo xác nhận, kết thúc 31/10 21:22:11 (unix 1793456531, `outputs/fpo/timing_confirm.txt`). **Các dòng có unix_time > 1793456531 không thuộc v1**; trong đó có cả bộ nhớ GPU của các kiểm tra và dry-run v2.

| # | Thao tác | Tác động |
|---|---|---|
| 1 | Dừng tmux `fpo_v2`, `fpo_v2_gpu`, `fpo_gpu` (23:33:52). Chuyển `outputs/fpo_v2` → `outputs/fpo_v2_aborted` (iter0 dở dang 7 nhóm, dry-run, loop.log, gpu.csv) | **GPU dừng**; đổi tên thư mục |
| 2 | `src/rollout.py`: thêm `checkpoint_file()` và tham số `load_policy(..., expert_fp32=False)`. Khi True: ép `lm_expert` sang fp32 và đọc lại giá trị từ file checkpoint; VLM giữ bf16. `scripts/fpo_collect.py`, `scripts/phase3_eval.py`: đọc key `expert_fp32` từ config (mặc định False → hành vi v1 không đổi). `scripts/fpo_train_v2.py`: dùng `checkpoint_file` chung | **SỬA REPO** (3 file v1, mặc định giữ nguyên hành vi) |
| 3 | Kiểm tra (scratch `verify_fp32_expert.py`, checkpoint `fpo_v2_aborted/dryrun/train/pretrained_model` có expert fp32): policy nạp như khi thu rollout (`expert_fp32=True`) và policy nạp như khi train (`build_policy_fp32`) | scratch |
| 4 | Đo chi phí thu rollout: `fpo_collect.py --max_groups 2` (official, seed 1200000), bf16 và fp32, mỗi loại chạy 2 lần | `outputs/fpo_v2_costtest/` (mới) |
| 5 | `fpo_v2.yaml`: thêm `expert_fp32: true` và mô tả quy tắc collapse. Xoá các config eval/report v2 cũ (chưa từng dùng), viết `eval_fpo_v2_s{1000,3000,4000,5000}.tmpl.yaml`, `report_fpo_v2_{s3000,s4000,s5000,other,sanity}.yaml`. Viết lại `scripts/fpo_v2_loop.sh`: thêm lần thu iteration 4 (θ4, không train) chỉ để áp quy tắc collapse, chọn checkpoint → `outputs/fpo_v2/selection.json`, eval checkpoint được chọn. Đã thử logic chọn trên 3 bộ số giả (không giảm → θ4; giảm ở iter 3 → θ2; giảm ở iter 1 → official) | **SỬA REPO** |
| 6 | Khởi động lại 23:43:29; `tee` không tạo được `loop.log` (thư mục chưa có) → dừng sau khoảng 20 s, xoá `outputs/fpo_v2` của lần chạy hỏng này, tạo thư mục trước rồi chạy lại lúc **23:43:59**: tmux `fpo_v2` (log `outputs/fpo_v2/loop.log`), tmux `fpo_v2_gpu` (`outputs/fpo_v2/gpu.csv`) | **GPU** |

Kết quả kiểm tra (bước 3):
- Trọng số: chênh lệch tối đa giữa policy thu rollout và policy train = 0.0 cho expert (fp32 khi thu), VLM (bf16 khi thu) và các projection (fp32). Nạp kiểu v1 thì expert lệch tối đa 4.0e-6.
- Còn lại khác biệt tính toán: khi thu, VLM (đóng băng) tính ở bf16, còn khi train toàn bộ fp32. Trên cùng batch (256 decision, c = 24.6): c·\|Δl\| trung bình 0.023 (std 0.057, max 0.45) giữa policy thu rollout và policy train; nạp kiểu v1: 0.047 (std 0.099, max 0.82). Ratio ở bước 1 (l_old và l_new cùng tính trong train) phải bằng 1; sẽ kiểm trên log iter0 của lần chạy thật.

Chi phí thu rollout (bước 4, 2 nhóm = 16 rollout): thời gian rollout bf16 61.97 s / 62.77 s, fp32 63.11 s / 62.98 s; bộ nhớ torch bf16 1.16 GB, fp32 1.36 GB. Mỗi chế độ cho kết quả giống hệt nhau giữa 2 lần chạy (bf16: 10/16 thành công, 640 decision; fp32: 11/16, 625 decision).

Quy tắc collapse (đăng ký trước, theo bạn): báo cáo vòng cuối của v2, trừ khi tỉ lệ thành công khi thu của vòng k+1 thấp hơn vòng k quá 15 điểm; khi đó báo cáo checkpoint cuối trước lần giảm. Chỉ dựa vào tỉ lệ thu, không dựa vào eval. Cách hiểu đã cài đặt: θ_k là checkpoint thu vòng k; để áp quy tắc cho θ4 cần thêm một lần thu bằng θ4 (iteration 4, 256 rollout, không train). Chọn lần giảm đầu tiên.

Eval (expert fp32 cho mọi nhánh): official, v1 θ3, v2 đã chọn × task 0, 8 × 50 × seed 3000/4000/5000; seed 1000: 8 task khác × 20 cho cả 3 nhánh, cùng dòng kiểm tra độ chính xác (official fp32, task 0, 8 × 50, so với official bf16 `outputs/eval_baseline`; chỉ báo cáo, không dùng để chọn).

ETA: thu 5 × khoảng 16 phút; train 4 × khoảng 12–37 phút; eval 1480 ep × khoảng 5 s ≈ 2 h → xong khoảng 03:55–05:35 ngày 1/11.

Kiểm tra lần chạy thật, iteration 0 (`outputs/fpo_v2/iter0/collect_info.json`, `iter0/train/train_log.csv`): thu 97/256, 1012 s, bộ nhớ torch 1.36 GB; pass loss cũ fp32 619 s (11245 decision × 2 cặp), c = 24.606. **Bước 1: r ∈ [0.99999976, 1.0000019], std log r 1.8e-7, clip_frac 0** → ratio bằng 1 khi l_old tính từ cùng trọng số fp32 đã dùng để thu. Bước 2–7: std log r 0.009, 0.055, 0.122, 0.061, 0.238, 0.127; clip_frac lên 0.10 ở bước 6. Mỗi bước khoảng 5.3 s.


## Phiên 01/11/2026 00:30–00:45: video cho phần Improve (theo yêu cầu; chưa chạy GPU)

Lúc này loop `fpo_v2` vẫn chạy (đang ở `train1`). Chưa chạy gì trên GPU; chỉ viết code và chọn episode.

| # | Thao tác | Tác động |
|---|---|---|
| 1 | Đọc `phase3_eval.py`, `fpo_v2_loop.sh`, `src/rollout.py`, `src/reward_r2.py`, `results/round2/episodes.csv`, `results/round2/seed2000/episodes.csv` | chỉ đọc |
| 2 | Viết script mới `improve/scripts/phase3_eval_video.py` (phương án (b)). **Không sửa** `phase3_eval.py`, `rollout.py`, `reward_r2.py`, `phase1_logging.py`. Lý do: loop v2 gọi `python scripts/phase3_eval.py` thành một process mới ở bước eval, nên nó sẽ đọc file trên đĩa tại thời điểm đó; mọi sửa đổi (kể cả lỗi cú pháp) sẽ ảnh hưởng đến eval của v2 | **SỬA REPO** (file mới) |
| 3 | Viết `results/videos/configs/videos.yaml` (9 cặp episode, xem bên dưới) | **SỬA REPO** (file mới) |
| 4 | Kiểm tra phần vẽ header trên CPU (`CUDA_VISIBLE_DEVICES=` rỗng), dùng episode Phase 1 `outputs/baseline/logged/task0/ep000` (npz + mp4): 520 frame, header đúng, sự kiện slip@71, collision@307, timeout@519 hiện đúng bước | scratch (`test_overlay.mp4`) |

Cách replay chính xác: `phase3_eval.py` gọi `set_seed(seed)` một lần cho mỗi (model, task) rồi dùng RNG torch lần lượt qua các batch 8 env. Nhiễu của episode k vì thế phụ thuộc vào mọi batch trước nó, nên script chạy lại các batch 0..k//8 (cùng NAS = 10, batch 8, expert bf16 như các eval gốc). Chỉ ghi frame ở các batch có episode được chọn. Mọi episode chạy lại đều được lưu npz và so với bản gốc (success và length) trong `replay_check_{arm}.csv`.

Episode được chọn (đều là init state held-out, không nằm trong `init_ids` của `fpo.yaml`/`fpo_v2.yaml`; nhãn lấy từ `episodes.csv`):

| Loại | seed | task | init | official | v1 θ3 |
|---|---|---|---|---|---|
| official thất bại, θ3 thành công | 1000 | 0 | 0 | thất bại, A (1 slip, 1 collision) | thành công |
| official thất bại, θ3 thành công | 2000 | 0 | 14 | thất bại, C (2 collision) | thành công |
| official thất bại, θ3 thành công | 2000 | 8 | 1 | thất bại, E (subtask 0.5) | thành công |
| official thành công, θ3 thất bại | 1000 | 0 | 43 | thành công | thất bại, C (2 slip) |
| official thành công, θ3 thất bại | 1000 | 8 | 29 | thành công | thất bại, E |
| official thành công, θ3 thất bại | 2000 | 0 | 20 | thành công | thất bại, A (6 slip) |
| cả hai thất bại | 1000 | 0 | 18 | thất bại, C (2 collision) | thất bại, C (2 collision) |
| cả hai thất bại | 2000 | 8 | 3 | thất bại, E (subtask 0.5) | thất bại, E (subtask 0.5) |
| cả hai thành công | 2000 | 8 | 0 | thành công | thành công |

Số episode phải chạy lại cho mỗi nhánh: seed 1000 task 0: 48 (batch 0–5), seed 1000 task 8: 32, seed 2000 task 0: 24, seed 2000 task 8: 8 → 112 episode/nhánh.

Ước tính thời gian (dựa trên log eval gốc: 5.5 s/ep task 0, 4.7 s/ep task 8; thu rollout v2 fp32 ≈ bf16, 63 s so với 62 s cho 16 rollout): khoảng 10 phút/nhánh + nạp model và ghi video → official + θ3 khoảng 22 phút, nhánh v2 thêm khoảng 11 phút.

Lưu ý: cấu hình gốc của official seed 1000 (`configs/phase3_r1.yaml`) đã bị xoá và không kiểm chứng được (xem D5). Nếu replay của official seed 1000 khác bản gốc thì có thể do cấu hình, không chỉ do tính không tất định. Nhánh v2 không có episode gốc ở seed 1000/2000 task 0/8, nên video v2 là episode mới và không có phép so khớp.

Chưa chạy: chờ loop `fpo_v2` xong hẳn (bao gồm eval và report).


## Phiên 01/11/2026 00:40–00:43: watcher tự chạy video sau v2 (theo yêu cầu)

| # | Thao tác | Tác động |
|---|---|---|
| 1 | Viết `improve/scripts/videos_index.py` (tạo `results/videos/index.md` từ `episodes_*.csv`, `replay_check_*.csv`; đánh dấu mọi episode khác bản gốc). Thử trên dữ liệu giả trong scratch; sửa lỗi kiểu dữ liệu (bool thành float khi gộp các nhánh) → đọc mọi cột dạng chuỗi | **SỬA REPO** (file mới) |
| 2 | Chạy thử `main()` của `phase3_eval_video.py` trên CPU, thay policy/env/rollout bằng episode thật đã lưu (`r2/eval/fpo_it3`, `r2/eval_s2000/fpo_it3`): 112 episode/nhánh, frame chỉ ghi ở 7 batch có episode được chọn, nhãn và sự kiện khớp `episodes.csv`, so khớp 0/112 khác | scratch |
| 3 | Viết `improve/scripts/videos_after_v2.sh` và chạy trong tmux `videos` lúc 00:42:25 (log `results/videos/run.log`, trạng thái `results/videos/watcher_status.txt`) | chờ, chưa dùng GPU |

Điều kiện bắt đầu: process loop (pid 2719389) đã dừng, **và** `loop.log` có dòng `=== [done]`, **và** có `outputs/fpo_v2/selection.json` và `summary.md` trong cả 5 thư mục `results/round2/fpo_v2/{seed3000,seed4000,seed5000,seed1000_other,seed1000_precision}`. Không đọc được trực tiếp exit code của loop (không phải process con; pipeline `| tee` không lưu mã). Vì `fpo_v2_loop.sh` chạy với `set -euo pipefail` và `=== [done]` là dòng cuối cùng, có dòng này tương đương exit 0. Nếu thiếu điều kiện nào: không chạy gì, ghi lý do và 30 dòng cuối `loop.log` vào WORKLOG.

Nếu đủ điều kiện: dừng tmux `fpo_v2_gpu`, để `outputs/fpo_v2/gpu.csv` không ghi thêm GPU của lần chạy video; ghi thời điểm vào `watcher_status.txt`. Sau đó chạy (a) official + v1 θ3, (b) v2 (đường dẫn từ `selection.json`, config sinh ra `results/videos/configs/videos_v2.yaml`, cùng 9 cặp episode), rồi chạy index. Phần chạy video chưa thử trên GPU (chỉ thử trên CPU như bước 2).

### 01/11/2026 — FPO v2: kết quả vòng lặp đầy đủ (tmux `fpo_v2`, 31/10 23:43:59 → 01/11 03:27:25)

Lệnh: `improve/scripts/fpo_v2_loop.sh` (đã duyệt). Tác động: SỬA REPO (output mới trong `improve/outputs/fpo_v2/`, `results/round2/fpo_v2/`); không sửa kết quả cũ. Tính lại thống kê gộp từ `results/round2/fpo_v2/seed{3000,4000,5000}/episodes.csv` và `seed1000_other/episodes.csv` (script trong scratchpad, chỉ đọc). Thời gian lấy từ `outputs/fpo_v2/timing.txt`.

**Theo iteration** (nguồn: `outputs/fpo_v2/iter_summary.csv`, `iter{k}/rollouts.csv`). Batch 128 ở mọi iteration; mọi iteration dừng sớm (clip_frac 10 bước cuối > 0.2), không iteration nào chạm trần 3 epoch.

| iter | bước | epoch dùng | clip_frac TB / 10 bước cuối | KL proxy vs θ0 (đầu → cuối) | thu (θ_k) task 0 / task 8 / tổng | thời gian thu / train (s) |
|---|---|---|---|---|---|---|
| 0 | 23 | 0.264 | 0.129 / 0.211 | 0 → 0.0325 | 55/128, 42/128, 97/256 = 37.9% | 1030 / 814 |
| 1 | 27 | 0.321 | 0.120 / 0.213 | 0.0270 → 0.0456 | 46/128, 60/128, 106/256 = 41.4% | 1080 / 814 |
| 2 | 16 | 0.193 | 0.136 / 0.213 | 0.0646 → 0.1825 | 47/128, 54/128, 101/256 = 39.5% | 763 / 735 |
| 3 | 18 | 0.228 | 0.154 / 0.226 | 0.1906 → 0.1986 | 50/128, 59/128, 109/256 = 42.6% | 763 / 719 |
| 4 (θ4, chỉ thu) | — | — | — | — | 38/128, 56/128, 94/256 = 36.7% | 768 / — |

**Quy tắc collapse** (`outputs/fpo_v2/selection.json`): không có bước giảm > 15 điểm (giảm lớn nhất 42.6 → 36.7 = 5.9 điểm, iter3→4) → chọn θ4 = `outputs/fpo_v2/iter3/train/pretrained_model`.

**Eval task 0, 8 × 50 init × seed 3000/4000/5000, expert fp32, gộp 300 episode/arm** (CI Wilson 95%; McNemar chính xác, ghép theo seed+task+init):

| arm | task 0 | task 8 | tổng | held-out (204) | init đã train (96) |
|---|---|---|---|---|---|
| official | 61/150 = 40.7% | 60/150 = 40.0% | 121/300 = 40.3% (35–46) | 84/204 = 41.2% | 37/96 = 38.5% |
| v1 θ3 | 62/150 = 41.3% | 36/150 = 24.0% | 98/300 = 32.7% (28–38) | 66/204 = 32.4% | 32/96 = 33.3% |
| v2 θ4 | 50/150 = 33.3% | 76/150 = 50.7% | 126/300 = 42.0% (37–48) | 86/204 = 42.2% | 40/96 = 41.7% |

Cặp (chỉ A / chỉ B, p): v2 vs official tổng 49/44 p = 0.68; task 0: 11/22 p = 0.080; task 8: 38/22 p = 0.052; held-out 32/30 p = 0.90. v1 vs official 40/63 p = 0.030 (task 8: 16/40 p = 0.002). v2 vs v1 67/39 p = 0.008 (task 8: 47/7 p < 0.001; task 0: 20/32 p = 0.13). Từng seed: xem `results/round2/fpo_v2/seed{3000,4000,5000}/summary.md`.

**Quên (8 task khác × 20, seed 1000)**: official 127/160 = 79.4%, v1 θ3 111/160 = 69.4% (cặp 11/27, p = 0.014), v2 θ4 108/160 = 67.5% (cặp 9/28, p = 0.003); v2 vs v1 16/19 p = 0.74. Theo task (official / v1 / v2): 1: 17/18/12, 2: 16/13/11, 3: 20/20/20, 4: 13/10/12, 5: 18/20/17, 6: 9/7/9, 7: 16/7/10, 9: 18/16/17.

**Kiểm tra độ chính xác (không dùng để chọn)**: official, seed 1000, task 0+8: expert bf16 (`outputs/eval_baseline`, phiên bản code không kiểm chứng được) 44/100 vs expert fp32 43/100; cặp 9/10, p = 1.00 (`results/round2/fpo_v2/seed1000_precision/summary.md`).

Nhận xét: v2 không còn kém official như v1 trên tổng task đã train (42.0% vs 40.3%, không có ý nghĩa thống kê), nhưng là đánh đổi giữa task: task 8 tăng (+10.7 điểm, p = 0.052), task 0 giảm (−7.3 điểm, p = 0.080). Thu θ4 cũng cho thấy task 0 giảm (38/128). KL proxy tăng nhanh ở iter2 (0.065 → 0.18) dù có neo BC. Quên trên 8 task khác có ý nghĩa thống kê cho cả v1 và v2 (≈ −10 đến −12 điểm).


## 01/11/2026 03:50: video replay xong (watcher tự chạy)

Loop fpo_v2 kết thúc sạch. Watcher dừng tmux `fpo_v2_gpu` rồi chạy (a) official + v1 θ3 (exit 0), (b) v2 = `/home/student/ffi26/improve/outputs/fpo_v2/iter3/train/pretrained_model` (exit 0), index (exit 0).
Kết quả: `results/videos/` (`index.md`, `replay_check_*.csv`, `episodes_*.csv`, `raw/`). Log: `results/videos/run.log`, `watcher_status.txt`.

## 01/11/2026: viết hướng dẫn đọc kết quả

Mục tiêu: bạn yêu cầu hướng dẫn đọc repo để phân tích kết quả; bạn đồng ý lưu thành file và tính lại số D_nas10.

| # | Lệnh / thao tác | Lý do | Tác động |
|---|---|---|---|
| 1 | `ls`, `find`, `cat`/`head` trên `readme.md`, `docs/*.md`, `results/round2/{README,summary}.md`, header CSV, `videos/index.md` | nắm cấu trúc repo và các nguồn kết quả | chỉ đọc |
| 2 | viết `scratchpad/agg_nas10.py`, chạy `python3 -I agg_nas10.py .../runs/D_nas10/lerobot_smolvla_libero` | kiểm tra 72,8% (readme) so với 73,6% (analyse) | scratch; chỉ đọc kết quả |
| 3 | **Ghi `docs/huong_dan_doc_ket_qua.md`** (file mới) | sản phẩm bạn yêu cầu | **SỬA REPO** |
| 4 | ghi mục này vào `docs/WORKLOG.md` | quy tắc nhật ký | **SỬA REPO** |

Kết quả bước 2 (30 job): seed 1000 368/500 = 73,6%, seed 1001 360/500 = 72,0%, seed 1002 364/500 = 72,8%, gộp 1092/1500 = 72,8%.
Không mâu thuẫn: 73,6% trong `analyse.md` là số của riêng seed 1000. Không chạy script nào của repo; không sửa file nào trong `~/ffi26`.

## 01/11/2026 ~07:50–08:30: kiểm tra và sửa failure labeler (CPU, không dùng GPU)

Yêu cầu: kiểm tra `improve/src/failure_labels.py` (`label_episode`, `subtask_stats`), sửa chỗ có vấn đề, chạy trên mọi episode eval đã lưu. Mọi lệnh Python chạy với `CUDA_VISIBLE_DEVICES=""`. GPU lúc bắt đầu: 96 %, 12,6 GB do `run.py` của người dùng khác; **không thấy process FPO v3 nào và không có `outputs/fpo_v3`**.

Dữ liệu: `improve/outputs/fpo_v2/eval_s{3000,4000,5000}/{official,fpo_v1_it3,fpo_v2_sel}/task{0,8}` (300 ep / arm) và `eval_s1000/<arm>/task{1..7,9}` (160 ep / arm): 1380 episode, 691 thành công, 689 thất bại. Ca kiểm tra bước 1: `results/videos/raw/v2/s2000/task8/ep001|ep003.npz` (v2 θ4, seed 2000).

| # | Lệnh / thao tác | Lý do | Tác động |
|---|---|---|---|
| 1 | `cat`/`sed` trên `src/failure_labels.py`, `src/reward.py`, `src/reward_r2.py`, `scripts/r2_report.py`, `scripts/phase3_eval_video.py`, config eval/video | hiểu labeler, nơi lưu log, cách render | chỉ đọc |
| 2 | `scratchpad/step1.py`, `step1b.py` trên ep001, ep003 | bước 1 | scratch; chỉ đọc |
| 3 | `scratchpad/load_all.py` (gom 1380 npz vào `scratchpad/all_eps.pkl`), `step2.py`, `step2b.py`, `regions.py` | bước 2 | scratch; chỉ đọc |
| 4 | **Thêm** `label_episode_v2` + hàm phụ vào cuối `src/failure_labels.py` (126 dòng cũ giữ nguyên, kiểm bằng `diff` với bản sao `scratchpad/failure_labels.orig.py`) | bước 3. Không sửa `label_episode` vì `fpo_collect.py` (v3) và các báo cáo cũ dùng nó. Không sửa `reward.py` / `reward_r2.py` (chỉ import `detect_events`) | **SỬA REPO** |
| 5 | **Tạo** `improve/tests/test_failure_labels_v2.py`, chạy `python -m pytest tests/test_failure_labels_v2.py -q` | unit test bằng log giả | **SỬA REPO**; 14/14 passed |
| 6 | **Tạo** `improve/scripts/failure_labels_run.py`, chạy 3 lần (sửa: thư mục render riêng cho mỗi config, cách chọn mẫu tay, cột index thừa) | bước 4–5 | **SỬA REPO**: ghi `improve/outputs/failure_labels/` |

### Bước 1: task 8, v2 θ4, seed 2000

Trong log, `moka_pot_2` lên bếp ở bước 174; vật còn lại là `moka_pot_1`.

| | init 1 (thành công, 418 bước) | init 3 (thất bại, 520 bước) |
|---|---|---|
| a) EE → tâm `moka_pot_1`: min; trung bình 270–519 | 0,073 m (bước 357); 0,145 m | 0,086 m (bước 520); **0,090 m** |
| b) gripper 250–519: số lần mở→đóng; số bước đóng | 1; 83/168 | **0; 0/270** (cả episode: 1 lần đóng, 69 bước, cho `moka_pot_2`) |
| c) \|a_xyz\| trung bình 0–250 → 270–519 | 0,373 → 0,508 | 0,469 → **0,239** |
| d) EE dịch mỗi bước 270–519 | 6,29 mm | **0,28 mm** (max 0,86, trung vị 0,24) |
| e) `label_episode` cũ | S | **E** (B, E; đoạn đứng dài nhất 253 bước) |

Init 3 trong 270–519: hướng action gần như không đổi (cos với hướng trung bình: trung vị 0,995), chủ yếu đi xuống (−z); `contact` với `moka_pot_1` ở 251/251 hàng log 270–520 (chạm từ bước 266); EE cao hơn tâm ấm 6,9 cm; ấm dịch < 7 mm. Mỗi đơn vị \|a_xyz\| cho 12,5 mm dịch EE ở 0–250 nhưng chỉ 1,2 mm ở 270–519.
**Kết luận:** policy **không** ra action ≈ 0. Nó ra action khác 0, cùng một hướng, nhưng EE bị chặn khi đang chạm `moka_pot_1` với gripper mở (đè lên ấm). Video chưa được xem để xác nhận.

### Bước 2: các nghi vấn (1380 episode)

| # | Nghi vấn | Trả lời | Số liệu |
|---|---|---|---|
| 1 | A bỏ sót khi đóng 1 lần rồi giữ | **CÓ** | Trong 600 (episode thất bại × vật mục tiêu chưa xong, chưa nhấc), 19 ca có đúng 1 lần đóng gần vật (< 0,10 m) và giữ đóng ≥ 20 bước gần vật, nhưng A không bắt. Thêm 45 ca không có lần đóng nào trong 0,10 m nhưng giữ đóng ≥ 20 bước trong 0,15 m. (Ca init 3 ở bước 1 không thuộc loại này: gripper không đóng lần nào.) |
| 2 | NEAR_DIST = 0,10 m đến tâm quá chặt với moka pot | **KHÔNG** với moka pot, **CÓ** với vật khác | Khoảng cách EE → tâm khi `grasp` = True (p5/p50/p95): moka_pot 0,052/0,069/0,087, 0 % > 0,10; khi `contact`: 14 % > 0,10. Init 3: 0,086–0,090 m. Nhưng black_book khi grasp 0,105/0,107/0,119 (**100 % > 0,10**, nên A không bao giờ bắt được cho sách), porcelain_mug 10 %, white_yellow_mug 8 % > 0,10. |
| 3 | Giỏ có trong goal_obj_mask ở task 0/1/7; C chỉ dựa vào moved_mask mà không cần chạm | Giỏ: **CÓ** trong mask. C không cần chạm: **KHÔNG**, nhưng C vẫn sai | goal_obj_mask: t0 {alphabet_soup, tomato_sauce, basket}; t1 {cream_cheese, butter, basket}; t2 {moka_pot}; t3 {akita_black_bowl}; t4 {porcelain_mug, white_yellow_mug, plate_1, plate_2}; t5 {black_book}; t6 {porcelain_mug, plate_1, chocolate_pudding}; t7 {alphabet_soup, cream_cheese, basket}; t8 {moka_pot_1, moka_pot_2}; t9 {white_yellow_mug}. Fixture (bếp, tủ, lò vi sóng, khay, vùng bàn) không phải vật di động nên không có trong log; không có gì để thêm. `moved_mask` có yêu cầu chạm (latch) nhưng đo dịch chuyển **từ vị trí reset**: 233/239 vật bị gán C đã dịch > 5 cm trước lần chạm đầu, và **113/239 không dịch > 5 cm sau lần chạm đầu** (chủ yếu task 0: orange_juice 89, ketchup 10, milk 7, xuất phát ở z 0,89–1,05 m rồi rơi xuống 0,51 m trước bước 20). |
| 4 | D sai do predicate nhấp nháy | **CÓ** | Trong 691 episode thành công: 78 lần True→False→True với đoạn False ≤ 5 bước, ở 57 episode; 58 episode thành công có lab_D. Trong 64 thất bại có lab_D, 45 chỉ do lật ngắn (True < 5 hoặc False ≤ 5 bước). |
| 5 | Ngưỡng stall bắt episode thành công | **CÓ, ít** (7,7 %), và **không phải** tiếp cận chậm trong không gian trống | 53/691 thành công có E (thất bại: 506/689). 56 đoạn đứng trong thành công: 98 % đoạn có chạm vật > 50 % số bước; trung bình 0,65 mm/bước; đoạn dài nhất trung vị 71 bước, max 222. Đó là thao tác chậm khi đang chạm vật. Với v2: E_free = 0/691 thành công, E_contact = 53/691. |

### Bước 3: đã sửa (`label_episode_v2`; tên nhãn cũ giữ nguyên, thêm F, G, E_contact, E_free)

- **A** (nghi vấn 1, 2): "gần" = EE < 0,10 m tới tâm **hoặc** gripper contact/grasp với vật (log không có bounding box; contact là tín hiệu điểm gần nhất có sẵn). A = (≥ 2 lần đóng khi gần) **hoặc** (đóng ≥ 20 bước khi gần), vật chưa xong và chưa từng được nhấc.
- **C** (nghi vấn 3): vật không phải mục tiêu bị grasp, hoặc có sự kiện `collision` của `reward_r2` (dịch > 5 cm **tính từ vị trí lúc chạm đầu**). Không cần thêm vùng đích vào mask.
- **D** (nghi vấn 4): chỉ tính khi predicate True ≥ 5 bước rồi False ≥ 5 bước.
- **E**: tách theo đoạn đứng. Đoạn là E_contact nếu ≥ 50 % số bước có gripper đóng hoặc EE "gần" một vật (định nghĩa như A), ngược lại là E_free. Ngưỡng 2 mm / 50 bước giữ nguyên (nghi vấn 5: chỉ bắt đoạn có chạm vật ở thành công).
- **F_drop**: dùng sự kiện `drop` của `reward_r2`: vật mục tiêu đã giữ và nhấc, được thả, rơi > 5 cm trong 30 bước, predicate vẫn False.
- **G_misplace**: vật mục tiêu đã từng được nhấc, được thả, đứng yên (< 1 mm/bước trong 10 bước, không còn bị giữ), predicate False, cách đích < 10 cm theo xy. Đích là vật di động (giỏ, đĩa) thì lấy vị trí vật trong log; đích là fixture thì dùng tâm **ước lượng từ log** (vị trí của vật tại hàng cuối predicate True, trung bình theo init). Lệch chuẩn giữa các init ≤ 2,2 cm (`outputs/failure_labels/region_centres.csv`). Đánh giá được ở 689/689 thất bại.
- **Thứ tự**: A > C > F > G > D > E_contact > E_free > B > O. Lý do: cơ chế trên vật mục tiêu trước (A: chưa lấy được vật); rồi nhầm vật (C); rồi làm mất vật đang giữ (F) hoặc đặt lệch (G). D đứng sau F/G vì rơi hay đặt lệch một vật đã xong cũng làm predicate lật. E là triệu chứng (E_contact trước E_free). B/O chỉ mô tả tiến độ.
- Unit test 14/14 (`tests/test_failure_labels_v2.py`). Sanity check trên 2 episode bước 1: init 1 → S; init 3 → **E_contact** (đoạn 267–520 loại contact) + B. Cũ: E + B.

### Bước 4: kết quả (`improve/outputs/failure_labels/summary.md`, `labels.csv`, ...)

v3: **chưa có dữ liệu eval** (không có thư mục `fpo_v3_sel` trong `outputs/fpo_v2/eval_s*`, không có `outputs/fpo_v3`), nên không chạy.

So với nhãn cũ trên 689 thất bại: lab_C 212 → 114, lab_D 64 → 23, lab_A 334 → 399. Nhãn chính cũ E (194 ep) chuyển thành A 43, F 17, G 14, E_contact 109, E_free 11. Nhãn cũ D (39 ep) chỉ còn 5 ở D (18 → G, 10 → E_contact). Trên 691 thành công: E_free 0, E_contact 53, F 9 (rơi rồi gắp lại), C 4, D 4, G 1.

Task 8 (150 ep / arm, 3 seed), số thất bại theo số ấm đã lên bếp (0 / 1 / 2):

| arm | thất bại | 0 ấm | 1 ấm | 2 ấm (D) | lab_A | lab_E_contact | E_contact ∧ B |
|---|---|---|---|---|---|---|---|
| official | 90 | 14 | 73 | 3 | 25 | 55 | 39 |
| v1 θ3 | 114 | 53 | 59 | 2 | 59 | 88 | 37 |
| v2 θ4 | 74 | 50 | 22 | 2 | 42 | 67 | 20 |

Task 1, 2, 7 (seed 1000, 20 init, cặp theo init): episode thất bại có C, official / v2: task 1: 1 / 1 (milk_1 cả hai); task 2: 0 / 0; task 7: 0 / 1 (tomato_sauce_1); McNemar p = 1,00. Nhãn chính của v2: task 1 A 6/8, task 7 A 8/10.

Task 0, thất bại có E_free: official 14/89, v1 9/88, **v2 51/100** (đoạn đứng free bắt đầu trung vị ở bước 336).

### Bước 5: kiểm chứng bằng tay

`outputs/failure_labels/manual_check.csv`: 20 thất bại (seed 20261101). Xoay vòng qua nhãn chính v2; mỗi lượt lấy arm được chọn ít nhất. Thứ tự dòng đã xáo trộn; cột `nhan_tay`, `nhan_script`, `ghi_chu` để trống. Phân bố: arm 7/7/6; mỗi nhãn 2–3; task 8: 14, task 0: 3, task 4: 2, task 1: 1. Chưa có video nào (0/20). Lệnh render: `outputs/failure_labels/render_commands.sh` (5 config trong `render/`, cùng thiết lập với eval gốc: NAS 10, batch 8, expert fp32, n_episodes 50 hoặc 20). **Chưa chạy**: cần GPU. Lưu ý: header video có in sự kiện drop/slip/collision của `reward_r2`, có thể gợi ý nhãn khi xem.


## 01/11/2026 07:40–08:05: FPO v3, cài đặt + smoke test, khởi chạy loop

Yêu cầu: FPO v3 = v2 + (1) demo replay trong số hạng BC, (2) chuẩn hoá advantage theo task + minibatch cân bằng 2 task, (3) tuỳ chọn: gấp đôi số cặp (τ, ε) cho ratio nếu tốn thêm < 30% thời gian/bước. Mọi thứ khác giữ như v2. Kế hoạch đã báo và được duyệt (kèm: dùng lại dữ liệu thu iteration 0 của v2). Không commit.

### File mới (SỬA REPO: chỉ thêm file, không sửa file v2)

- `improve/scripts/fpo_build_demo_cache.py`: lấy mẫu demo LIBERO-10, đưa qua đúng preprocessor của checkpoint official (như `lerobot_train._preprocess_dataset_batch`), lưu cùng định dạng với `decisions/g*.npz`.
- `improve/scripts/fpo_v3_demo_check.py`: smoke test bắt buộc về định dạng demo.
- `improve/scripts/fpo_train_v3.py`: bản sao của `fpo_train_v2.py` cộng 3 thay đổi, log thêm L_CFM trên demo probe theo từng task.
- `improve/scripts/fpo_v3_loop.sh`, `improve/scripts/fpo_v3_pooled.py` (thống kê theo cặp, tính lại từ `ep*.meta.json`), `improve/scripts/v2_forget_wrong_object.py`.
- Config: `results/round2/configs/fpo_v3.yaml`, `eval_fpo_v3_s{1000,3000,4000,5000}.tmpl.yaml`, `report_fpo_v3_{s3000,s4000,s5000,other}.yaml`, `results/videos/v2_forget/configs/videos_v2_forget.yaml`.
- Dữ liệu: `improve/data/fpo_v3/demo_train.npz` (2000 mẫu = 200/task), `demo_probe.npz` (640 mẫu = 64/task, lấy từ 6 episode/task để riêng, không dùng để train), `demo_cache_info.json`, `demo_check.json`, `demo_vs_rollout.png`.

### (1) Demo replay: đã dùng DEMO THẬT (không dùng phương án self-distillation)

Nguồn: `improve/data/r2/demos_libero10` (379 episode của `lerobot/libero`, đủ 10 task; đây là dữ liệu train của `lerobot/smolvla_libero`). `task_index` của demo KHÁC task id của env. Ánh xạ theo câu lệnh ngôn ngữ (`demo_cache_info.json`): demo→env {0: 4, 1: 6, 2: 9, 3: 2, 4: 7, 5: 0, 6: 8, 7: 1, 8: 3, 9: 5}. Chunk 50 action liên tiếp (delta_timestamps i/fps, fps 10), `action_is_pad` → `n_exec` = số bước hợp lệ.

Smoke test (`python scripts/fpo_v3_demo_check.py --config ~/ffi26/results/round2/configs/fpo_v3.yaml --data outputs/fpo_v2/iter0`, kết quả `improve/data/fpo_v3/demo_check.json`). θ0 fp32 giống lúc train; mọi biến thể dùng cùng (τ, ε) theo chỉ số mẫu, 128 mẫu/tập:

| tập | L_CFM(θ0), trung bình (median) |
|---|---|
| rollout thành công task 0, bước đã thực thi (≤ 10) | 0.0373 (0.0144) |
| rollout thành công task 8 | 0.0365 (0.0128) |
| demo task 0, 10 bước đầu | 0.0655 (0.0225) |
| demo task 8, 10 bước đầu | 0.0499 (0.0165) |
| demo cả 10 task, 10 bước đầu / cả chunk hợp lệ | 0.0700 (0.0227) / 0.0647 (0.0247) |
| đối chứng: ảnh demo xoay 180° | 2.077 (1.166) |
| đối chứng: đổi camera1 ↔ camera2 | 1.567 (0.855) |
| đối chứng: token ngôn ngữ của task khác | 0.947 (0.358) |

Loss trên demo thấp, cùng cỡ với rollout thành công (khoảng 1.9 lần; rollout là action do chính θ0 sinh ra nên dự kiến thấp hơn). Cả 3 đối chứng sai định dạng đều cao hơn 13–30 lần, và cao hơn trên 100% mẫu (so theo cặp). Kiểm tra khác: token ngôn ngữ của demo task 0 và task 8 GIỐNG HỆT token trong rollout v2 iter0 (mỗi bên 1 chuỗi duy nhất). Khoảng cách pixel trung bình giữa ảnh đầu episode demo và rollout: camera1 task 0 là 8.7 khi giữ nguyên vs 46.5 khi xoay 180°; task 8 là 6.4 vs 94.0. Trung bình/độ lệch chuẩn state đã chuẩn hoá của demo và rollout cùng cỡ (`demo_check.json`). Kết luận: định dạng khớp, ảnh demo dùng nguyên (env processor đã xoay ảnh sim 180° lúc eval, `lerobot/processor/env_processor.py:59`).

Quan sát phụ (không ảnh hưởng v3 so với v2): ở ảnh rollout bước 0 của task 0 thiếu vài vật (milk, orange juice, ketchup); các vật này xuất hiện từ bước 10 (`demo_vs_rollout.png`, so với ảnh bước 0/10/20/50 của cùng một rollout). Có vẻ là lỗi render khung đầu sau reset; ảnh hưởng như nhau tới v2 và v3.

Số hạng train: mỗi bước + λ_demo · L_CFM(demo), λ_demo = λ_bc = 5, 32 mẫu demo/bước (3 mẫu/task cho cả 10 task + 2 mẫu cho 2 task bốc ngẫu nhiên không lặp), loss trên cả chunk hợp lệ (đúng mục tiêu lúc train gốc), (τ, ε) mới mỗi bước.

### (2) Chuẩn hoá advantage theo task

Â = clip((A − mean_t)/std_t, −5, 5), tính riêng cho task 0 và task 8 ở mỗi iteration (v2: A/std chung). Mỗi bước gradient gồm 64 quyết định task 0 + 64 quyết định task 8 (v2: một task/bước), lấy từ hai luồng xáo trộn riêng. Một epoch = Σ_t ⌊n_t/128⌋ bước như v2, tối đa 3 epoch. BC trên rollout thành công: 16 mẫu/task (tổng 32 như v2). Ví dụ trên dữ liệu iter0, 400 quyết định/task: mean/std task 0 = −0.0006/0.160, task 8 = −0.039/0.267.

### (3) Gấp đôi số cặp (τ, ε) cho ratio: BỎ

Đo trên dữ liệu v2 iter0, 6 bước, bỏ qua bước đầu. 1 cặp/quyết định: 6.29 s/bước, đỉnh bộ nhớ 11.4 GB. 2 cặp (ratio = exp(c · trung bình 2 cặp của l_old − l_new)): với micro-batch 32 thì hết bộ nhớ GPU; với micro-batch 16 (cùng phép tính) thì 10.71 s/bước, tức +70% > 30%. Giữ `pairs_per_step: 1` như v2.

### Smoke test train

`fpo_train_v3.py --max_steps 12 --max_decisions 400 --set kl_every=5` trên `outputs/fpo_v2/iter0` với θ0 (output ở scratchpad, không lưu vào repo): bước 1 có ratio = 1.000000 (min 0.9999984, max 1.0000037); chạy đủ 12 bước; KL proxy 0 → 0.0110; demo probe (8 task không train) 0.05730 → 0.05779; 6.29 s/bước; đỉnh bộ nhớ 10.7 GB.

### Kiểm tra script thống kê

`fpo_v3_pooled.py` chạy trên dữ liệu có sẵn (chưa có v3) cho lại đúng số của v2: official 121/300, v2 126/300, cặp 49/44 p = 0.68; quên: official 127/160, v2 108/160, cặp 9/28 p = 0.0026; từng task khớp với mục v2 ở trên. File thử đã xoá.

### Khởi chạy

08:05: tmux `fpo_v3`, `bash scripts/fpo_v3_loop.sh` (log `improve/outputs/fpo_v3/loop.log`, thời gian `timing.txt`). iter0: kiểm tra trường collect / reward / n_action_steps / expert_fp32 / init_policy của `fpo_v2.yaml` và `fpo_v3.yaml` giống nhau → symlink `decisions/`, `logs/` và copy các file csv/json nhỏ từ `outputs/fpo_v2/iter0`. Eval chỉ chạy arm v3: seed 1000 (quên) TRƯỚC, sau đó seed 3000/4000/5000. official và v2 dùng lại episode của `outputs/fpo_v2/eval_s*`.


## 01/11/2026 07:45–08:10: video v2 θ4 thất bại ở task 1 và 7 (eval quên, seed 1000)

Lệnh: tmux `v3_videos`, `python scripts/phase3_eval_video.py --config ~/ffi26/results/videos/v2_forget/configs/videos_v2_forget.yaml` (chạy trong lúc GPU rảnh trước khi khởi chạy loop v3). Tác động: SỬA REPO (output mới trong `results/videos/v2_forget/`). Kết quả: 18 video (task 1: 8, task 7: 10); `replay_check_v2.csv`: 0/40 episode replay khác bản gốc. Sau đó chạy `python scripts/v2_forget_wrong_object.py` (chỉ đọc log eval gốc, ghi `results/videos/v2_forget/index.md`, `wrong_object.csv`).

Vật của task 0 không phải vật mục tiêu: task 1 có alphabet soup + tomato sauce; task 7 chỉ có tomato sauce (alphabet soup là vật mục tiêu của task 7). Bộ phát hiện tự động (`src/reward.py`; CHƯA xem video để xác nhận):

| arm | task | thất bại | chạm vật task 0 (trong số thất bại) | nhấc vật task 0 (trong số thất bại) | chạm vật task 0 (mọi episode) |
|---|---|---|---|---|---|
| official | 1 | 3/20 | 0/3 | 0/3 | 0/20 |
| v2 θ4 | 1 | 8/20 | 0/8 | 0/8 | 0/20 |
| official | 7 | 4/20 | 1/4 | 0/4 | 2/20 |
| v2 θ4 | 7 | 10/20 | 4/10 | 1/10 (init 6, tomato sauce) | 4/20 |

Theo bộ phát hiện: ở task 1, v2 không chạm vật nào của task 0 (5/8 thất bại có nhấc cream cheese/butter, đúng vật mục tiêu). Ở task 7, v2 chạm tomato sauce ở 4/10 thất bại nhưng chỉ nhấc nó ở 1 episode (init 6). Như vậy dữ liệu log không cho thấy việc "gắp nhầm vật của task 0" là cơ chế chính của việc quên; cần xem video để xác nhận (danh sách trong `index.md`).

Xem tay 1 trường hợp "nhấc" (task 7, init 6, `v2_t7_init6_s1000_fail.mp4`): bộ phát hiện đánh dấu tomato sauce "nhấc" ở bước 357–366 (10 bước). Ở 3 khung đã xem (bước 327, 366, 367), gripper ép vào lon tomato sauce, cạnh lon alphabet soup đã bị đổ, và không thấy rõ là đang gắp. Có thể là lon bị nghiêng/đẩy lên khi va chạm (header ghi `collision tomato_sauce_1 @357`) chứ không phải gắp có chủ đích. 17 video còn lại CHƯA xem.

## 01/11/2026 ~08:05–08:20: câu hỏi tiếp theo về labeler, video "mù", commit

| # | Lệnh / thao tác | Lý do | Tác động |
|---|---|---|---|
| 1 | `grep`/`sed` trên `scripts/fpo_collect.py`, `fpo_train_v3.py`, `fpo_train_v2.py`, `fpo_v3_loop.sh`, `src/reward_r2.py` | `label_episode` dùng vào việc gì trong v3 | chỉ đọc |
| 2 | `scratchpad/t8_two.py`, `t8_two_b.py` (CPU) | 7 thất bại task 8 có cả hai predicate `on` từng True | scratch; chỉ đọc |
| 3 | **Sửa** `scripts/phase3_eval_video.py`: thêm `--blind` (mặc định tắt; hành vi cũ giữ nguyên) | video cho kiểm chứng tay | **SỬA REPO** |
| 4 | **Sửa** `scripts/failure_labels_run.py` (lệnh render có `--blind`), chạy lại (CPU) | sinh lại `render_commands.sh`; `manual_check.csv` giữ nguyên (cùng seed, cùng dữ liệu) | **SỬA REPO** (`outputs/failure_labels/`) |
| 5 | `scratchpad/blind_test.py`: `annotate` trên frame giả, xem 2 ảnh header (CPU) | kiểm tra chế độ blind chỉ in step và Φ | scratch |
| 6 | **Tạo** `outputs/failure_labels/render_watch.sh`, chạy trong tmux `fl_render_watch` | render sau khi v3 xong | chạy nền; chưa dùng GPU |
| 7 | `git commit` các file labeler (bạn yêu cầu) | bước 4 | **COMMIT** |

**1. `label_episode` trong v3 chỉ để log.** `fpo_collect.py:105` gọi `fl = label_episode(...)`, `:109` ghi `"primary": fl["primary"]` vào `roll_rows`, `:119` ghi ra `rollouts.csv`. Advantage tính từ `group_segment_labels` của `src/reward_r2.py` (`fpo_collect.py:84`, `:90–93`). `reward_r2.py` chỉ import `reward` (dòng 46), không import `failure_labels`. Trong train, `rollouts.csv` chỉ được đọc cột `success`: `fpo_train_v2.py:94–95` (`load_success_decisions`, chọn decision của rollout thành công cho BC, được `fpo_train_v3.py:39,94` dùng lại) và `fpo_train_v3.py:313–316` (đếm tỉ lệ thành công cho `train_info.json`). `grep primary` trong `fpo_train_v3.py`, `fpo_train_v2.py`, `fpo_v3_pooled.py`, `fpo_v3_loop.sh`: không có. Như vậy nhãn thất bại **không** ảnh hưởng lọc, advantage hay sampling. Collect của v3 từ iteration 1 sẽ import `failure_labels.py` đã sửa: `label_episode` không đổi, chỉ thêm `from reward_r2 import detect_events` ở cấp module (`fpo_collect.py` vốn đã import `reward_r2`).

**2. Task 8, 7 thất bại có cả hai ấm từng lên bếp** (cột "2 ấm" trong mục trước; `n_pred_new_ever_true = 2`). Cả 7 cùng một mẫu: `moka_pot_2` lên bếp trước (True 14–38 bước, bắt đầu ở bước 170–252), rồi rời vùng bếp; sau đó `moka_pot_1` lên bếp (bước 383–516) và còn True đến cuối. **Không có hàng nào hai ấm cùng True.** Lúc kết thúc, `moka_pot_2` lệch về −y so với chỗ nó đứng khi True, cách tâm ước lượng (−0,098, −0,227) từ 0,027 đến 0,128 m.

| arm | seed | init | `moka_pot_2` True | dịch sau khi thả: gripper chạm / không chạm | z cuối (reset 0,966) | nhãn v2 |
|---|---|---|---|---|---|---|
| official | 3000 | 13 | 246–284 (nhấp nháy 246–270) | 0,000 / 0,057 m | 0,989 | G (+D) |
| official | 3000 | 28 | 184–198 | 0,000 / 0,011 m | 0,997 | G (+D, E_contact) |
| official | 4000 | 20 | 252–265 | 0,009 / 0,068 m | 0,978 | G (+D, E_contact) |
| v1 θ3 | 5000 | 19 | 180–211 | 0,015 / 0,121 m | 0,962 | G (+D) |
| v1 θ3 | 5000 | 25 | 189–216 | 0,000 / 0,099 m | 0,964 | D (cách tâm 0,120 m > 0,10 nên không phải G) |
| v2 θ4 | 5000 | 3 | 170–206 | 0,002 / 0,047 m | 0,997 | G (+D, E_contact) |
| v2 θ4 | 5000 | 22 | 190–223 | 0,036 / 0,035 m | 0,997 | G (+D, E_contact) |

Diễn giải: ấm thứ nhất được đặt lên bếp nhưng không ở lại. Phần lớn quãng dịch xảy ra khi **gripper không chạm ấm**. Ở 2 ca v1, z cuối ≈ z reset, tức ấm đã xuống mặt bàn. Log chỉ đo tiếp xúc của gripper (`check_contact(gripper, obj)`), nên **không xác định được** ấm tự trượt hay bị phần khác của cánh tay đẩy; cần xem video. Ca official s3000 init 13 có `slip` ở bước 120 (`reward_r2`).

**3. Video mù.** `--blind`: header chỉ có `step i/T  Phi x` (đã xem ảnh thử); `episodes_<arm>.csv` để trống `primary`, `subtask_completion`, `events`. Tên file vẫn có arm và `_fail`, như trong `manual_check.csv`. **Chưa render.** Lúc 08:10, v3 đang train iter0 (pid 3573373, 3,8 GB). Train v2 cần tới 11,4 GB (`outputs/fpo_v2/iter*/train/train_info.json`, `torch_max_mem_allocated_gb`), nên chạy song song có nguy cơ OOM làm hỏng v3. Đổi batch size thì replay không còn khớp episode gốc (`phase3_eval_video.py`: nhiễu policy phụ thuộc thứ tự batch). Vì vậy render chạy sau v3: tmux `fl_render_watch` chạy `outputs/failure_labels/render_watch.sh`, kiểm tra mỗi 2 phút; khi không còn process `fpo_v3_loop.sh` và GPU trống ≥ 6000 MiB thì chạy `render_commands.sh`. Trạng thái: `render_watch_status.txt`, log: `render.log`, video: `outputs/failure_labels/videos/<arm>_<s1000|s3k5k>/`, kiểm tra tất định: `replay_check_<arm>.csv` trong từng thư mục.
