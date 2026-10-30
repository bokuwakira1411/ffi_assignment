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

## Quy tắc Claude tuân theo từ nay trong repo này

1. Trước khi chạy bất cứ thứ gì: nêu mục tiêu và kế hoạch (các bước, file sẽ đọc, file sẽ ghi). Nếu không chỉ là đọc nhanh thì chờ bạn duyệt.
2. Sau khi chạy: ghi thêm một mục vào file này, gồm các lệnh, lý do, tác động (chỉ đọc / scratch / SỬA REPO) và kết quả.
3. Không sửa kết quả trong `~/ffi26` và không chạy các script của repo (`run_queue.sh`, `train_more.sh`, `collect_results.py`, ...) khi chưa được bạn cho phép rõ ràng.
4. Chat với bạn bằng tiếng Anh; mọi file báo cáo (md trong repo) viết bằng tiếng Việt.

