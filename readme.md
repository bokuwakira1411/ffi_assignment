# FFI26: Tái lập và phân tích SmolVLA trên LIBERO

Hạn nộp: **08:00 sáng thứ Ba 03/11/2026**. Cập nhật lần cuối: tối thứ Sáu 30/10/2026.

Bản kế hoạch gốc (viết ngày 29/10) nằm trong lịch sử git (`git show 3d7a55f:readme.md`). File này ghi lại **những gì đã làm**
và **những gì làm tiếp**. Mọi con số dưới đây lấy từ `differences.md` và `failure_analysis.md`; hai file đó ghi nguồn
(đường dẫn file kết quả) cho từng số.

## 1. Tóm tắt tình trạng

| Phần | Trạng thái | File chính |
|---|---|---|
| Setup (5070 Ti, LeRobot 0.6.2) | **Xong** | `setup_lerobot.sh`, `check_env.py`, `ffi26/ffi26/results/{versions.txt,pip_freeze.txt,environment.yml,env_report.json}` |
| Reproduce: eval checkpoint chính | **Xong** | `run_queue.sh`, `ffi26/ffi26/results/runs/` |
| Ablation (n_action_steps, control_mode) | **Xong** | `ffi26/ffi26/results/runs/{D_nas1,D_nas50,F_spatial_nas50,F_spatial_abs}` |
| Probe pipeline | **Đã chạy** | `probe_smolvla.py`, `ffi26/ffi26/logs/probe_full.log`, `ffi26/ffi26/results/probe_report.json` |
| Ghi khác biệt so với bản gốc | **Xong** | `differences.md` |
| Phân tích lỗi từ video | **Xong** (bản để bạn kiểm tra) | `failure_analysis.md` |
| Study: literature matrix, diffusion vs flow matching | **Chưa làm** | danh sách bài đọc trong `DAY1.md` mục D |
| Improve | **Chưa chạy** (mới có script) | `train_more.sh` |
| Report, slide, đóng gói | **Chưa làm** | |

Nhật ký từng lần Claude chạy lệnh trong repo: `WORKLOG.md`.

## 2. Những gì đã làm

### 2.1 Mô hình và cấu hình
- Checkpoint chính: `lerobot/smolvla_libero` (commit a24f1f38), chạy trên LeRobot 0.6.2 (commit 72e1d9e5).
- **Đổi so với kế hoạch gốc:** `HuggingFaceVLA/smolvla_libero` (6721902) bị bỏ vì là model khác; các số chạy bằng nó
  (smoke 29/10, `demo_tasks`, `demo_spatial`) không được gộp vào kết quả.
- **Không chạy được bản đối chứng `lerobot/smolvla_base`** trên LIBERO: action 6 chiều, state 6 chiều, 3 camera, trong khi
  LIBERO cần action 7 chiều, state 8 chiều (lỗi `expected 7, got 6`). Thay bằng số baseline trong paper, ghi rõ là số của họ.
- Cờ eval bắt buộc: `--rename_map` (image→camera1, image2→camera2), `control_mode=relative`, `n_action_steps` đặt tường minh.

### 2.2 Kết quả chính (10 episode/task, 3 seed, 3 seed dùng chung 10 init state)

| Run | Suite | Cấu hình | Thành công |
|---|---|---|---|
| B_main | libero_10 | NAS=1, relative | **158/300 = 52,7%** (CI 47,0–58,2) |
| F_spatial | libero_spatial | NAS=1, relative | **237/300 = 79,0%** (CI 74,0–83,2) |

Task yếu nhất: libero_10 task 0 và task 8 (1/30 mỗi task), libero_spatial task 5 (0/30).

### 2.3 Ablation (seed 1000, so cặp episode cùng init state, McNemar)

| Phép thử | Kết quả | p |
|---|---|---|
| libero_10, NAS 1 vs 50 (50 ep/task) | 54,6% vs 47,8%; riêng task 8 NAS=50 thắng (19 vs 7) | 0,013 |
| libero_spatial, NAS 1 vs 50 | 78/100 vs 58/100 | 8,8e-5 |
| libero_spatial, relative vs absolute | 78/100 vs 2/100 | 2,7e-23 |

Hai lần chạy cùng cấu hình trùng 99/100 episode (gần tất định, không tất định từng bit).

### 2.4 Phân tích lỗi (chi tiết trong `failure_analysis.md`)
- Mọi thất bại đều là **hết giờ**. Thất bại do **cảnh ban đầu** quyết định nhiều hơn seed: 111/142 thất bại libero_10
  rơi vào 37 cảnh thất bại ở cả 3 seed.
- libero_10: kiểu phổ biến nhất là **xong bước con thứ nhất rồi không chuyển sang bước thứ hai**, thường đứng yên
  (91/142 có chuyển động cuối < 0,3).
- libero_spatial: **mang bát tới đĩa nhưng không đặt được** (14/63). Task 5 thất bại vì bát chồng trên ramekin.
- NAS=1 hay đứng yên khi thất bại, NAS=50 thì không. Hai "thành công" của absolute là do va chạm.

## 3. Cấu trúc repo

```
readme.md              # file này
DAY1.md                # ghi chú setup ngày 1 và danh sách bài đọc
study.md               # template phần Study (literature matrix, diffusion vs flow matching)
differences.md         # khác biệt so với bản gốc, kết quả, ablation, nhật ký lỗi
failure_analysis.md    # phân tích lỗi từ video
WORKLOG.md             # nhật ký các lệnh Claude đã chạy
run_steps              # các lệnh đã dùng (ghi tay)
setup_lerobot.sh, check_env.py, smoke_test.sh
run_queue.sh, run_D.sh, collect_results.py
probe_smolvla.py, train_more.sh
ffi26/ffi26/           # dữ liệu thật (trước 30/10 là symlink tới ~/ffi26; ~/ffi26 giờ là symlink ngược về đây)
  results/runs/        # B_main, D_nas1, D_nas50, F_spatial, F_spatial_nas50, F_spatial_abs (eval_info.json, video, log)
  results/smoke_*/     # các lần chạy thử
  logs/, ckpt_files/
  lerobot/             # clone upstream, có .git riêng, KHÔNG commit (.gitignore)
```

Lưu ý: env conda `lerobot` cài LeRobot dạng editable từ `/home/student/ffi26/lerobot`, nên **không được xoá symlink `~/ffi26`**.

## 4. Chạy lại

```bash
conda activate lerobot
cd ~/Lan_K68AI/temp
TAG=B_main NAS=1 CM=relative SEEDS="1000 1001 1002" bash run_queue.sh
SUITE=libero_spatial TAG=F_spatial NAS=1 CM=relative N_EP=10 SEEDS="1000 1001 1002" bash run_queue.sh
python collect_results.py --tag B_main
```

**Lưu ý:** các CSV tổng hợp hiện có đã cũ, không khớp eval_info.json (xem `differences.md` mục 6). Cần chạy lại
`collect_results.py` trước khi dùng, hoặc lấy số từ eval_info.json như hai file báo cáo.

## 5. Kế hoạch tiếp theo

### 5.1 Improve
Kế hoạch gốc (bản 29/10, mục 6): RL thu nhỏ (GRPO/RLOO hoặc PPO kiểu ReinFlow), reward theo subgoal từ predicate BDDL,
chỉ update action expert, trên 1–2 task yếu nhất. Script đang có là `train_more.sh` (fine-tune tiếp từ checkpoint,
**chưa chạy lần nào**):

1. `MODE=PREFLIGHT bash train_more.sh`: kiểm tra cờ của `lerobot-train` 0.6.2.
2. `MODE=SPEED bash train_more.sh`: đo giây/step và VRAM với batch 32 và 64, từ đó chọn số step.
3. `MODE=RUN`, rồi eval lại bằng `run_queue.sh` (`TAG=G_spatial_5k`, 3 seed), so cặp với F_spatial bằng McNemar.

Đề xuất của Claude dựa trên phân tích lỗi (**chưa được duyệt, bạn quyết định**):
- Mục tiêu hợp lý cho reward theo subgoal: libero_10 task 0 và 8, nơi robot thường xong bước con thứ nhất rồi dừng.
  Điều này khớp với giả thuyết gốc ("không biết làm tiếp sau subgoal đầu").
- Spatial task 5 (0/30) là mục tiêu rõ nhưng có thể là lỗi nhận thức cảnh (bát chồng trên ramekin), nên fine-tune thêm
  chưa chắc sửa được.
- Khi đánh giá, kiểm tra cả "quên" trên các task khác, và báo theo init state vì seed không đổi cảnh.
- `train_more.sh` mặc định ghi checkpoint vào `~/ffi26/train`, giờ nằm **trong repo**. Cần thêm `ffi26/ffi26/train/`
  vào `.gitignore` trước khi chạy, nếu không checkpoint sẽ lọt vào git.

### 5.2 Study
- Literature matrix ≥ 5 nguồn: 4 bài đã kiểm chứng (SimpleVLA-RL, ReinFlow, πRL, π\*0.6), cần kiểm chứng thêm ít nhất 1
  bài trong VLA-RL, RL4VLA, RIPT-VLA, ConRFT (`DAY1.md` mục D, E).
- Viết phần diffusion vs flow matching.

### 5.3 Việc còn treo trong phần Reproduce/Analyse
- Các trường recipe chỉ có trong `train_config.json` (chưa có bản lưu trên máy) đang đánh dấu [CHƯA KIỂM CHỨNG]
  trong `differences.md`.
- Một số nhóm lỗi trong `failure_analysis.md` chỉ dựa trên khung cuối; có thể kiểm chứng thêm bằng dòng thời gian.
- Viết 3 strengths và 3 limitations, mỗi ý gắn với số liệu hoặc video.

### 5.4 Đóng gói (thứ Hai 02/11)
Report 4–6 trang, `literature_matrix.xlsx`, slide 8–10 trang, `code/`, `setup/README.md` + `environment.yml`,
`results/` (bảng, biểu đồ, video thành công và thất bại), nén thành `[FFI26] Ngoc_Lan.zip`. Gửi từ tối 02/11.
