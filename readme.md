# FFI26: SmolVLA trên LIBERO — Reproduce, Analyse, Study, Improve

Hạn nộp: **08:00 thứ Ba 03/11/2026**. Cập nhật: 01/11/2026 18:40.

Model: `lerobot/smolvla_libero` (commit `a24f1f38`) trên LeRobot 0.6.2, 1× RTX 5070 Ti 16 GB.
Cấu hình chính từ 31/10: libero_10, `n_action_steps = 10`, `control_mode = relative`.

## 1. Tình trạng

| Phần | Trạng thái | Đọc ở đâu |
|---|---|---|
| Reproduce | xong | `docs/reproduce.md` (chi tiết, nhật ký lỗi: `differences.md`) |
| Analyse | xong | `docs/analyse.md` (nguyên nhân thất bại: `docs/failure_overview.md`, từng run: `docs/failures/`) |
| Study | xong | `docs/study.md` |
| Improve (RL) | xong: sàng lọc (A1, AWR, FPO v1) → FPO v2 → v3 → v4 | `ffi26/ffi26/results/round2/fpo_v4/pooled.md`, `results/round2/summary.md`, `docs/inventory_storage.md` |
| Report | bản đầy đủ, còn ô `[Cần bạn điền]` | `report/main.tex` → `main.pdf`; ghi chú: `report/MAIN_TEX_NOTES.md` |
| Demo | chạy được (đã test) | `ffi26/ffi26/demo/README.md` |

## 2. Kết quả chính

| | Thành công | Nguồn |
|---|---|---|
| libero_10, NAS = 1 (cách hiểu paper), 3 seed × 10 init | 158/300 = 52,7% | `docs/reproduce.md` §4 |
| **libero_10, NAS = 10, 3 seed × 50 init** | **1092/1500 = 72,8%** (paper: 71% [CHƯA KIỂM CHỨNG]) | như trên |
| libero_spatial, NAS = 1, 3 seed × 10 init | 237/300 = 79,0% | như trên |
| Improve, sàng lọc (seed 1000, 100 ep task 0+8) | A1/AWR không hơn official (44/100); FPO v1 θ3 57/100 (p = 0,01) | `results/round2/summary.md` |
| **Improve, FPO v4** (300 ep task 0+8, 3 seed; quên: 160 ep) | **140/300 = 46,7%** so với official 121/300 = 40,3% (p = 0,025); task 8: 80 vs 60/150 (p = 0,002); task 0: 60 vs 61; quên: 125 vs 127/160 (p = 0,83) | `results/round2/fpo_v4/pooled.md` |

## 3. Cấu trúc repo

```
readme.md                 file này
differences.md            khác biệt so với bản gốc, kết quả theo task, ablation, nhật ký lỗi (có nguồn từng số)
docs/
  reproduce.md            pipeline, setup, kết quả, khác biệt (bản ngắn)
  analyse.md              3+ điểm mạnh, 3+ giới hạn, quỹ đạo thành công / thất bại
  study.md                5 bài RL cho VLA; diffusion và flow matching
  failure_overview.md     so sánh NAS 1/10/50, bộ nguyên nhân F0–F6, thiết kế reward
  failures/               từng run: bảng, nhãn (labels/), khung hình (frames/), script (tools/)
  WORKLOG.md              nhật ký lệnh Claude đã chạy
  inventory_storage.md    danh sách mọi thí nghiệm, checkpoint, dung lượng, đề xuất archive / dọn
setup_lerobot.sh, check_env.py, smoke_test.sh     cài đặt và kiểm tra môi trường
run_queue.sh, run_D.sh, collect_results.py        chạy lerobot-eval theo hàng đợi, gộp kết quả
probe_smolvla.py          probe pipeline (shape, tham số, loss, độ trễ) → results/probe_report.json
run_steps                 các lệnh đã dùng (ghi tay)
report/                   main.tex / main.pdf, figures/ (sinh bằng report/scripts/make_figures.py, make_keyframes.py)
ffi26/ffi26/              dữ liệu thật (~/ffi26 là symlink tới đây)
  results/runs/           B_main, D_nas1, D_nas10, D_nas50, F_spatial, F_spatial_{nas10,nas50,abs}
  results/round2/         Improve: config, bảng kết quả, kế hoạch FPO
  results/demo_tasks/     chạy thử với checkpoint cũ HuggingFaceVLA/smolvla_libero (không dùng; đề xuất xoá, docs/inventory_storage.md §6)
  improve/                code Improve (src/, scripts/), nhật ký improve_2026-10-31.md; data/, checkpoint, npz không commit
  demo/                   chạy demo từng phương pháp (run_demo.py, methods.yaml), archive (archive_tool.py), README
  results/videos/v4_paired/  video official / v3 / v4 cùng episode, keyframes, sidebyside
  lerobot/                clone LeRobot (editable install của env `lerobot`), không commit
```

**Không xoá** `ffi26/ffi26/lerobot` hay symlink `~/ffi26`: env conda `lerobot` cài LeRobot editable từ đó.

## 4. Chạy lại

```bash
conda activate lerobot && cd ~/Lan_K68AI/temp
# Reproduce (libero_10, NAS=10, 3 seed, 50 episode/task)
TAG=D_nas10 NAS=10 CM=relative N_EP=50 SEEDS="1000 1001 1002" bash run_queue.sh
# NAS=1 như paper
TAG=B_main NAS=1 CM=relative SEEDS="1000 1001 1002" bash run_queue.sh
SUITE=libero_spatial TAG=F_spatial NAS=1 CM=relative N_EP=10 SEEDS="1000 1001 1002" bash run_queue.sh
# Improve: xem ffi26/ffi26/results/round2/README.md và fpo_plan.md §9
```

Các bảng trong tài liệu được tính thẳng từ `eval_info.json` của từng job. `episode_log.csv` của B_main và F_spatial
đã cũ (chỉ có seed 1000, `differences.md` §6): chạy lại `python collect_results.py --tag <TAG>` trước khi dùng.
