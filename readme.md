# FFI26: SmolVLA trên LIBERO — Reproduce, Analyse, Study, Improve

Model `lerobot/smolvla_libero` (commit `a24f1f38`), LeRobot 0.6.2, 1× RTX 5070 Ti 16 GB.
Cấu hình chính: libero_10, `n_action_steps = 10`, `control_mode = relative`. Cập nhật: 01/11/2026.

## 1. Kết quả chính

| | Thành công | Nguồn |
|---|---|---|
| Reproduce libero_10, NAS = 10, 3 seed × 50 init | **1092/1500 = 72,8%** (paper: 71% [CHƯA KIỂM CHỨNG]) | `docs/reproduce.md` §4 |
| Reproduce libero_10, NAS = 1, 3 seed × 10 init | 158/300 = 52,7% | như trên |
| Reproduce libero_spatial, NAS = 1, 3 seed × 10 init | 237/300 = 79,0% | như trên |
| **Improve FPO v4** (task 0 + 8, 3 seed × 50 init) | **140/300 = 46,7%** so với official 121/300 = 40,3% (p = 0,025); quên 8 task khác: 125 vs 127/160 (p = 0,83) | `$R/fpo_v4/pooled.md` |

Tài liệu: `docs/reproduce.md`, `docs/analyse.md`, `docs/study.md`, `docs/failure_overview.md`;
video: `~/ffi26/results/videos/README.md`; nhật ký lỗi và khác biệt: `differences.md`; danh sách thí nghiệm / checkpoint: `docs/inventory_storage.md`.

## 2. Đường dẫn

`~/ffi26` là symlink tới `ffi26/ffi26/` trong repo. Viết tắt dùng bên dưới:

| Viết tắt | Đường dẫn | Nội dung |
|---|---|---|
| `$I` | `~/ffi26/improve` | code Improve (`src/`, `scripts/`), `outputs/` (rollout, checkpoint, eval thô) |
| `$R` | `~/ffi26/results/round2` | config (`configs/`), bảng kết quả Improve (`fpo_v3/`, `fpo_v4/`, `summary.md`) |
| `$O` | `$I/outputs/fpo_v{3,4}` | `iter{k}/train/pretrained_model` (checkpoint θ_{k+1}), `selection.json`, `eval_s*/`, `timing.txt` |

Checkpoint cuối: `$I/outputs/fpo_v3/iter3/train/pretrained_model` (v3) và `.../fpo_v4/iter3/...` (v4), mỗi cái 1,1 GB, không commit.
Kết quả Reproduce: `~/ffi26/results/runs/<TAG>/`.

## 3. Cài đặt

```bash
bash setup_lerobot.sh          # conda env `lerobot` (Python 3.12, torch cu128), LeRobot editable, LIBERO
conda activate lerobot
python check_env.py && bash smoke_test.sh
```

Render headless: `export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl`. **Không xoá** `ffi26/ffi26/lerobot` hay `~/ffi26`
(env `lerobot` cài LeRobot editable từ đó).

## 4. Chạy lại Reproduce

```bash
conda activate lerobot && cd ~/Lan_K68AI/temp
TAG=D_nas10 NAS=10 CM=relative N_EP=50 SEEDS="1000 1001 1002" bash run_queue.sh   # cấu hình chính
TAG=B_main  NAS=1  CM=relative SEEDS="1000 1001 1002" bash run_queue.sh            # NAS=1 như paper
SUITE=libero_spatial TAG=F_spatial NAS=1 CM=relative N_EP=10 SEEDS="1000 1001 1002" bash run_queue.sh
python collect_results.py --tag <TAG>                                              # gộp eval_info.json
```

## 5. Chạy lại Improve: FPO v3, v4

FPO trên action expert (VLM đóng băng), train task 0 và 8 của libero_10, 4 vòng on-policy, 256 rollout/vòng.
v3 = v2 + demo replay chống quên + chuẩn hoá advantage theo task + batch cân bằng 2 task.
v4 = v3 + trọng số advantage theo segment, không trừ mean (khác duy nhất: `adv_weighting`, `adv_center` trong `$R/configs/fpo_v4.yaml`).

**Bước 1 — dữ liệu demo** (một lần, cho demo replay):

```bash
conda activate lerobot && cd ~/ffi26/improve && R=~/ffi26/results/round2
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
python scripts/download_demos.py                                               # lerobot/libero → data/
python scripts/phase2_demos_subset.py --config $R/configs/round1.yaml          # → data/r2/demos_libero10
python scripts/fpo_build_demo_cache.py --config $R/configs/fpo_v3.yaml         # → data/fpo_v3/demo_*.npz
```

**Bước 2 — vòng 0 và nhóm đối chứng từ v2.** v3 và v4 dùng lại rollout vòng 0 của v2 (`outputs/fpo_v2/iter0`, cùng θ0 và seed)
và eval official / v2 trong `outputs/fpo_v2/eval_s*`. Nếu chưa có:

```bash
bash scripts/fpo_v2_loop.sh
```

**Bước 3 — chạy v3, v4** (mỗi cái ≈ 2,6–3 giờ; nên chạy trong tmux; v4 so với v3 nên chạy v3 trước):

```bash
bash scripts/fpo_v3_loop.sh 2>&1 | tee outputs/fpo_v3/loop.log
bash scripts/fpo_v4_loop.sh 2>&1 | tee outputs/fpo_v4/loop.log
```

Mỗi script: train 4 vòng (`fpo_train_v3.py`) → thu rollout θ4 → chọn checkpoint theo luật sụp đổ đặt trước (`selection.json`)
→ eval task 0, 8 × 50 init ở seed 3000/4000/5000 và 8 task khác × 20 ở seed 1000 → bảng `$R/fpo_v{3,4}/pooled.md`.
Script bỏ qua bước đã có kết quả nên chạy lại được sau khi gián đoạn.

Kiểm tra phụ (không bắt buộc): `fpo_adv_check.py`, `fpo_v3_grad_steps.py` (v3), `fpo_v4_sanity.py` (v4) → `$R/fpo_v{3,4}/`.

**Demo nhanh một checkpoint** (không cần train, 1–3 phút):

```bash
cd ~/ffi26 && python demo/run_demo.py --methods official,fpo_v3,fpo_v4 --task 8 --init 0 --seed 3000
```

Xem thêm: `~/ffi26/demo/README.md` (demo, archive), `$R/README.md` và `$R/fpo_plan.md` (thiết kế FPO),
`$I/improve_2026-10-31.md` (nhật ký chi tiết).
