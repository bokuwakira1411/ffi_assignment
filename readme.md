# FFI26: SmolVLA trên LIBERO — Reproduce, Analyse, Study, Improve

Checkpoint `lerobot/smolvla_libero` trên LeRobot, 1 GPU 16 GB. Cấu hình chính: libero_10, `n_action_steps = 10`.

## Kết quả

| | Tỉ lệ thành công |
|---|---|
| Reproduce libero_10 | 72,8% (paper: 71%, chưa kiểm chứng) |
| Improve FPO v4, task 0 + 8 | 46,7% so với checkpoint gốc 40,3%; 8 task còn lại không giảm đáng kể |

Chi tiết: `docs/reproduce.md`, `docs/analyse.md`, `docs/study.md`, `ffi26/ffi26/results/round2/readme_ROUND2.md`.

## Cấu trúc

```
docs/                     Reproduce / Analyse / Study, phân tích thất bại
differences.md            khác biệt so với paper, nhật ký lỗi
setup_lerobot.sh, check_env.py, smoke_test.sh     cài đặt
run_queue.sh, collect_results.py                  eval Reproduce
ffi26/ffi26/
  improve/                code Improve (src/, scripts/), output eval
  results/runs/           kết quả Reproduce
  results/round2/         config + bảng kết quả Improve
  results/videos/         video so sánh các phương pháp
  demo/                   chạy demo một checkpoint
```

## Cài đặt

Các script dùng đường dẫn `~/ffi26`, nên cần tạo symlink trước:

```bash
git clone <repo> && cd <repo>
ln -s "$PWD/ffi26/ffi26" ~/ffi26
LEROBOT_REF=72e1d9e5e76ee719bacfc78bc34d817bee5fb521 bash setup_lerobot.sh   # env conda `lerobot`, LeRobot, LIBERO
conda activate lerobot && python check_env.py && bash smoke_test.sh
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
```

Các script `improve/scripts/*.sh` gọi `source /opt/miniconda3/etc/profile.d/conda.sh`. Nếu conda cài ở chỗ khác thì sửa dòng này.

## Chạy lại Reproduce

```bash
TAG=D_nas10 NAS=10 CM=relative N_EP=50 SEEDS="1000 1001 1002" bash run_queue.sh
python collect_results.py --tag D_nas10
```

Kết quả nằm ở `~/ffi26/results/runs/D_nas10/`.

## Chạy lại Improve (FPO v3, v4)

```bash
cd ~/ffi26/improve && R=~/ffi26/results/round2/configs
# 1. dữ liệu demo (một lần)
python scripts/download_demos.py
python scripts/phase2_demos_subset.py --config $R/round1.yaml
python scripts/fpo_build_demo_cache.py --config $R/fpo_v3.yaml
# 2. v2 (v3 và v4 dùng lại rollout vòng 0 và eval official của v2)
bash scripts/fpo_v2_loop.sh
# 3. v3, v4 (mỗi cái khoảng 3 giờ)
bash scripts/fpo_v3_loop.sh
bash scripts/fpo_v4_loop.sh
```

Mỗi loop: train 4 vòng → chọn checkpoint → eval → bảng `results/round2/fpo_v{3,4}/pooled.md`.
Checkpoint lưu ở `improve/outputs/fpo_v{3,4}/iter3/train/pretrained_model` (không có trong repo).
Khi chạy lại, script bỏ qua các bước đã xong.

Demo nhanh một checkpoint: `ffi26/ffi26/demo/readme_DEMO.md`.
