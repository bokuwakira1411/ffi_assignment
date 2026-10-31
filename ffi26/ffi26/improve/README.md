# FFI26 Phần 4 "Improve": SmolVLA trên LIBERO-10 với reward theo tiến độ và chính sách điều kiện theo advantage

Thư mục: `~/ffi26/improve/` (= `/home/student/Lan_K68AI/temp/ffi26/ffi26/improve`).
Nhật ký chi tiết từng bước (lệnh, thời gian, số liệu, sự cố): `improve_2026-10-31.md`.
Kết quả cuối: `outputs/results.md`.

Đây là **nghiên cứu quy mô nhỏ** (2 task mục tiêu, 50 episode eval mỗi task mỗi model, 2 vòng A.1).

**Trạng thái (31/10 10:20):** đang làm lại A.1; kết quả A.1 cũ đã xóa (xem nhật ký mục 10:20).

## 1. Môi trường

- Server lab, 1× RTX 5070 Ti 16 GB, 32 nhân CPU; conda env `lerobot` (Python 3.12, torch 2.11.0+cu128, lerobot 0.6.2 cài editable từ `~/ffi26/lerobot/src`, mujoco 3.8.1).
- Render: `MUJOCO_GL=egl PYOPENGL_PLATFORM=egl` (đã kiểm tra, không cần osmesa).
- Không sửa package lerobot đã cài; mọi thay đổi là subclass/wrapper trong `src/`.
- Không cài thêm package. (`pandas.to_markdown` cần `tabulate` không có → `phase1_analyze.md()` tự viết bảng.)
- Mọi job dài chạy trong tmux:
  ```bash
  tmux new -d -s <tên> 'source /opt/miniconda3/etc/profile.d/conda.sh && conda activate lerobot && cd ~/ffi26/improve && MUJOCO_GL=egl PYOPENGL_PLATFORM=egl <lệnh> 2>&1 | tee <log>'
  ```

Cấu hình chung: checkpoint `lerobot/smolvla_libero` (rev `a24f1f38…`), `n_action_steps=10`, `control_mode=relative`, libero_10, eval batch 8. Episode k của một task luôn dùng init state k và seed env `seed+k` (giống `lerobot-eval --seed=seed`).

## 2. Cấu trúc

```
src/
  reward.py          LiberoProbe (vị từ BDDL, Φ, grasp/contact, EE, gripper), RewardConfig, step_rewards, episode_return
  logged_env.py      LoggedLiberoEnv(LiberoEnv): log mỗi bước, chọn init state, khởi động từ sim state đã lưu
  rollout.py         load_policy, make_vec_env, run_batch (theo đúng lerobot_eval.rollout + hậu tố task, noise_scale, ghi frame)
  failure_labels.py  nhãn thất bại A–E, O; subtask completion
scripts/
  phase0_speedtest.sh       đo tốc độ lerobot-eval (bs 4, 8)
  probe_env.py              in tên thuộc tính LIBERO thật (goal_state, objects, ...)
  test_rollout.py           kiểm thử rollout + khôi phục sim state
  phase1_logging.py         rollout baseline có logging
  phase1_analyze.py         nhãn thất bại, kiểm tra reward
  download_demos.py         tải lerobot/libero vào data/
  phase2_collect.py         thu rollout theo nhóm, advantage, ghi dataset LeRobot
  phase2_demos_subset.py    demo của task mục tiêu, gắn "Advantage: positive"
  phase2_build_dataset.py   gộp demo + rollout, chép stats của checkpoint
  phase2_train.sh           lerobot-train (chỉ action expert + state proj)
  phase3_eval.py            eval nhiều model cùng init state/seed
  phase3_analyze.py         Wilson CI, McNemar, phân bố thất bại, biểu đồ
configs/  phase1.yaml, phase2_r1.yaml
outputs/  speedtest/, baseline/ (Phase 1), eval_baseline/ (baseline 50 ep × task 0, 8)
data/     lerobot_libero/ (bản tải), demos_targets/ (demo task mục tiêu, hậu tố positive)
```

## 3. Cách chạy

```bash
cd ~/ffi26/improve
# Phase 0
bash scripts/phase0_speedtest.sh "4 8"
# Phase 1: logging baseline (task 0, 8 × 20 episode) + phân tích
python scripts/phase1_logging.py --config configs/phase1.yaml
python scripts/phase1_analyze.py --logged outputs/baseline/logged --out outputs/baseline \
    --baseline_dir ~/ffi26/results/runs/D_nas10/lerobot_smolvla_libero
# Phase 2: rollout → dataset → fine-tune
python scripts/download_demos.py
python scripts/phase2_collect.py --config configs/phase2_r1.yaml
python scripts/phase2_demos_subset.py --config configs/phase2_r1.yaml
python scripts/phase2_build_dataset.py --config configs/phase2_r1.yaml
bash scripts/phase2_train.sh ~/ffi26/improve/data/a1_r1_train local/ffi26_a1_r1_train \
    ~/ffi26/improve/outputs/train_a1_r1 2000 32
# Phase 2 vòng 2 (thu bằng a1_r1, train lại từ model gốc trên demo + rollout v1 + v2)
python scripts/phase2_collect.py --config configs/phase2_r2.yaml
python scripts/phase2_build_dataset.py --config configs/phase2_r2.yaml
bash scripts/phase2_train.sh ~/ffi26/improve/data/a1_r2_train local/ffi26_a1_r2_train \
    ~/ffi26/improve/outputs/train_a1_r2 2000 32
# Phase 3 (model đã chạy xong sẽ được bỏ qua; --only <tên> để chạy riêng)
python scripts/phase3_eval.py --config configs/phase3_r1.yaml
python scripts/phase3_analyze.py --eval outputs/eval_r1 --out outputs \
    --ref_dir ~/ffi26/results/runs/D_nas10/lerobot_smolvla_libero
```

## 4. Phương pháp tóm tắt

**B.1 reward (Phase 1).** Φ(s) = (# vị từ goal BDDL đúng) / (# vị từ goal), đọc qua `LiberoEnv._env.env.parsed_problem["goal_state"]` và `_eval_predicate`. r_t = 1[success] + λ(Φ(s_{t+1}) − Φ(s_t)) − c, λ=0.5, c=0.001; tùy chọn grasp_bonus (một lần cho mỗi vật mục tiêu được nhấc), wrong_obj_cost, stall_cost (một lần mỗi đoạn dừng). Reward tính offline từ log nên có thể đổi định nghĩa mà không chạy lại.

**Nhãn thất bại.** A gắp hỏng (≥2 lệnh đóng kẹp gần vật mục tiêu chưa hoàn thành, vật không được nhấc); B kẹt sau subtask (một phần vị từ mới đúng); C nhầm vật (vật không thuộc goal bị gắp, hoặc bị kẹp chạm rồi dịch > 5 cm); D làm hỏng tiến độ (vị từ mới True → False); E dừng (EE di chuyển < 2 mm/bước trong ≥ 50 bước); O khác. Nhãn chính theo ưu tiên A > C > D > E > B > O.

**A.1 (Phase 2), kiểu RECAP rút gọn.** Không policy gradient, không critic. Mỗi nhóm = (task, trạng thái bắt đầu), G=8 rollout; G_i = return, A_i = (G_i − mean)/(std + 1e-6); nhóm có mọi G bằng nhau bị loại. Rollout có A_i > 0 → task "… Advantage: positive", còn lại "… Advantage: negative"; demo gốc → positive. Fine-tune bằng loss flow matching bình thường (chỉ action expert + state proj). Inference luôn thêm " Advantage: positive".

**Reset-to-failure.** Với task 8, rollout bắt đầu từ sim state lưu trong Phase 1 ngay khi vị từ mới đầu tiên đúng (ấm 1 vừa lên bếp), số bước còn lại = 520 − bước đó.

## 5. Hạn chế đã biết

- Quy mô nhỏ: 2 task, 50 episode eval/task/model, 1 vòng, 64 rollout. Khác biệt nhỏ hơn khoảng tin cậy được coi là chưa kết luận được.
- Trong LIBERO mọi thất bại đều hết 520 bước, nên G giữa các thất bại chỉ khác nhau theo số subtask xong (+ grasp bonus); ở task 0 phần lớn thất bại có Φ_T = 0.
- `_check_grasp` của robosuite bỏ sót một số lần gắp thật (ví dụ quai ấm moka) → "nhấc" dùng grasp **hoặc** contact; C cần robot chạm vật vì một số init state của task 0 có hộp rơi từ trên cao.
- Chưa có nhãn riêng cho "làm đổ vật" (thấy ở task 8: ấm thứ hai bị đổ khi gắp).
- Ở nhóm toàn thành công, rollout chậm hơn bị gắn "negative" chỉ do −c·T.
- Demo được giải mã và mã hóa lại (thêm một lần nén AV1) vì `dataset_tools.delete_episodes` lỗi trên bản hub `lerobot/libero` (thiếu cột `meta/episodes/chunk_index`).
- Kiểm tra hồi quy chỉ 3 task × 10 episode, so với số của lerobot-eval `D_nas10` seed 1000 trên cùng init state/seed (RNG nhiễu khác pipeline).
- Log loss của lerobot-train dừng sau ~bước 800 (không rõ nguyên nhân); checkpoint vẫn lưu đủ.
- Vòng 2 có 64% frame là rollout (> 50% mục tiêu) vì tích lũy rollout của 2 vòng.
- Init state dùng thu rollout (task 0: 1, 4, 6, 7; task 8: 6, 7, 9, 14) nằm trong tập eval 0–49.
- Dung lượng: `data/` và `outputs/train_*` (checkpoint) lớn — xem mục 6.

## 6. Dung lượng (trước khi commit)

Các thư mục lớn nằm trong cây làm việc của repo (không nên commit vào git thường): `data/lerobot_libero` (1.9 GB, có thể tải lại bằng `scripts/download_demos.py`), `data/*_train`, `data/rollouts_*`, `data/demos_targets`, `outputs/train_a1_r{1,2}` (checkpoint), `outputs/baseline/logged`, `outputs/phase2_r*/logs`, `outputs/eval_r1`. Xem `du -sh` trong nhật ký.
