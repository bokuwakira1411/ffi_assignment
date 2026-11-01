# Improve: code

Code RL cho SmolVLA trên LIBERO-10. Mọi thay đổi nằm trong `src/` và `scripts/` dưới dạng subclass hoặc wrapper; không sửa package LeRobot.
Kết quả và config: `results/round2/`. Cách chạy lại FPO v3/v4: `readme.md` ở gốc repo.

## `src/`

| File | Nội dung |
|---|---|
| `rollout.py` | rollout theo batch, cùng logic với `lerobot_eval.rollout` |
| `logged_env.py` | env LIBERO ghi log từng bước, chọn init state |
| `reward.py`, `reward_r2.py` | reward theo tiến độ (vị từ BDDL); phát hiện drop / slip / collision; advantage theo segment |
| `fpo_capture.py` | ghi lại từng quyết định (action chunk) khi thu rollout cho FPO |
| `fpo_adv.py` | chuẩn hoá và trọng số advantage của FPO v3, v4 |
| `adv_embed.py` | embedding nhãn advantage (A1) |
| `failure_labels.py` | gắn nhãn kiểu thất bại tự động |

## `scripts/`

| Nhóm | Script |
|---|---|
| FPO (phương pháp cuối) | `fpo_loop.sh`, `fpo_v{2,3,4}_loop.sh` (chạy toàn bộ v1–v4), `fpo_collect.py`, `fpo_train.py`, `fpo_train_v{2,3}.py`, `fpo_build_demo_cache.py`, `fpo_v{3,4}_pooled.py` (bảng kết quả) |
| Kiểm tra FPO | `fpo_verify.py`, `fpo_adv_check.py`, `fpo_v3_demo_check.py`, `fpo_v3_grad_steps.py`, `fpo_v4_sanity.py`, `fpo_weight_delta.py`, `task0_diag_*.py` |
| Sàng lọc A1 / AWR | `r2_collect.py`, `r2_build.py`, `train_r2.py`, `r2_report.py`, `r2_*.sh` |
| Bản đầu (phase 0–3) | `phase*.py`, `phase*.sh`, `download_demos.py` |
| Eval, nhãn, video | `phase3_eval.py` (eval nhiều model cùng init / seed), `failure_labels_*.py`, `phase3_eval_video.py` |

`outputs/`: kết quả eval từng episode (`ep*.meta.json`), tóm tắt mỗi vòng FPO, log train. Checkpoint và dữ liệu (`data/`) không có trong repo.
Nhật ký chi tiết: `improve_2026-10-31.md`.
