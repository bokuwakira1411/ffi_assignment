# FFI26 - Ghi chú Reproduce: khác biệt so với bản gốc và nhật ký lỗi

Cách dùng: điền dần trong lúc chạy. Ô "[ĐIỀN]" là việc của bạn. Phần đã điền lấy từ train_config.json của
lerobot/smolvla_libero (đọc 30/10/2026). Đối chiếu lại với file bạn tải về trước khi trích dẫn.

## 1. Mô hình và checkpoint

| Mục | Giá trị | Nguồn |
|---|---|---|
| Checkpoint chính | lerobot/smolvla_libero | model card |
| Commit SHA đã dùng | [ĐIỀN: lấy từ results/runs/<tag>/<policy>/checkpoint_revision.txt] | script tự ghi |
| Khởi tạo từ | lerobot/smolvla_base | model tree |
| Đối chứng | lerobot/smolvla_base (chưa fine-tune LIBERO) | |
| Lý do chọn | cùng nguồn với base, có train_config.json công bố, so sánh base vs fine-tuned sạch | |

## 2. Recipe huấn luyện công bố (train_config.json)

| Mục | Giá trị |
|---|---|
| Dataset | lerobot/libero, revision: null (không ghim, bản lúc train không xác định được) |
| Số step / batch | 25.000 / 32 |
| Optimizer | AdamW, lr 1e-4, warmup 1.000 step, cosine decay về 2,5e-6, seed 1000 |
| Tham số được cập nhật | toàn bộ (freeze_vision_encoder: false, train_expert_only: false, không LoRA) |
| Augmentation ảnh | tắt |
| Kiến trúc | SmolVLM2-500M-Video-Instruct, 16 layer VLM, expert width x0,75, self-attn mỗi 2 layer |
| Action | chunk_size 50, n_action_steps 50, 10 bước flow, 7 chiều |
| Env ghi trong config | control_mode relative, 2 camera (agentview, wrist), env.task libero_spatial (eval_freq 0, nên chỉ là cấu hình ghi lại) |

## 3. Bảng khác biệt giữa bản gốc và lần chạy của bạn

| Hạng mục | Bản gốc / docs | Lần chạy của bạn | Lý do / ghi chú |
|---|---|---|---|
| Phần cứng | [ĐIỀN: theo paper SmolVLA] | RTX 5070 Ti, driver [ĐIỀN] | |
| LeRobot | [ĐIỀN] | 0.6.2, commit [ĐIỀN từ versions.txt] | |
| PyTorch / Python | docs: >= 2.10, >= 3.12 | 2.11 cu128, 3.12 | |
| Suite eval | paper: 4 suite (Spatial, Object, Goal, Long) | libero_10 (LIBERO-Long) [+ suite khác nếu còn giờ] | |
| Episode / task | paper: 10 trial/task, chấm nhị phân; docs: 10 | [ĐIỀN] | |
| Số seed | docs: 3 (paper không nêu seed eval) | [ĐIỀN] | |
| Huấn luyện | paper (Sec 4.3): 100.000 step, batch 64, đa task trên cả 4 suite | checkpoint: 25.000 step, batch 32 (train_config) | KHÁC: không tái lập đúng điểm số paper là có thể |
| n_action_steps | paper (Sec 4.3): dự đoán lại sau mỗi action (ngầm hiểu = 1); chunk 50 | [ĐIỀN: NAS đã dùng] | train_config ghi 50; ablation D |
| control_mode | paper: action delta của end-effector (tương ứng relative) | [ĐIỀN] | kết quả phép thử E nếu có |
| Dataset khi chạy probe | lúc train: không ghim | [ĐIỀN: revision đã ghim] | |
| Số liệu paper, SmolVLA 0.45B (Table 2, p.10) | Spatial 90 / Object 96 / Goal 92 / Long 71 / TB 87,3 | [ĐIỀN: kết quả của bạn, theo suite] | paper chỉ báo trung bình theo suite, không theo từng task |

## 4. Nhật ký lỗi

| Thời điểm | Lỗi | Nguyên nhân | Cách sửa |
|---|---|---|---|
| 30/10 | `ModuleNotFoundError: huggingface_hub`, `lerobot-eval: command not found` | chạy trong env `base`, chưa `conda activate lerobot` | bật env `lerobot` |
| 30/10 | `Feature mismatch ... Missing camera1/camera2, Extra image/image2` | env LIBERO đặt tên camera `image`/`image2`, policy cần `camera1`/`camera2` | `--rename_map` (map lấy từ train_config.json của smolvla_libero); script đã thêm sẵn `RENAME_MAP` |
| 30/10 | `AssertionError: environment got invalid action dimension -- expected 7, got 6` khi chạy `lerobot/smolvla_base` trên LIBERO | env LIBERO cần action 7 chiều; base trả 6 chiều (nghi huấn luyện cho tay robot khác, [ĐIỀN: xác nhận bằng config.json của base]) | không chạy được base thẳng trên LIBERO. Đối chứng thay bằng số baseline trong Table 2 của paper, ghi rõ là số của họ, không phải đo lại |

## 5. Việc cần ghi khi báo cáo kết quả

- Smoke test cũ (0/1, 209 giây/episode) thuộc checkpoint HuggingFaceVLA/smolvla_libero, không phải checkpoint chính hiện tại.
- Giới hạn bước của episode và cách env xác định thành công: [ĐIỀN: đọc code env LIBERO trong LeRobot].
- Số episode thực tế mỗi job: lấy từ summary_per_job.csv, đừng dùng giá trị N_EP đã đặt.