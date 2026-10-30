# FFI26 – Ngày 1 (Thứ Năm 29/10): Setup + bắt đầu đọc

## A. Điều chỉnh so với kế hoạch trước (đã kiểm tra docs LeRobot ngày 29/10/2026)

| Mục | Kế hoạch cũ | Đúng theo docs |
|---|---|---|
| Python | 3.10 | **≥ 3.12** |
| PyTorch | "cu128 trở lên" | **≥ 2.10**, wheel cu128, driver **≥ 570.86** |
| Số episode eval | 50 / task | Docs khuyên **10 episode / task**, trung bình **3 seed** |
| Dataset (chỉ cần khi chạy probe loss) | `HuggingFaceVLA/libero` | Docs khuyên `lerobot/libero` (**1,9 GB**, video MP4) thay vì 69,9 GB. Nhớ ghim revision (với `probe_smolvla.py` là `--dataset-revision`) |

Docs ở đây là nhánh `main`. Nếu lệnh trong bài khác phiên bản bạn cài, ghi lại commit hash (script đã làm sẵn).

## B. Việc cần làm tối nay (có giới hạn thời gian)

| # | Việc | Thời gian | Lệnh |
|---|---|---|---|
| 1 | Setup môi trường | 30–60 phút (phần lớn là tải) | `bash setup_lerobot.sh` |
| 2 | Kiểm tra môi trường | 2 phút (script gọi sẵn) | `python check_env.py` |
| 3 | Smoke test (30/10: đo tốc độ song song, 5 episode) | 10–20 phút | `conda activate lerobot && N_EP=5 EVAL_BS=5 NAS=50 CM=relative bash smoke_test.sh` |
| 4 | Đọc 3–4 bài đầu (mục D) | 2–3 giờ | – |

**Quy tắc:** nếu bước 1–3 kẹt quá 3 giờ thì dừng, chuyển phần eval sang Kaggle và báo mình lỗi cụ thể.

### Kết quả mong đợi
- `check_env.py`: các dòng `[OK]`. Dòng `gpu_kernels_fp32_bf16` mà FAIL nghĩa là PyTorch không đúng bản cho 5070 Ti.
- `smoke_test.sh`: có file video trong thư mục `results/smoke_*` và dòng success rate trong `eval.log`. **Một episode không phải kết quả để báo cáo**, chỉ chứng minh pipeline chạy được.

### Lỗi thường gặp
- **LIBERO hỏi câu Y/N khi import lần đầu** (có thể treo hoặc báo EOFError): chạy tay `python -c "import libero.libero"` trong terminal thường, trả lời N, rồi chạy lại.
- **Render lỗi EGL**: thử `export MUJOCO_GL=osmesa` (chậm hơn nhiều). Không có root thì nhờ admin cài `libegl1`.
- **Lỗi build khi cài** (PyAV, cmake): docs LeRobot liệt kê gói hệ thống cần có (`cmake`, `build-essential`, các `lib*-dev` của ffmpeg). Không có sudo thì báo admin.
- **Thiếu VRAM/chậm khi eval**: giữ `--eval.batch_size=1`.

## C. Ba điều phải quyết định TRƯỚC khi chạy eval chính (ngày 2)

**Cập nhật 30/10:** checkpoint chính đổi sang `lerobot/smolvla_libero` (cùng nguồn với `smolvla_base`, fine-tune từ base,
có `train_config.json` công bố). Recipe đầy đủ nằm trong `notes/differences.md`. `HuggingFaceVLA/smolvla_libero` không còn
là checkpoint chính; chỉ chạy lại nếu docs LIBERO của LeRobot ghi benchmark cho nó mà không ghi cho bản `lerobot/`.

1. **`--env.control_mode`**: train_config ghi `relative` nhưng `eval_freq` là 0 nên đó chỉ là cấu hình ghi lại, chưa chắc là bằng chứng. Mình chưa kiểm tra dataset lưu action kiểu nào. Chạy phép thử 5 episode cho mỗi chế độ (phần A/E) và ghi cả hai kết quả.
2. **`--policy.n_action_steps`**: train_config ghi 50 (open-loop 50 bước). Bản HuggingFaceVLA cũ dùng 1. Đặt tường minh bằng `NAS=` và ghi vào report. Ablation 1 vs 50 là phần D. Issue #2354 của LeRobot có nói về giá trị 1; đọc rồi mới trích.
3. **Dataset lúc train**: `lerobot/libero` với `revision: null`, không xác định được bản dùng lúc train. Khi chạy probe loss, ghim revision bằng `--dataset-revision` và ghi lại.

Cả ba đều thuộc phần "differences from the original work".

## D. Danh sách đọc (RL cho VLA)

### Đã kiểm chứng ngày 29/10 (tìm thấy trang arXiv hoặc venue): bài 1-4. Bài 5-8 bên dưới CHƯA kiểm chứng.

| # | Bài | arXiv | Ghi chú nhanh để đọc có mục tiêu |
|---|---|---|---|
| 1 | SimpleVLA-RL: Scaling VLA Training via RL | 2509.09674 (ICLR 2026) | Xây trên veRL, áp dụng cho OpenVLA-OFT (VLA sinh action dạng token). Reward theo outcome. Có mục "Failure Modes" (6.2) rất hợp phần Analyse |
| 2 | ReinFlow: Fine-tuning Flow Matching Policy with Online RL | 2505.22094 (NeurIPS 2025) | **Không phải VLA** mà là flow policy cho điều khiển liên tục. Chèn nhiễu học được vào đường đi tất định để tính log-likelihood chính xác. Đọc để hiểu ý tưởng cho phần Improve |
| 3 | πRL: Online RL Fine-tuning for Flow-based VLAs | 2510.25889 | Flow-Noise và Flow-SDE giải bài toán log-likelihood của flow cho π0/π0.5, chạy trong khung RLinf, benchmark có LIBERO. **Gần phần Improve của bạn nhất** |
| 4 | π\*0.6: a VLA That Learns From Experience (RECAP) | 2511.14759 | Offline RL pretrain, advantage conditioning, học từ rollout thật và can thiệp của người. Khác hẳn 3 bài trên (không phải online sim RL), nên có giá trị so sánh |

### Chưa kiểm chứng (tìm trên arXiv trước khi đưa vào matrix)

| # | Bài | Gợi ý tìm |
|---|---|---|
| 5 | VLA-RL | "VLA-RL scalable robotic reinforcement learning" |
| 6 | RL4VLA (Liu et al.) | "What can RL bring to VLA generalization" |
| 7 | RIPT-VLA | "RIPT-VLA interactive post-training" |
| 8 | ConRFT | "ConRFT reinforced fine-tuning VLA consistency policy" |

Chỉ cần ≥ 5 nguồn: nếu hết thời gian, 4 bài trên cộng 1 bài từ bảng thứ hai là đủ.

### Có trong menu docs LeRobot (mình chưa đọc nội dung)
"Train RL in Simulation" (HIL-SERL) và các trang Reward Models (SARM, ROBOMETER, TOPReward). Đáng xem nhanh cho phần reward signal ở mục Improve.

## E. Cột của literature matrix

`Bài | Năm/venue | Model nền | Dạng action (token / diffusion / flow) | Thuật toán RL | Reward signal | Môi trường & benchmark | Phần được update (full / LoRA / action head) | Giả định chính | Kết quả chính (số + điều kiện) | Câu hỏi còn mở / hạn chế`

**Khi đọc mỗi bài, chỉ cần tìm đủ các ô trên.** Mỗi con số đưa vào matrix phải ghi rõ benchmark và điều kiện (ví dụ SFT few-shot hay full-data).

## F. Việc ghi lại cho report (làm ngay khi chạy)
- `results/versions.txt`, `pip_freeze.txt`, `environment.yml`, `env_report.json` do script tạo ra.
- `results/smoke_*/checkpoint_revision.txt` (commit hash của checkpoint).
- Mọi lỗi gặp phải và cách sửa (cho mục "setup and differences").
- Smoke test ngày 29/10: libero_10 task 0, **0/1 thành công, 209 giây/episode, chạy tuần tự**. Đó là checkpoint `HuggingFaceVLA/smolvla_libero` (đã bỏ), không phải checkpoint chính hiện tại. Chưa có ý nghĩa thống kê; cần chạy lại với `lerobot/smolvla_libero`.