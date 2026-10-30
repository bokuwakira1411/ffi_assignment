# Kế hoạch FFI26 (bản chỉnh lại)

Hạn nộp là **08:00 sáng thứ Ba 03/11**, hôm nay là thứ Năm 29/10. Mình khuyên gửi từ tối 02/11 để có dư thời gian.

## 1. Quyết định chính

| Mục | Chọn |
|---|---|
| Model | SmolVLA (flow matching action expert, code trong LeRobot) |
| Benchmark | LIBERO-10 (long-horizon) |
| Checkpoint chính | `HuggingFaceVLA/smolvla_libero` (đã fine-tune trên LIBERO) |
| Checkpoint đối chứng | `lerobot/smolvla_base` (pretrain, chưa fine-tune), chạy ngắn 1–2 giờ |
| Train lại | **Không**, trừ khi checkpoint hỏng |

Lý do chọn bản fine-tuned sẽ viết **sau khi có kết quả** của bản base (xem Giai đoạn B). Chuỗi nguồn gốc để vẽ vào report: SmolLM2 → SmolVLM2 → smolvla_base → smolvla_libero.

## 2. Phân bổ tài nguyên

| Tài nguyên | Dùng cho |
|---|---|
| **5070 Ti (lab)** | Toàn bộ eval chính, probe pipeline, RL thu nhỏ |
| **Kaggle (1 tài khoản)** | Chạy eval phụ (nhiều seed, thu video) khi máy lab bận |
| **Máy cá nhân** | Đọc paper, viết report và slide |

Không dùng nhiều tài khoản Kaggle, vì điều khoản cấm. Kaggle T4 không có bf16 tốt, mỗi session tối đa 12 giờ và phải lưu checkpoint ra Dataset.

## 3. Setup (5070 Ti, không Docker)

1. Cài `micromamba` hoặc `miniconda` trong thư mục home, tạo env Python 3.10.
2. Cài **PyTorch bản cu128 trở lên**, vì 5070 Ti là Blackwell (sm_120), bản cũ sẽ lỗi "no kernel image".
3. Cài LeRobot kèm extras `smolvla` và `libero`. Nếu extras lỗi, cài LIBERO từ repo gốc.
4. Đặt `MUJOCO_GL=egl` để render headless. Thiếu EGL và không có root thì dùng `osmesa` (chậm hơn nhiều).
5. Chạy trong `tmux` hoặc `nohup`, log ra file.
6. Kiểm tra: `nvidia-smi`, `torch.cuda.is_available()`, chạy 1 episode LIBERO ra video.
7. Lưu `pip freeze`, `environment.yml`, commit hash của LeRobot, và commit hash của từng checkpoint.

Nếu kẹt setup quá 3 giờ, chuyển sang chạy trên Kaggle.

## 4. Lịch từng ngày

### Thứ Năm 29/10 (tối nay): setup và bắt đầu đọc
- Hoàn thành setup, chạy thử 1 episode LIBERO.
- Bắt đầu đọc 8 nguồn RL cho VLA (xem mục 5).

### Thứ Sáu 30/10: Reproduce

**Giai đoạn A, chạy bản base (1–2 giờ):**
- Chạy `lerobot/smolvla_base` trên 2–3 task LIBERO-10, vài episode mỗi task.
- Ghi lại mọi lỗi (shape, tên camera, chiều state/action, normalization).
- Đánh dấu kết quả thuộc loại nào: không tương thích pipeline, hay chạy được nhưng làm sai.

**Giai đoạn B, chạy bản fine-tuned (phần chính):**
- Chạy `HuggingFaceVLA/smolvla_libero` trên LIBERO-10, **cùng seed và cùng init state** với bản base.
- Cấu hình tham khảo: mỗi task 10–50 episode, 1–3 seed, lưu success rate theo task và video. Kiểm tra cờ `n_action_steps` trong docs và issue #2354 của LeRobot rồi ghi rõ giá trị đã dùng.
- Viết lý do A, B, C chọn bản fine-tuned dựa trên kết quả giai đoạn A.

**Pipeline probe (viết một script):**
- In shape input (ảnh từ 2 camera, state, câu lệnh), số tham số từng khối, các bước flow matching khi sinh action, và giá trị loss trên vài batch mà không cập nhật trọng số.

### Thứ Bảy 31/10: Analyse và Study
- Xem video, phân loại lỗi: grasp lỗi, sai thứ tự subgoal, kẹt lặp, sai ngữ nghĩa, rơi vật.
- Lập bảng success rate theo task. Task thấp nhất là mục tiêu cho phần RL.
- Viết 3 strengths và 3 limitations, mỗi ý gắn với số liệu hoặc video cụ thể.
- Viết phần diffusion vs flow matching.

### Chủ Nhật 01/11: Improve
- Triển khai bản RL thu nhỏ trên 1–2 task yếu nhất (xem mục 6).
- Nếu RL không kịp có kết quả thì vẫn nộp như evaluation plan đầy đủ, kèm log chạy thử.

### Thứ Hai 02/11: đóng gói
- Sáng: hoàn thiện report và literature matrix.
- Chiều: làm slide, dọn code, viết README.
- Tối: đóng zip, chạy thử theo README, gửi email.

## 5. Phần Study

**Literature matrix** (ít nhất 5 nguồn, nên làm 7–8). Các nguồn mình đề xuất, bạn cần tự kiểm tra tiêu đề và liên kết trên arXiv:
- SimpleVLA-RL, VLA-RL, RL4VLA, RIPT-VLA, ReinFlow, πRL, π*0.6 (RECAP), ConRFT.

**Các cột:** learning objective, thuật toán, reward signal, môi trường, phần model được update, giả định, câu hỏi còn mở.

**Diffusion vs flow matching:**
- Kiến trúc: VLM backbone cộng action expert, attention giữa prefix (ảnh, ngôn ngữ, state) và action tokens.
- Diffusion: dự đoán nhiễu hoặc score, nhiều bước khử nhiễu.
- Flow matching: học vector field theo đường thẳng nối nhiễu và action, inference khoảng 10 bước Euler.
- Hệ quả cho RL: flow là tất định nên cần chèn nhiễu để có log-likelihood, đây là ý tưởng của ReinFlow và πRL.

## 6. Phần Improve

| Thành phần | Đề xuất |
|---|---|
| Limitation mục tiêu | Chỉ học bằng imitation nên không biết phục hồi sau lỗi ở subgoal đầu |
| Hypothesis | RL với reward theo subgoal giúp học hành vi phục hồi mà demo không có |
| Thuật toán | GRPO hoặc RLOO (không cần critic), hoặc PPO kiểu ReinFlow |
| Reward | Sparse (cả task) cộng reward từng subgoal từ predicate BDDL của LIBERO |
| Môi trường | LIBERO-10, 1–2 task yếu nhất |
| Phần update | Chỉ action expert (có thể thêm LoRA), đóng băng VLM |
| Đánh giá | So trước và sau RL trên task đó và trên task khác (kiểm tra quên), ít nhất 3 seed, kèm khoảng tin cậy |

Nút thắt thường là tốc độ rollout mô phỏng, nên giới hạn quy mô ngay từ đầu.

## 7. Cấu trúc file nộp (`[FFI26] Ngoc_Lan.zip`)

```
report.pdf              # 4–6 trang
literature_matrix.xlsx
slides.pptx             # 8–10 slide
code/                   # eval, probe pipeline, phân tích lỗi, RL
setup/README.md + environment.yml
results/                # bảng, biểu đồ, video thành công và thất bại
```

## 8. Điểm cần xác minh trước khi viết vào report

- Checkpoint `smolvla_libero` được train trên suite nào, bao nhiêu step (xem `train_config.json` trong tab Files).
- Số chiều action/state, chunk size, số bước Euler, số token ảnh: lấy từ chạy thật, không dùng số từ trí nhớ.
- Cờ lệnh của `lerobot-eval` và `lerobot-train` (thay đổi nhanh giữa các phiên bản).

## 9. Rủi ro

| Rủi ro | Cách xử lý |
|---|---|
| Setup LIBERO hoặc EGL lỗi | Kẹt quá 3 giờ thì chuyển sang Kaggle |
| Checkpoint fine-tuned không chạy | Fine-tune rút gọn trên LIBERO-10 (LoRA, 20–30k step), ghi rõ khác biệt |
| Bản base chạy bất ngờ tốt | Báo cáo đúng kết quả, điều chỉnh lý do chọn bản fine-tuned |
| RL không cải thiện | Vẫn nộp, trình bày trung thực kèm phân tích nguyên nhân |
| Thiếu thời gian | Cắt RL còn evaluation plan, giữ nguyên baseline và phân tích lỗi |

Bạn muốn bắt đầu bằng script setup cho 5070 Ti hay script eval hai checkpoint?