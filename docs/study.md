# FFI26 – Study: RL cho VLA, diffusion và flow matching

Nguồn đọc ngày 31/10/2026 trên bản HTML arXiv (Diffusion Policy: bản PDF v5). Mỗi số có ghi vị trí trong bài.
Số liệu SmolVLA lấy từ `ffi26/ffi26/results/probe_report.json` (viết tắt **probe**). Venue của bài 1 lấy theo
`DAY1.md` mục D (đã xoá; bản cuối: `git show e852bb7:docs/DAY1.md`); trang arXiv không ghi venue.

## 1. Năm bài đề xuất phương pháp RL cho VLA

| # | Bài | arXiv | Model nền | Dạng action |
|---|---|---|---|---|
| 1 | SimpleVLA-RL (ICLR 2026) | 2509.09674 | OpenVLA-OFT 7B, sửa head thành token (256 token, chunk 8 trên LIBERO) (§4.1) | token |
| 2 | πRL | 2510.25889 | π0, π0.5, 3.3B (App. B.2) | flow |
| 3 | π\*0.6 / RECAP | 2511.14759 | π0.6: Gemma 3 4B + action expert 860M (§V-A) | flow (+ token FAST phụ) |
| 4 | RIPT-VLA | 2505.17016 | QueST ~20M (token), OpenVLA-OFT 7B (hồi quy liên tục) (§5.1) | token / liên tục |
| 5 | VLA-RL | 2505.18719 | OpenVLA-7B (§3) | token |

## 2. So sánh

### 2.1 Mục tiêu học, reward, cách huấn luyện

| # | Mục tiêu học (objective) | Reward | Cách huấn luyện | Phần được update |
|---|---|---|---|---|
| 1 | GRPO, không critic, bỏ KL; clip [0.8, 1.28]; bỏ các nhóm toàn thắng/toàn thua; nhiệt độ rollout 1.6 (§3.3–3.4, Eq. 10–11) | nhị phân 0/1 theo kết quả, gán cho mọi token của trajectory (§3.2, Eq. 9) | online, sim (LIBERO, RoboTwin); SFT trước rồi RL; 8×A800 (§4.1) | toàn bộ (§4.1) |
| 2 | PPO + GAE ở mức chunk; log-likelihood qua Flow-Noise (§4.1) hoặc Flow-SDE (§4.2) | nhị phân trên LIBERO (App. C.1); ManiSkill có thêm 0.1 khi gripper chạm vật (App. C.2) | online, sim, khung RLinf; SFT few-shot trước; 8×H100 (App. B.2) | chỉ action expert ~300M, VLM đóng băng (App. B.2) |
| 3 | Advantage conditioning: `−log π(a\|o,ℓ) − α·log π(a\|I,o,ℓ)` với `I = 1(A > ε_ℓ)` (§IV-B, Eq. 3); value phân phối 201 bin, VLM 670M (Eq. 1, §V-C) | −1 mỗi bước, 0 nếu thành công, −C_fail nếu thất bại (Eq. 5) | lặp offline trên robot thật: thu rollout + can thiệp của người → huấn luyện lại từ checkpoint pretrain (Alg. 1, §V-D) | toàn bộ VLA + value function (§V-D) |
| 4 | PPO clip + advantage leave-one-out (RLOO), K = 16 (QueST) / 8 (OFT); bỏ context toàn thắng/toàn thua (Eq. 5–6, §4.2) | nhị phân 0/1, không critic (§3.1) | online, sim (LIBERO, MetaWorld ML45); OFT cần thêm head dự đoán scale của Laplace để có log-prob (§4.3) | OFT: LoRA r=32 + action head (§5.1) |
| 5 | PPO + GAE, có critic, critic warmup (§3.3, §3.5); log-prob = tổng log-prob của 7 token action (Eq. 1) | sparse của môi trường + reward model quá trình (RPRM) học từ nhãn giả: tách bước con theo độ mở gripper và vận tốc ≈ 0 (§3.4) | online, sim (LIBERO 4 suite, mỗi suite train riêng); ~48 GPU-giờ | LoRA |

### 2.2 Giả định và kết quả chính

| # | Giả định chính | Kết quả chính (benchmark, điều kiện, vị trí) |
|---|---|---|
| 1 | SFT phải có tỉ lệ thành công > 0 (SFT 0 trajectory → RL vẫn 0%, §6.2); reset được môi trường | LIBERO trung bình 91.0 → 99.1, SFT full (Table 2); SFT 1 trajectory/task: 48.9 → 96.9, Long 17.3 → 91.7 (Table 5) |
| 2 | có SFT trước; sim đủ nhanh để chạy rollout online | LIBERO few-shot π0: 57.6 → 97.6 (Flow-Noise) / 96.1 (Flow-SDE); π0.5: 77.1 → 98.3 / 97.9 (Table 3) |
| 3 | quan sát Markov; can thiệp của người luôn tốt (gán I = True) (§IV-B); có người gán reward, can thiệp, reset (§VII) | throughput tăng hơn 2 lần, tỉ lệ thất bại giảm khoảng một nửa trên các task khó (abstract, Fig. 7); success > 90% trừ task giặt đồ đa dạng (Fig. 8) |
| 4 | SFT có success > 0; reset được đúng context ban đầu cho cả nhóm rollout (§4.1–4.2) | LIBERO, QueST: 82.7 → 93.6, Long 68.8 → 87.5; OFT: 96.7 → 97.5 (Table 1); 1 demo: 3.5% → 97.2% (§5.4) |
| 5 | nhãn giả từ gripper/vận tốc đúng với ranh giới bước con | LIBERO trung bình 76.5 → 81.0, Long 53.7 → 59.8 (Table 1) |

### 2.3 Câu hỏi còn mở

Tổng hợp của mình từ các bài trên:

1. **Học khi success ≈ 0.** Reward nhị phân không cho gradient nếu không rollout nào thành công (1 §6.2; 4 §4.1).
   Reward quá trình của bài 5 dựa trên heuristic, tác giả thừa nhận khó áp dụng cho thao tác khéo léo (§5).
2. **Sample efficiency và reset.** RL online cần nhiều rollout và phải reset về cùng cảnh ban đầu (2 §6; 4 §4.2).
   Trên robot thật, bài 3 phải nhờ người reset và gán reward (§VII).
3. **Chọn cách tính log-likelihood cho flow.** Flow-Noise và Flow-SDE hơn nhau tuỳ model
   (2 Table 1: π0 thì Flow-Noise hơn, π0.5 thì Flow-SDE hơn).
4. **Tổng quát hoá.** Kết quả OOD không ổn định: MetaWorld OOD không cải thiện (2 §5.2.2). Policy có thể học đường tắt
   khác demo, ví dụ đẩy vật thay vì gắp (1 §6.1, "pushcut").
5. **Trần hiệu năng.** Khi SFT đã mạnh thì RL tăng ít (4: OFT chỉ +0.8, Table 1).

## 3. Diffusion và flow matching trong VLA

### 3.1 Kiến trúc

Hai cách giống nhau ở chỗ: một mạng nhận **chunk action đang bị nhiễu** cùng bước thời gian/nhiễu, có điều kiện là
quan sát, rồi trả ra một đại lượng (nhiễu hoặc vận tốc) để khử nhiễu dần.

- **Diffusion Policy** (Chi et al., 2303.04137): điều kiện là quan sát `O_t`. Có hai backbone (§3.1): CNN 1D với FiLM
  theo `O_t` và bước `k`, hoặc transformer, trong đó token action nhiễu đi qua cross-attention với embedding quan sát.
- **VLA dùng flow** (π0 2410.24164; SmolVLA 2506.01844): VLM xử lý ảnh, câu lệnh và state (prefix). Một **action expert**
  là transformer nhỏ xử lý token action và attend vào đặc trưng của VLM. KV của prefix được cache, nên chỉ phần action
  phải tính lại ở mỗi bước tích phân (π0 §IV, App. A-D).

SmolVLA (`lerobot/smolvla_libero`):

| Mục | Giá trị | Nguồn |
|---|---|---|
| VLM | SmolVLM2-500M, dùng 16 lớp ngôn ngữ; vision encoder 12 lớp | probe `1_load_policy_and_config.config`, `2_parameters.modulelist_lengths` |
| Tham số | tổng 450.0M; vision 86.4M; text 204.6M; action expert 98.2M (16 lớp, rộng 0.75 lần VLM) | probe `2_parameters` (expert = tổng `lm_expert.*`) |
| Input | 2 camera 256×256, pad lên 512×512, 64 token/ảnh; 48 token ngôn ngữ; state 8 chiều pad lên 32 → prefix 177 token | probe `3_…`, `4_pipeline_trace` |
| Output | chunk 50 × 7 chiều (pad lên 32 khi train), mỗi bước chunk là 1 token | probe `4_pipeline_trace`, `6_inference_trace` |
| Attention | action expert dùng cross-attention vào VLM, xen kẽ self-attention cứ 2 lớp một lần | probe `config.attention_mode`, `self_attn_every_n_layers` |
| Inference | 10 bước Euler mỗi chunk, 96.5 ms/chunk (batch 1, RTX 5070 Ti) | probe `6_inference_trace` |

### 3.2 Diffusion (DDPM)

- **Huấn luyện:** lấy action sạch `x⁰`, chọn ngẫu nhiên bước `k`, cộng nhiễu `εᵏ`, rồi cho mạng **dự đoán nhiễu**:
  `L = MSE(εᵏ, ε_θ(O_t, x⁰ + εᵏ, k))` (Chi §2.2, Eq. 3, có điều kiện `O_t` theo §2.3).
- **Inference:** xuất phát từ nhiễu Gaussian `xᴷ`, lặp K bước
  `xᵏ⁻¹ = α(xᵏ − γ ε_θ(xᵏ, k) + N(0, σ²I))` (Eq. 1). Mỗi bước có cộng nhiễu, nên quá trình sinh là ngẫu nhiên.
  α, γ, σ là noise schedule theo k.
- **Số bước:** train với 100 bước; inference dùng DDIM với 10 bước, mất 0.1 s trên RTX 3080 (§3.4).

### 3.3 Flow matching

- **Đường đi:** nội suy thẳng giữa action và nhiễu. Theo quy ước trong code SmolVLA:
  `x_t = t·ε + (1−t)·A`, với t = 1 là nhiễu thuần (probe `5_loss…note`).
  Bài π0 và SmolVLA viết với τ ngược chiều: `A^τ = τA + (1−τ)ε`.
- **Huấn luyện:** mạng **dự đoán vận tốc**, loss `‖v_θ(x_t, o) − (ε − A)‖²` (probe; SmolVLA §3.1; Lipman et al. 2210.02747).
  t được lấy mẫu theo phân phối Beta, ưu tiên mức nhiễu cao (π0 App. A-B: `Beta((s−τ)/s; 1.5, 1)`, s = 0.999).
- **Inference:** lấy mẫu nhiễu một lần, rồi tích phân ODE bằng Euler từ t = 1 về 0, mỗi bước `x ← x − Δt·v_θ`.
  π0 và SmolVLA dùng 10 bước (π0 §IV, δ = 0.1; probe `num_steps = 10`). Ngoài nhiễu ban đầu, quá trình sinh là **tất định**.

### 3.4 So sánh

| Tiêu chí | Diffusion (DDPM) | Flow matching |
|---|---|---|
| Mạng dự đoán | nhiễu ε | vận tốc ε − A |
| Đường từ nhiễu đến action | theo noise schedule α, γ, σ | đường thẳng (nội suy tuyến tính) |
| Số bước inference trong các nguồn | 10 (DDIM; train 100) | 10 Euler |
| Ngẫu nhiên khi sinh | ở mọi bước (DDPM) | chỉ ở nhiễu ban đầu (ODE) |
| Log-likelihood của action | mỗi bước DDPM là chuyển Gaussian, nên log-prob của cả chuỗi khử nhiễu tính được | ODE tất định: log-likelihood chính xác phải tích phân divergence của v_θ dọc đường đi (như continuous normalizing flow, Lipman et al.), rất tốn; loss flow chỉ là cận dưới (π\*0.6 Eq. 4) |

### 3.5 Hệ quả cho RL

Policy gradient (PPO, GRPO) cần `log π(a|o)`. Flow policy tất định không cho đại lượng này một cách rẻ, nên có ba hướng:

1. **Chèn nhiễu học được vào mỗi bước Euler** (πRL Flow-Noise §4.1). Mỗi bước trở thành Gaussian, và
   log-prob là tổng log-Gaussian. Mạng nhiễu bị bỏ sau khi fine-tune.
2. **Đổi ODE thành SDE giữ nguyên phân phối biên** (πRL Flow-SDE §4.2, Eq. 6–9). Toàn bộ thành MDP hai tầng: tầng khử nhiễu
   lồng trong tầng môi trường.
3. **Không cần log-likelihood**: điều kiện policy theo chỉ báo advantage, rồi train bằng loss flow thông thường
   (π\*0.6 RECAP Eq. 3).

SmolVLA có cùng cấu trúc với π0 (action expert dạng flow, 10 bước Euler, chunk 50; §3.1). Vì vậy hướng 1 và 2 áp dụng
được trực tiếp: chỉ cần thêm nhiễu vào vòng Euler trong hàm sinh action và thêm critic.
