# FPO v2 / v3: danh sách cần người review kiểm tra lại

Viết ngày 01/11/2026. Mục đích: liệt kê **chính xác** những chỗ có thể làm FPO gần như không thay đổi checkpoint gốc
(hoặc làm phép so sánh sai). Mỗi mục ghi: chỗ cần xem (file:dòng), vì sao quan trọng, cách kiểm tra, và những gì
Claude **đã** kiểm tra (kèm bằng chứng) hoặc **chưa** kiểm tra. Đường dẫn tính từ `~/ffi26/improve` nếu không ghi khác.

Thứ tự = mức ưu tiên (mục 1 quan trọng nhất).

---

## Tóm tắt: vì sao "cải tiến" gần như không tác động lên checkpoint gốc

Số đo trực tiếp (script `scripts/fpo_weight_delta.py`, kết quả `results/round2/fpo_v3/weight_delta.csv`):

| checkpoint | số tensor thay đổi / 500 | thay đổi tương đối trung vị theo tensor | max | toàn bộ tham số train |
|---|---|---|---|---|
| v2 θ4 | 155 (145 của action expert + 10 projection) | 0,047 % | 0,081 % | 0,044 % |
| v3 θ4 | 155 | 0,089 % | 0,146 % | 0,083 % |

Thay đổi tương đối = ‖W − W0‖ / ‖W0‖ theo từng tensor, W0 = `lerobot/smolvla_libero`.

- Sau 4 vòng, trọng số chỉ dịch **dưới 0,15 %**. VLM và vision encoder bị đóng băng (đúng thiết kế).
- Policy vẫn thay đổi hành vi đo được: v3 vs official có 77/300 cặp bất đồng ở task 0+8; task 8 tăng có ý nghĩa.
  Nhưng tổng thể chỉ là một bước rất nhỏ quanh θ0.
- Nguyên nhân trực tiếp: mọi vòng dừng sớm (clip_frac 10 bước cuối > 0,2) sau 16–138 bước với lr 1e-6. Tổng cộng
  v2 có 84 bước, v3 có 402 bước. Để so sánh: SmolVLA gốc fine-tune với lr 1e-4, warmup 1000, decay 25 000 bước
  (`config.json` của checkpoint).
- Câu hỏi cho reviewer: trust region (mục 1) có bị đặt chặt quá mức do cách scale ratio hay không.

---

## 1. [Ưu tiên cao] Scale của ratio FPO: c = 1 / mean(l_old) có đúng với FPO không?

- **Chỗ xem:** `scripts/fpo_train_v3.py:168` (`c = 1.0 / mean_lold`), `:218` (`r = exp(clamp(c * dl, -5, 5))`),
  `:220` (PPO clip). v2 giống hệt (`scripts/fpo_train_v2.py`). Config: `fpo.ratio_scale: auto` trong
  `results/round2/configs/fpo_v{2,3}.yaml`.
- **Vì sao:** c = 24,61 (iter 0, `train_info.json`). Một mẫu bị clip khi |l_old − l_new| > log(1,2)/c = 0,0074, tức
  **18 %** của l_old trung bình (0,0406). Nếu c = 1 (ratio = exp(l_old − l_new)), ngưỡng là 0,18, gấp 24,6 lần.
  Vậy clip_frac > 0,2 (điều kiện early stop) đến sớm hơn rất nhiều, và việc dừng sớm này là thứ giới hạn mức thay
  đổi trọng số.
- **Cần reviewer kiểm tra:**
  1. Bài FPO (McAllister và cộng sự, 2025) định nghĩa ratio thế nào: có scale hay không; L_CFM là trung bình hay
     tổng theo chiều action và theo bước chunk. **Claude chưa đối chiếu với bài báo.**
  2. Loss per-sample của LeRobot là **trung bình** MSE theo (bước chunk × chiều action) (`modeling_smolvla.py`,
     `F.mse_loss(..., reduction="none")` dòng 774, rồi lấy mean). Nếu FPO dùng **tổng**, scale sẽ khác rất xa.
  3. "auto" được chọn ở v1/v2 với lý do gì (xem `docs/WORKLOG.md`, mục fpo_v2.yaml 31/10) và lý do đó còn đứng không.
- **Đã kiểm tra:** giá trị c, mean l_old, ngưỡng 0,0074 (in bởi `scripts/fpo_weight_delta.py`).

## 2. [Ưu tiên cao] Learning rate 1e-6 và early stop: có phải lý do chính khiến update quá nhỏ?

- **Chỗ xem:** `fpo.lr: 1.0e-6`, `warmup_samples: 640`, `grad_clip: 10.0`, `early_stop_clip_frac: 0.2`,
  `early_stop_window: 10`, `epochs: 3` trong `fpo_v3.yaml`; optimizer AdamW betas (0,9; 0,95) ở
  `scripts/fpo_train_v3.py:182`.
- **Vì sao:** với AdamW, mỗi bước thay đổi mỗi tham số khoảng ≤ lr, bất kể độ lớn gradient. 402 bước × 1e-6 cho
  thay đổi tuyệt đối tối đa khoảng 4e-4 mỗi tham số. Thay đổi đo được (bảng tóm tắt) khớp với bậc này.
  Grad norm bị clip về 10 ở 80 % (v2) và 91 % (v3) số bước (`results/round2/fpo_v3/grad_steps.md`).
- **Cần reviewer quyết định:** đây có phải thiết kế mong muốn (bảo thủ, tránh sụp) hay lr / tiêu chí dừng quá chặt.
  Mục 1 và mục 2 liên quan với nhau: lr cao hơn sẽ làm clip_frac tăng nhanh hơn và dừng sớm hơn, trừ khi sửa mục 1.
- **Đã kiểm tra:** không vòng nào chạm trần 3 epoch; cả 8 vòng (v2 và v3) dừng do early stop (`iter_summary.csv`).

## 3. [Đã kiểm tra] Dấu của objective PPO và hướng của ratio

- **Chỗ xem:** `scripts/fpo_train_v3.py:210–225` (tính `dl`, `r`, `surr`, loss).
- **Vì sao:** l = loss CFM (thấp = likelihood cao). Đúng phải là: dl = l_old − l_new, nên r > 1 khi l_new < l_old;
  loss_ppo = −mean(min(r·Â, clip(r)·Â)). Sai dấu ở bất kỳ chỗ nào sẽ làm policy học ngược advantage.
- **Bằng chứng gián tiếp (không thay thế kiểm tra code):** success khi thu thập tăng (v3: 97 → 127/256 sau vòng
  đầu); task 8 tăng có ý nghĩa ở eval. Nếu sai dấu thì khó có kết quả này, nhưng hãy đọc lại 15 dòng code đó.
- **Đã đọc lại (`fpo_train_v3.py:213–219`):** `dl = l_old − l_new`; `r = exp(clamp(c·dl, −5, 5))`;
  `surr = min(r·Â, clip(r, 0,8, 1,2)·Â)`; `loss = −sum(surr) / B`. **Dấu đúng.** Một điểm phụ: `clamp(−5, 5)` trên c·dl
  làm gradient bằng 0 khi |c·dl| > 5 (ratio > e⁵); hiếm xảy ra trước early stop, nhưng reviewer có thể xác nhận.

## 4. [Đã kiểm tra phần advantage thô; còn câu hỏi thiết kế] Trung bình âm của advantage và chuẩn hoá

- **Chỗ xem:** `src/reward_r2.py:173–217`; `scripts/fpo_collect.py:89–92`; `scripts/fpo_train_v2.py:133–134, 220`
  (Â = A / std, không trừ mean); `scripts/fpo_train_v3.py:101–102` (Â = (A − mean_t) / std_t, mean_t theo decision).
- **Đã kiểm tra (`results/round2/fpo_v3/adv_check.md`, script `scripts/fpo_adv_check.py`):** advantage thô **tính và
  gán đúng**. Mọi decision khớp đúng rollout và segment; trung bình trong mỗi (nhóm, stage) theo rollout ≈ 0. Trung bình
  âm (−0,028 đến −0,107) là do **trọng số theo decision**: rollout thất bại luôn có 52 decision (cả 1172 thất bại đều
  dài 520 bước), rollout thành công có 29–42; trung bình theo segment ≈ 0.
- **Số liệu về dấu (v3, sau chuẩn hoá per-task):** decision của rollout thành công có Â < 0: 0,0–3,6 % (thô 13,6–29,5 %);
  decision của rollout thất bại có Â > 0: 26–53 % (thô 15,3–27,9 %), task 0 là 44,7–52,6 %.
- **Câu hỏi thiết kế cho reviewer (chưa có câu trả lời):** (a) thất bại có nên nặng gấp 1,2–1,8 lần (52 / 29–42 decision) mỗi rollout chỉ vì
  dài hơn không; (b) mean_t có nên tính theo rollout hoặc segment thay vì theo decision không; (c) trong nhóm toàn thành
  công, segment có adv âm nhưng label "positive" (trainer dùng adv, không dùng label), điều đó có đúng ý không.
- **Không kết luận** về nguyên nhân task 0 giảm: v2 không trừ mean, v3 có trừ, và task 0 giảm ở cả hai.

## 5. [Trung bình] Hai cơ chế của "v3 chạy nhiều bước hơn" chưa được tách

- **Chỗ xem:** `results/round2/fpo_v3/grad_steps.md`, mục "Kết luận".
- **Vì sao:** v3 chạy 402 bước so với 84 của v2. Log khớp với hai giả thuyết: (a) trừ trung bình advantage; (b) số
  hạng demo chiếm một phần bước cập nhật. Không tách được nếu không có ablation (không được train thêm theo yêu cầu).
- **Cần reviewer:** chấp nhận ghi là "khớp với log, chưa xác nhận", hoặc cho phép 2 ablation ngắn (chỉ iter 0).

## 6. [Trung bình] Dữ liệu demo: mapping task và định dạng

- **Chỗ xem:** `scripts/fpo_build_demo_cache.py` (mapping task_index của dataset → env task id:
  {0: 4, 1: 6, 2: 9, 3: 2, 4: 7, 5: 0, 6: 8, 7: 1, 8: 3, 9: 5}); `data/fpo_v3/demo_cache_info.json`;
  `data/fpo_v3/demo_check.json`.
- **Đã kiểm tra:** token ngôn ngữ của demo task 0 / 8 trùng với token rollout; L_CFM của θ0 trên demo 0,070, trên
  rollout thành công 0,037. Các bản đối chứng đều cao hơn rõ: xoay ảnh 180° 2,08, đổi camera 1,57, sai task 0,95;
  100 % mẫu đối chứng cao hơn mẫu demo tương ứng.
- **Chưa kiểm tra:** mapping cho cả 10 task bằng cách so chuỗi lệnh (đã so token chỉ cho task 0 và 8). L_CFM demo
  (0,070) cao gần gấp đôi rollout (0,037). Claude cho rằng hợp lý (demo của người, rollout của chính policy), nhưng
  chưa chứng minh.

## 7. [Trung bình] Tính công bằng của phép so sánh eval

- **Đã kiểm tra:**
  - Code eval (`src/rollout.py`, `scripts/phase3_eval.py`) sửa lần cuối 31/10 23:34. Episode official / v2 tạo lúc
    01/11 01:49, của v3 lúc 10:41: **cùng code**.
  - Cả ba arm đều chạy expert fp32, n_action_steps 10, batch 8; episode k dùng init k và env seed = seed + k; RNG
    reset bằng `set_seed(seed)` mỗi (model, task) (`phase3_eval.py:3–5, 53`).
  - McNemar chính xác hai phía (`scripts/fpo_v3_pooled.py:32–37`) cho lại số của v2 (49/44, p = 0,68; quên 9/28,
    p = 0,0026).
- **Chưa kiểm tra:** phiên bản gói `lerobot` / `libero` / `mujoco` có đổi giữa 01:49 và 10:41 không (không có log
  `pip freeze` theo từng lần chạy).
- **Cần reviewer:** batch 8 trong vec env có làm RNG noise của policy phụ thuộc vào thứ tự episode trong batch không?
  Nếu có, "cùng seed" chưa chắc là cùng noise giữa các arm, nhưng điều đó chỉ làm nhiễu tăng, không làm lệch kết quả.

## 8. [Đã kiểm tra] Lưu checkpoint: trọng số fp32 có thật sự được lưu và nạp khi eval?

- **Đã kiểm tra:** checkpoint v3 θ4 có 155 tensor fp32 (official chỉ có 26), đúng 155 tensor khác official (145 của
  `lm_expert` + 10 projection); các tensor đóng băng giữ nguyên.
- **Đã kiểm tra thêm (CPU):** nạp v3 θ4 đúng như eval (`load_policy(path, 10, device="cpu", expert_fp32=True)`)
  rồi so với `model.safetensors`: **155/155 tensor đã train được nạp chính xác** (cùng dtype fp32, trùng từng bit),
  0 tensor bị làm tròn. Vậy update **không** bị mất khi eval. Code: `src/rollout.py:69–76` (cast `lm_expert` sang fp32
  rồi đọc lại giá trị từ file). Reviewer chỉ cần xác nhận là kiểm tra này đủ.
- **Kết luận:** loại bỏ giả thuyết "update bị xoá do làm tròn bf16". Update nhỏ là do huấn luyện (mục 1–2), không
  phải do nạp checkpoint.

## 9. [Thấp] Tách held-out / init đã train và độ mạnh thống kê

- **Chỗ xem:** `results/round2/fpo_v3/split_task_init.md`; init đã train lấy từ `collect.init_ids` (16 init/task).
- **Lưu ý:** 4 ô × 2 phép so sánh, p chưa hiệu chỉnh. Tổng task 0+8 (300 cặp) có 77 cặp bất đồng (v3 vs official);
  với số cặp này, chỉ phát hiện được chênh lệch khoảng ±8–10 điểm. Chênh lệch tổng +4,3 điểm (p = 0,17) nằm dưới
  ngưỡng đó. **Con số ±8–10 là ước lượng sơ bộ của Claude, chưa tính power chính thức.**

## 10. [Thấp] Failure labeler trên v3

- **Chỗ xem:** `outputs/failure_labels_v3/summary.md`, `scripts/failure_labels_v3.py`.
- **Đã kiểm tra:** nhãn official + v2 tính lại trùng 100 % với `outputs/failure_labels/labels.csv` (920 episode).
- **Chưa kiểm tra:** chưa xem video nào của v3. Nhãn G (đặt lệch 3–7 cm, task 4) và "1 ấm" (task 8) mới chỉ đến từ
  detector tự động.

---

## Việc reviewer nên làm trước (đề xuất)

1. (Mục 8 đã kiểm tra xong: trọng số fp32 được nạp chính xác khi eval.)
2. Mục 1: đối chiếu định nghĩa ratio (scale c, mean hay sum của L_CFM) với bài FPO. (Mục 3, dấu: đã đọc, đúng.)
3. Mục 4: advantage thô đã kiểm tra là đúng; còn 3 câu hỏi thiết kế (trọng số theo decision, cách tính mean_t, nhóm toàn thành công).
4. Sau đó mới quyết định có thay lr / ratio scale / tiêu chí dừng (cần train lại, ngoài phạm vi phân tích này).
