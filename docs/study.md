# FFI26 – Phần Study (template)

Trạng thái: **TEMPLATE, chưa điền**. Ô nào để `…` là chưa có nội dung.

## 0. Quy tắc điền

- Mỗi con số phải kèm **benchmark + điều kiện** (ví dụ: "LIBERO-Long, SFT few-shot, 3 seed") và **vị trí trong bài**
  (mục / bảng / hình / trang). Không có vị trí thì không ghi số.
- Chưa đọc tới, hoặc bài không nói, thì ghi `[bài không nêu]` hoặc `[chưa đọc]`, không đoán.
- Tên bài và mã arXiv trong bảng lấy từ `DAY1.md` mục D. Bài 1–4 đã kiểm chứng ngày 29/10; bài 5–8 phải tìm trên arXiv
  và điền cột "Kiểm chứng" trước khi đưa vào matrix.
- Cần **≥ 5 nguồn**: 4 bài đã kiểm chứng + ít nhất 1 bài từ nhóm 5–8.
- Bảng ở mục 1 sẽ được chép sang `literature_matrix.xlsx` khi đóng gói.

## 1. Literature matrix (bảng tổng hợp)

Cột lấy theo `DAY1.md` mục E.

| # | Bài | Năm / venue | arXiv | Kiểm chứng | Model nền | Dạng action (token / diffusion / flow) | Thuật toán RL | Reward signal | Môi trường & benchmark | Phần được update (full / LoRA / action head) | Giả định chính | Kết quả chính (số + điều kiện + vị trí) | Câu hỏi còn mở / hạn chế |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | SimpleVLA-RL | ICLR 2026 | 2509.09674 | đã (29/10) | … | … | … | … | … | … | … | … | … |
| 2 | ReinFlow | NeurIPS 2025 | 2505.22094 | đã (29/10) | … | … | … | … | … | … | … | … | … |
| 3 | πRL | … | 2510.25889 | đã (29/10) | … | … | … | … | … | … | … | … | … |
| 4 | π\*0.6 (RECAP) | … | 2511.14759 | đã (29/10) | … | … | … | … | … | … | … | … | … |
| 5 | VLA-RL | … | … | chưa | … | … | … | … | … | … | … | … | … |
| 6 | RL4VLA | … | … | chưa | … | … | … | … | … | … | … | … | … |
| 7 | RIPT-VLA | … | … | chưa | … | … | … | … | … | … | … | … | … |
| 8 | ConRFT | … | … | chưa | … | … | … | … | … | … | … | … | … |

## 2. Ghi chú từng bài

Chép khối dưới đây cho mỗi bài. Bảng mục 1 điền từ các khối này.

### [#] Tên bài

- **Trích dẫn đầy đủ:** tác giả, năm, venue, arXiv / link
- **Câu hỏi bài giải quyết (1–2 câu):** …
- **Model nền và dạng action:** … (vị trí: …)
- **Thuật toán:** … (on-policy / off-policy / offline; có critic không; cách tính log-likelihood nếu policy là diffusion/flow)
- **Reward signal:** … (sparse / dense / học được; lấy từ đâu)
- **Môi trường và benchmark:** … (sim / thật; có LIBERO không, suite nào)
- **Phần được update:** … (full / LoRA / action head; phần nào đóng băng)
- **Giả định chính:** … (ví dụ: cần reset môi trường, cần SFT trước, cần reward từ predicate)
- **Kết quả chính:**

  | Benchmark | Điều kiện | Trước | Sau | Vị trí trong bài |
  |---|---|---|---|---|
  | … | … | … | … | … |

- **Chi phí tính toán:** … (GPU, giờ, số rollout) `[bài không nêu]` nếu không có
- **Hạn chế / câu hỏi còn mở:** …
- **Liên quan tới phần Improve của mình:** … (dùng được gì cho SmolVLA trên LIBERO; khác gì về giả định)

## 3. Diffusion vs flow matching

Dàn ý theo kế hoạch gốc (`git show 3d7a55f:readme.md`, mục 5). Số liệu kiến trúc lấy từ lần chạy thật
(`ffi26/ffi26/results/probe_report.json`, `ffi26/ffi26/logs/probe_full.log`), không dùng số từ trí nhớ.

### 3.1 Kiến trúc SmolVLA (từ probe)

| Mục | Giá trị | Nguồn |
|---|---|---|
| VLM backbone | … | `probe_report.json` → … |
| Số tham số từng khối (VLM / action expert) | … | … |
| Input: ảnh (số camera, kích thước), state (số chiều), câu lệnh | … | … |
| Action: số chiều, chunk size | … | … |
| Số bước tích phân khi sinh action | … | … |
| Attention giữa prefix (ảnh, ngôn ngữ, state) và action tokens | … | … |

### 3.2 Diffusion policy
- Mục tiêu huấn luyện: … (dự đoán nhiễu / score)
- Sinh action: … (số bước khử nhiễu, sampler)
- Nguồn: …

### 3.3 Flow matching
- Mục tiêu huấn luyện: … (vector field trên đường nối nhiễu và action)
- Sinh action: … (tích phân ODE, số bước Euler trong SmolVLA: lấy từ 3.1)
- Nguồn: …

### 3.4 So sánh

| Tiêu chí | Diffusion | Flow matching | Nguồn |
|---|---|---|---|
| Mục tiêu huấn luyện | … | … | … |
| Số bước khi inference | … | … | … |
| Tính ngẫu nhiên khi sinh action | … | … | … |
| Tính log-likelihood của action | … | … | … |
| Hệ quả cho RL (policy gradient) | … | … | … |

### 3.5 Hệ quả cho RL
- Vì sao flow tất định gây khó cho policy gradient: …
- Cách ReinFlow xử lý: … (vị trí trong bài: …)
- Cách πRL xử lý (Flow-Noise, Flow-SDE): … (vị trí: …)
- Áp dụng được cho SmolVLA không, cần sửa phần nào: …

## 4. Nối với kết quả của mình

Mỗi dòng nối một hiện tượng đã đo (`failure_analysis.md`, `differences.md`) với phương pháp trong các bài.

| Hiện tượng đã đo | Số liệu (nguồn) | Bài nào nhắm tới hiện tượng này | Cách bài đó xử lý | Áp dụng cho mình được không |
|---|---|---|---|---|
| Xong bước con thứ nhất rồi không làm tiếp (libero_10) | `failure_analysis.md` mục 4, 7 | … | … | … |
| Mang bát tới đĩa nhưng không đặt được (Spatial) | 14/63, `failure_analysis.md` mục 5.3 | … | … | … |
| Thất bại do cảnh ban đầu, không do seed | 111/142, `failure_analysis.md` mục 2.2 | … | … | … |
| NAS=1 vs 50 đổi chiều theo suite | `differences.md` mục 3b | … | … | … |

## 5. Checklist

- [ ] Kiểm chứng ít nhất 1 bài trong nhóm 5–8 (tên chính xác, arXiv, venue)
- [ ] Điền đủ ghi chú mục 2 cho ≥ 5 bài
- [ ] Mọi ô "Kết quả chính" có benchmark, điều kiện và vị trí
- [ ] Điền mục 3.1 từ `probe_report.json`
- [ ] Viết xong mục 3.2–3.5
- [ ] Điền mục 4
- [ ] Chép bảng mục 1 sang `literature_matrix.xlsx`
