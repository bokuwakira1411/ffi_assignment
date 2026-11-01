# Video Improve (FPO)

Mọi video là **replay chính xác** episode của eval gốc (cùng init state, seed, batch), sinh bằng
`improve/scripts/phase3_eval_video.py`. Mỗi video: camera agent | camera cổ tay, 20 fps, header ghi nhánh, task, init, seed, bước, Φ, sự kiện reward.
Tên file: `{nhánh}_t{task}_init{k}_s{seed}_{success|fail}.mp4`.

| Thư mục | Nội dung | Mở trước |
|---|---|---|
| `3_v4_paired/` | official vs FPO v3 vs FPO v4 trên **cùng** 8 episode (task 0, 8; seed 3000/4000/5000): v4 thắng, v4 thua, cả ba cùng thất bại | `sidebyside/` (3 video cạnh nhau), `keyframes/*.png` |
| `1_v1_v2/` | official vs FPO v1 θ3 (`v1t3`) vs FPO v2 θ4 trên 9 cặp (seed, task, init) held-out (task 0, 8; seed 1000/2000) | `index.md` (bảng có link + ghi chú nhãn thất bại) |
| `2_v2_forget/` | v2 θ4 thất bại ở task 1 và 7 (eval quên, seed 1000): gắp nhầm vật của task 0 | `index.md` |

Trong mỗi thư mục:

- `official/`, `v1t3/`, `v2/`, `v3/`, `v4/`: video từng nhánh;
- `_meta/`: file phụ, không cần mở khi xem video:
  - `episodes_*.csv`: episode đã chọn, kết quả, nhãn;
  - `replay_check_*.csv`: replay so với bản gốc (`match`);
  - `raw/`: log từng bước (`.npz`, `.meta.json`);
  - `configs/`, `run.log`.

Sắp xếp lại ngày 01/11/2026 (trước đó: `videos/*.mp4`, `videos/v2_forget/`, `videos/v4_paired/`, mọi file để phẳng).
Script sinh video và config trong `_meta/configs/` vẫn ghi theo cấu trúc phẳng cũ. Nếu chạy lại, hãy ghi ra thư mục mới rồi sắp xếp như trên.
Riêng `report/scripts/make_keyframes.py` đã đọc theo cấu trúc mới.
Archive A1/A2 trong `~/ffi26_archives/` được tạo trước khi sắp xếp, nên giữ đường dẫn cũ.
