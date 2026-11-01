# Demo và khôi phục dữ liệu (FFI26)

Viết ngày 01/11/2026. `~/ffi26` là symlink tới `~/Lan_K68AI/temp/ffi26/ffi26`.

## 1. Xem kết quả (không cần GPU)

| Muốn xem | Mở |
|---|---|
| Báo cáo | `~/Lan_K68AI/temp/report/main.pdf` (nguồn: `main.tex`, hình: `report/figures/`) |
| Bảng số cuối (v4 so với official, v3) | `results/round2/fpo_v4/pooled.md` |
| Số liệu từng episode | `results/round2/fpo_v4/episodes_all.csv`; `improve/outputs/*/eval_s*/*/task*/ep*.meta.json` |
| Video so sánh official / v3 / v4 trên cùng episode | `results/videos/v4_paired/sidebyside/*.mp4` (3 video cạnh nhau), từng arm: `results/videos/v4_paired/*.mp4` |
| Khung hình tóm tắt | `results/videos/v4_paired/keyframes/*.png`, `report/figures/fig_keyframes.pdf` |
| Video official / v1 / v2, quên của v2 | `results/videos/*.mp4`, `results/videos/v2_forget/` |
| Thất bại của checkpoint gốc (Analyse) | `~/Lan_K68AI/temp/docs/failures/frames/`, `docs/failures/*.md` |

## 2. Chạy demo (GPU, khoảng 1–3 phút mỗi phương pháp)

```bash
conda activate lerobot && cd ~/ffi26
python demo/run_demo.py --list                       # 11 phương pháp đã eval, checkpoint, seed có eval gốc
python demo/run_demo.py --methods official,fpo_v4 --task 0 --init 4 --seed 5000
python demo/run_demo.py --methods official,fpo_v3,fpo_v4 --task 8 --init 0 --seed 3000
python demo/run_demo.py --methods a1emb_pos,awr --task 0 --init 1 --seed 1000   # vòng sàng lọc (expert bf16)
```

- Video và bảng kết quả ghi vào `demo/out/t{task}_init{k}_s{seed}/`.
- Dòng `[result] ... match=True` nghĩa là replay cho đúng kết quả và độ dài episode như lần eval gốc.
- Giới hạn của `init`: task 0 và 8 dùng init 0–49; các task khác dùng init 0–19 (chỉ seed 1000). Với seed 3000/4000/5000, phương pháp nào cũng chỉ có eval gốc trên task 0 và 8.
- Đã thử (01/11/2026):
  - `fpo_v4,official`, task 0 init 4 seed 5000: v4 thành công 498 bước, official thất bại 520 bước, cả hai `match=True`.
  - `a1emb_pos,awr`, task 0 init 1 seed 1000: cả hai thành công (256 / 259 bước), `match=True`.
- Cần có: checkpoint đã chọn (bảng ở `docs/inventory_storage.md` §2), HF cache `~/.cache/huggingface/hub` (`lerobot/smolvla_libero`, `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`), asset LIBERO `~/.cache/libero`, env conda `lerobot` (LeRobot cài editable từ `ffi26/ffi26/lerobot`).
- **Không** cần decision cache (`iter*/decisions`), dataset demo hay `data/`.

## 3. Archive

| Archive | Nội dung | Vị trí | Trạng thái |
|---|---|---|---|
| A1_research_results | csv/json/md/log/config, `ep*.meta.json`, docs (6998 file, 73,8 MB → 7,9 MB) | `~/ffi26_archives/` | đã tạo, VERIFIED |
| A2_visualization | video, keyframe, frame, hình (2380 file, 286,7 MB → 271,7 MB) | `~/ffi26_archives/` | đã tạo, VERIFIED |
| A3_reproducibility | decision cache, checkpoint không được chọn, optimizer state, smoke/aborted/costtest (1108 file, 105,8 GB) | đề xuất `/mnt/hdd4tb/ffi26_archives/` | **chưa tạo**, chờ duyệt |

Kiểm tra lại một archive:

```bash
python demo/archive_tool.py verify --zip ~/ffi26_archives/A1_research_results.zip
```

Lệnh này kiểm CRC của mọi file, so manifest với danh sách trong zip (tên và kích thước), rồi giải nén 200 file mẫu và so sha256.

Khôi phục về đúng chỗ cũ. Lệnh không ghi đè file đang có; `--only` giới hạn theo tiền tố đường dẫn:

```bash
python demo/archive_tool.py restore --zip /mnt/hdd4tb/ffi26_archives/A3_reproducibility.zip \
    --only improve/outputs/fpo_v4/iter1/decisions
```

Sau khi khôi phục `iter{k}/decisions`, có thể chạy lại:
- `fpo_train*.py` cho iteration đó;
- `fpo_adv_check.py`, `fpo_v3_grad_steps.py`, `fpo_v4_sanity.py`;
- `task0_diag_grad.py`, `fpo_v3_demo_check.py`.
