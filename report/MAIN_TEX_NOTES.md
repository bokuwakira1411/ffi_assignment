# Ghi chú về `report/main.tex` (01/11/2026)

## Đã làm
- **Viết lại toàn bộ `main.tex`.** Bản cũ là kế hoạch A3/A5 viết ngày 31/10, vẫn xem được bằng `git show e245088:report/main.tex`. Bản mới gồm 7 trang, tiếng Việt, biên dịch bằng XeLaTeX:
  - tóm tắt;
  - sơ đồ luồng công việc (TikZ);
  - 1 Study, 2 Reproduce (sơ đồ pipeline TikZ), 3 Analyse, 4 Improve;
  - 5 Hạn chế và kết luận;
  - tài liệu tham khảo.
- **Biên dịch:** `cd report && latexmk -xelatex main.tex`. Không còn lỗi; còn một cảnh báo overfull 2 pt ở bảng tái lập.
- **6 hình sinh từ dữ liệu thật**, không nội suy, không có số tự đặt:

  | Hình | Script | Dữ liệu gốc |
  |---|---|---|
  | `figures/fig_nas.pdf` | `scripts/make_figures.py` | `results/runs/D_nas{1,10,50}/**/eval_info.json` |
  | `figures/fig_round1.pdf` | như trên | `results/round2/episodes.csv` |
  | `figures/fig_fpo_final.pdf` | như trên | `improve/outputs/fpo_v{2,3,4}/eval_s*/**/ep*.meta.json` |
  | `figures/fig_training.pdf` | như trên | `iter*/train/train_log.csv`, `train_info.json`, `results/round2/fpo_v3/weight_delta.csv` |
  | `figures/fig_advantage.pdf` | như trên | `fpo_v2/iter0/{segments,rollouts,groups}.csv`, `results/round2/fpo_v4/sanity.csv` |
  | `figures/fig_keyframes.pdf` | `scripts/make_keyframes.py` | `results/videos/v4_paired/*.mp4` (replay chính xác) |

  Mọi số được vẽ có trong `figures/figure_data.json`.
- **2 khung thất bại** chép từ `docs/failures/frames/B_main/`: `fail_B_t0_e1_s1000.jpg` và `fail_B_t8_e3_s1000.jpg`.
- **Nguồn của số trong bảng:**

  | Bảng | Nguồn |
  |---|---|
  | Tái lập | `docs/reproduce.md` §4 |
  | Analyse | `docs/analyse.md` |
  | Sàng lọc | `results/round2/summary.md` |
  | Giao thức cuối | `results/round2/fpo_v4/pooled.md` |

  Riêng FPO v1 trong giao thức cuối, tính lại từ `improve/outputs/fpo_v2/eval_s*/fpo_v1_it3`: held-out 66/204, so với official 84/204, p = 0,038.
- **Chỉ đưa vào báo cáo kết quả đã có.** Không chạy train hay eval mới cho báo cáo. Video replay chỉ dựng lại những episode đã eval.

## Ô để bạn điền (`\review{}` / `\reviewinline{}`, khung màu cam)
- Tên, ngày.
- Đối chiếu số của bài SmolVLA (71 % Long, 90 % Spatial).
- Các số trích từ 5 bài trong Bảng 1; định nghĩa tỉ lệ FPO gốc (scale $c$, mean hay sum).
- Mô tả hành vi trong video v4 so với official/v3.
- Kết luận và hướng tiếp theo.
- Tên tác giả, năm, venue của tài liệu tham khảo. Ba bài SimpleVLA-RL, RIPT-VLA, VLA-RL chưa ghi tác giả vì trong repo không có nguồn xác nhận.

## Những điều người đọc cần biết (đã ghi trong báo cáo)
- Mỗi phiên bản FPO chỉ train một lần.
- p theo tập con không hiệu chỉnh nhiều phép so sánh.
- Mức tăng trên init held-out chưa có ý nghĩa thống kê.
- Trọng số chỉ thay đổi dưới 0,15 %.
