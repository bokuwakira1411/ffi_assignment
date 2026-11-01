# Kiểm kê thí nghiệm, môi trường demo và đề xuất dọn ổ đĩa

Viết ngày 01/11/2026, 18:35. Nguồn: quét `~/ffi26` (= `ffi26/ffi26/`), `docs/WORKLOG.md`, các config eval và `train_info.json`.
Dung lượng tính bằng `du -b`, đơn vị GiB (1024³ byte). **Chưa xoá file gốc nào.**

## 1. Danh sách phương pháp và thí nghiệm đã chạy

| # | Phương pháp / thí nghiệm | Thư mục | Trạng thái | Checkpoint dùng để báo cáo | Kết quả eval | Video / khung hình | Log, config |
|---|---|---|---|---|---|---|---|
| R | Reproduce: checkpoint gốc, NAS 1/10/50, Spatial | `results/runs/{B_main,D_nas1,D_nas10,D_nas50,F_spatial*}` | xong | `lerobot/smolvla_libero` (HF cache) | `eval_info.json` từng job; `docs/reproduce.md` | 1400 mp4 trong `results/runs/**/videos`; `docs/failures/frames/` (262 jpg) | `cmd.txt`, `eval.log` |
| R' | Checkpoint cũ `HuggingFaceVLA/smolvla_libero` | `results/demo_tasks/` | bỏ (model khác) | — | không dùng | 3 mp4 | — |
| 0 | Baseline có log predicate (phase 0–3) | `improve/outputs/{baseline,eval_baseline,speedtest}` | xong | official | `eval_baseline/task{0,8}` (bf16, seed 1000) | `baseline/frames`, 40 mp4 | `*.log` |
| 1 | Thu rollout vòng 1 (256 rollout, reward_r2) | `improve/outputs/r2/round1` | xong | — | — | — | `rollouts.csv`, `segments.csv`, `logs/` |
| A1 | Nhãn advantage trong prompt (kiểu RECAP) | `improve/outputs/r2/train_a1_round1` | xong | `checkpoints/002000` | `r2/eval/a1_r1_{pos,none,neg}` | — | `results/round2/configs/eval_round1.yaml` |
| A1e | Nhãn advantage qua embedding | `improve/outputs/r2/train_a1emb_round1` | xong | `checkpoints/002000` | `r2/eval/a1emb_r1_*` | — | `eval_round1_emb.yaml` |
| A2 | AWR 2000 bước | `improve/outputs/r2/train_awr_round1` | xong | `checkpoints/002000` | `r2/eval/awr_r1` | — | `eval_round1_awr.yaml` |
| A2' | AWR 6000 bước | `improve/outputs/r2/train_awr6k_round1` | xong | `002000`, `004000`, `006000` (cả 3 đã eval) | `r2/eval/awr6k_r1_s*` | — | `eval_round1_awr6k.yaml` |
| F1 | FPO v1 (bf16, batch 32) | `improve/outputs/fpo` | xong | θ3 = `iter2/train/pretrained_model` | sàng lọc: `r2/eval/fpo_it{1..4}`, `r2/eval_s2000/fpo_it3`; giao thức cuối: `fpo_v2/eval_s*/fpo_v1_it3` | `results/videos/1_v1_v2/v1t3/` | `fpo.yaml`, `loop.log` |
| F2 | FPO v2 (fp32, batch 128, neo BC) | `improve/outputs/fpo_v2` | xong | θ4 = `iter3/train/pretrained_model` | `fpo_v2/eval_s{1000,3000,4000,5000}` | `results/videos/1_v1_v2/v2/`, `2_v2_forget/` | `fpo_v2.yaml`, `iter_summary.csv` |
| F3 | FPO v3 (+ demo replay, chuẩn hoá theo task) | `improve/outputs/fpo_v3` | xong | θ4 = `iter3/train/pretrained_model` | `fpo_v3/eval_s*` | `results/videos/3_v4_paired/v3/` | `fpo_v3.yaml` |
| F4 | FPO v4 (trọng số theo segment) | `improve/outputs/fpo_v4` | xong | θ4 = `iter3/train/pretrained_model` | `fpo_v4/eval_s*`, `results/round2/fpo_v4/pooled.md` | `results/videos/3_v4_paired/` | `fpo_v4.yaml` |
| T | Chẩn đoán task 0: nhãn + cosine gradient (phiên khác) | `improve/outputs/task0_diag` | xong (GPU n=32, n=256) | — | `tables.md`, `grad_cos_*.csv` | — | `run_after_v4.log` |
| L | Failure labeler v2 (official, v1, v2, v3, v4) | `improve/outputs/failure_labels{,_v3,_v4}` | xong (tự động) | — | `summary.md` | `render/` (20 cấu hình, chưa render) | — |
| S | Smoke / dry-run / thử chi phí | `improve/outputs/fpo/smoke`, `fpo_v2_aborted`, `fpo_v2_costtest` | dừng / chỉ thử | — | số đã ghi trong WORKLOG | — | `loop.log` |

**Không tìm thấy:** PPO/GRPO độc lập (ngoài FPO), Filtered BC độc lập, Flow-SDE, Flow-Noise. Các hướng này chỉ có trong
đề xuất của `main.tex` cũ và trong `docs/study.md`. A1 là cách gán nhãn advantage kiểu RECAP; A2 là AWR.

## 2. Môi trường demo (mức B): cần giữ giải nén

| Thành phần | Đường dẫn | GiB |
|---|---|---|
| Checkpoint đã chọn, FPO | `improve/outputs/fpo/iter2`, `fpo_v{2,3,4}/iter3` → `train/pretrained_model` | 4,1 |
| Checkpoint đã eval, vòng sàng lọc | `r2/train_{a1,a1emb,awr}_round1/checkpoints/002000/pretrained_model`, `r2/train_awr6k_round1/checkpoints/00{2,4,6}000/pretrained_model` | 5,1 |
| Checkpoint gốc, VLM nền | `~/.cache/huggingface/hub/models--lerobot--smolvla_libero`, `models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct` | (cache) |
| Code, env | `improve/{src,scripts}`, `ffi26/ffi26/lerobot` (editable), env conda `lerobot`, `~/.cache/libero` | — |
| Registry, script demo | `demo/methods.yaml`, `demo/run_demo.py` (đã test 2 lần, §README) | — |
| Eval gốc (để kiểm tra replay, cho labeler) | `improve/outputs/*/eval_s*/`, `r2/eval*/`, `eval_baseline/` (npz + meta.json) | 0,2 |

**Decision cache không cần cho demo.** `phase3_eval*.py` và `rollout.py` không đọc `iter*/decisions` (đã grep).

## 3. Decision cache `iter*/decisions/*.npz` dùng cho việc gì

| Việc | Cần decisions? | Script đọc |
|---|---|---|
| Suy luận, eval, render video, demo | **không** | `phase3_eval.py`, `phase3_eval_video.py`, `demo/run_demo.py` |
| Train lại đúng iteration k (cùng dữ liệu) | có, đủ ảnh (iter k) | `fpo_train{,_v2,_v3}.py` (`load_decisions`) |
| Chạy lại cả loop từ đầu | không (thu rollout lại), nhưng kết quả có thể lệch nhỏ | `fpo_*_loop.sh` |
| `fpo_adv_check.py`, `fpo_v3_grad_steps.py`, `fpo_v4_sanity.py` | có, nhưng **chỉ khoá vô hướng** (adv, rejected, task_id, group, env, step, n_exec) | như tên |
| `task0_diag_grad.py`, `fpo_v3_demo_check.py`, `fpo_verify.py` | có, đủ ảnh | như tên |
| Labeler thất bại | không (dùng `eval_s*/…/ep*.npz`, `iter*/logs`) | `failure_labels*.py` |

**Gợi ý (chưa làm):** trước khi xoá decisions, ghi thêm một file nhỏ `iter*/decisions_meta.npz` chỉ chứa các khoá vô hướng
(vài MB mỗi iteration), rồi sửa 3 script phân tích để đọc file này khi không có `decisions/`. Như vậy các phân tích advantage và số bước vẫn chạy được mà không cần giải nén 72 GiB.

## 4. Kiểm kê dung lượng (dữ liệu chưa track trong git, trừ khi ghi khác)

| Nhóm | Đường dẫn | GiB | Git | Xem kết quả | Demo | Tái lập eval | Tái lập train |
|---|---|---|---|---|---|---|---|
| Decision cache | `outputs/fpo{,_v2,_v3,_v4}/iter*/decisions` (v3, v4 iter0 là symlink tới v2) | 71,8 | không (`*.npz`) | không | không | không | có |
| Checkpoint FPO không chọn | `fpo/iter{0,1,3}`, `fpo_v{2,3,4}/iter{0,1,2}` → `train/pretrained_model` | 12,3 | không (`*.safetensors`) | không | không | chỉ cho eval FPO v1 θ1/θ2/θ4 (vòng sàng lọc) | để chạy tiếp từ iteration đó |
| Checkpoint vòng 1 chưa eval | `r2/train_{a1,a1emb,awr}_round1/checkpoints/001000` | 2,5 | không | không | không | không | không |
| Optimizer state vòng 1 | `r2/train_*/checkpoints/0*/training_state` | 3,5 | không | không | không | không | chỉ để train tiếp |
| Smoke FPO v1 | `outputs/fpo/smoke` | 4,3 | không | không | không | không | không |
| v2 bị dừng / thử chi phí | `outputs/fpo_v2_aborted`, `outputs/fpo_v2_costtest` | 4,1 | không | không | không | không | không |
| Checkpoint đã chọn + đã eval | §2 | 9,2 | không | — | **có** | **có** | (điểm xuất phát) |
| Dataset | `improve/data/{lerobot_libero,r2,fpo_v3,demos_targets}` | 5,2 | không (gitignore) | không | không | không | có (vòng 1: `r2`; v3/v4: `fpo_v3` cache demo) |
| Log rollout, eval npz | `iter*/logs`, `eval_s*/**/ep*.npz` | 0,3 | không | không | có (kiểm tra replay) | có | có (labeler) |
| Video, khung hình, hình | `results/runs/**/videos`, `results/videos`, `docs/failures/frames`, `report/figures` | 0,3 | phần lớn có | **có** | có | — | — |
| Bảng, log, config | csv/json/md/yaml | 0,07 | phần lớn có | **có** | có | có | có |

## 5. Archive

| Archive | Nội dung | File | Gốc | Zip | Tỉ lệ | Vị trí | Trạng thái |
|---|---|---|---|---|---|---|---|
| A1 research results | bảng, log, config, `ep*.meta.json`, docs, WORKLOG, main.tex | 6998 | 73,8 MB | 7,9 MB | 9,4× | `~/ffi26_archives/` | **VERIFIED** (CRC, manifest, 200 mẫu sha256) |
| A2 visualization | mp4, png, jpg, hình báo cáo, `main.pdf` | 2380 | 286,7 MB | 271,7 MB | 1,06× | `~/ffi26_archives/` | **VERIFIED** |
| A3 reproducibility | toàn bộ nhóm "Decision cache" + "Checkpoint không chọn" + "chưa eval" + "Optimizer" + "Smoke" + "aborted/costtest" ở §4 | 1108 | 105,8 GB (98,5 GiB) | ước tính ~55 GB | ~1,9× (đo trên 1 file decisions: 85,4 → 41,5 MB, zip mức 6; safetensors gần như không nén được) | đề xuất `/mnt/hdd4tb/ffi26_archives/` | **chưa tạo** |

- A3 không tạo được trên ổ chính, vì zip ~55 GB mà ổ chỉ trống 58,8 GB.
- `/mnt/hdd4tb` (3,6 TB, trống 2,9 TB) thuộc user `student` và ghi được, nhưng có thư mục của người khác (`thanhlt`, `minhlesonle`). Cần bạn đồng ý mới ghi lên đó.
- Ghi zip thẳng lên `/mnt/hdd4tb` thì không tốn chỗ tạm trên ổ chính.
- Thời gian ước tính: zip mức 6 khoảng 23 MB/s mỗi luồng, tức ~75 phút cho 105 GB. Nếu chia mỗi phiên bản FPO một file zip (`A3_fpo_v1.zip` … `A3_r2.zip`) thì khôi phục được từng phần.
- Dung lượng lấy lại sau khi xoá bản gốc của nhóm 2: **~98,5 GiB** (ổ chính từ 94 % xuống khoảng 79 %: 842 − 98,5 = 743,5 trên 936 GiB).

## 6. Danh sách dọn (CHỜ BẠN DUYỆT, chưa xoá gì)

### Nhóm 1: xoá được ngay (không cần archive)

| Đường dẫn | Dung lượng | Lý do |
|---|---|---|
| `improve/scripts/__pycache__`, `improve/src/__pycache__` | < 1 MB | tự sinh lại |
| `docs/huong_dan_doc_ket_qua.md` | 8 KB | gitignore, đã cũ (nói "FPO đang eval"); `readme.md` và `demo/README.md` thay thế |
| `results/demo_tasks/` | 1,1 MB (đang track) | chạy với checkpoint cũ `HuggingFaceVLA/smolvla_libero`, không dùng trong kết quả; vẫn còn trong git history (`git rm`) |

### Nhóm 2: archive vào A3 rồi mới xoá bản gốc

Gồm mọi dòng của §4 từ "Decision cache" tới "v2 bị dừng / thử chi phí", tổng 98,5 GiB. Thứ tự làm:
1. Tạo A3 trên `/mnt/hdd4tb` (`archive_tool.py build`).
2. Verify.
3. Thử `restore --only` một iteration vào thư mục tạm, rồi chạy `fpo_v4_sanity.py` trên đó.
4. Mới xoá.

Phần mất đi (cho tới khi khôi phục):
- train lại đúng một iteration cũ;
- các phân tích advantage và số bước, trừ khi làm `decisions_meta.npz` (§3);
- eval lại FPO v1 θ1/θ2/θ4.

### Nhóm 3: phải giữ giải nén

- Checkpoint ở §2, kèm HF cache, `ffi26/ffi26/lerobot` và symlink `~/ffi26`.
- Toàn bộ `results/`, `improve/{src,scripts,configs}`, eval gốc (`eval_s*`, `r2/eval*`, `eval_baseline`), `iter*/logs`, các file csv/json/md/log trong `iter*`.
- `task0_diag/`, `failure_labels*`, `demo/`, `report/`, `docs/`, `~/ffi26_archives/`.
- `improve/data/fpo_v3` (cache demo, 1 GiB; cần để train v3/v4) và `improve/data/r2` (dataset vòng 1). Hai thư mục này có thể chuyển sang A3 nếu bạn chấp nhận mất khả năng train lại mà không cần giải nén.

## 7. Câu hỏi còn mở

1. **18 file mp4 của `results/videos/v2_forget/` (nay là `2_v2_forget/`)** đã nằm trong commit `9bf49db` của bạn. Có cần `git rm --cached` không?
2. **Cho phép ghi A3 (~55 GB) lên `/mnt/hdd4tb/ffi26_archives/` không?** Nếu có, chọn: một file zip hay mỗi phiên bản một file? Có làm `decisions_meta.npz` (§3) trước khi xoá không?
3. **Replay video:** 167/168 episode replay trùng bản gốc. Episode lệch là v3, seed 3000, task 8, init 1: vẫn thành công nhưng 429 bước so với 431. Episode này không thuộc tập được chọn làm video, nhưng cho thấy replay không tất định tuyệt đối.
