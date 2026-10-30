# FFI26 - Ghi chú Reproduce: khác biệt so với bản gốc và nhật ký lỗi

Điền lại ngày 30/10/2026 từ các file trong `~/ffi26/results`, `~/ffi26/logs`, `~/ffi26/lerobot` và cache
`~/.cache/huggingface/hub`. Mọi số dưới đây đều tính lại từ `eval_info.json` thô của từng job, **không** lấy từ
`summary_*.csv` (xem mục 6, các file summary của B_main và F_spatial đã cũ).

Quy ước nhãn nguồn:
- **[file]**: đọc trực tiếp từ file trên máy, có ghi đường dẫn.
- **[CHƯA KIỂM CHỨNG]**: không có file nào trên máy chứa giá trị này; giữ lại từ bản template cũ, phải đối chiếu với nguồn gốc trước khi trích dẫn.

Các đường dẫn `results/...` là tương đối với `~/ffi26/`.

## 1. Mô hình và checkpoint

| Mục | Giá trị | Nguồn |
|---|---|---|
| Checkpoint chính | `lerobot/smolvla_libero` | `results/runs/*/lerobot_smolvla_libero/checkpoint_revision.txt` |
| Commit SHA đã dùng | `a24f1f380461b665158d52acd140152c79e8b36d` (last_modified 2026-10-12 14:03:28 UTC). Giống nhau ở cả 7 thư mục `runs/*` và ở `probe_report.json` | như trên; `results/probe_report.json` → `0_versions.policy_commit_sha` |
| Khởi tạo từ | `lerobot/smolvla_base` | `config.json` của snapshot a24f1f3: `"pretrained_path": "lerobot/smolvla_base"` |
| Repo tác giả ghi trong config | `pepijn223/smolvla_libero` | `config.json`: `"repo_id"` |
| Đối chứng dự kiến | `lerobot/smolvla_base`, commit `5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6`. **Không chạy được trên LIBERO** (mục 4) | `results/smoke_20261030_110908/checkpoint_revision.txt` |
| Checkpoint cũ (đã bỏ) | `HuggingFaceVLA/smolvla_libero`, commit `6721902bc4d61e50a3bfdb11dfb4cb626f05d102`. Chỉ dùng trong `smoke_20261029_103016`, `demo_tasks`, `demo_spatial` | `checkpoint_revision.txt` của 3 thư mục đó; `ckpt_files/.cache/huggingface/trees/6721902b….json` |

### 1b. Checkpoint cũ khác checkpoint chính thế nào (để không lẫn số liệu)

`ckpt_files/config.json` là config của `HuggingFaceVLA/smolvla_libero` (cache trỏ tới commit 6721902). So với `lerobot/smolvla_libero`:

| Trường | HuggingFaceVLA/smolvla_libero | lerobot/smolvla_libero |
|---|---|---|
| vlm_model_name | SmolVLM2-500M-**Instruct** | SmolVLM2-500M-**Video**-Instruct |
| num_vlm_layers | 0 | 16 |
| expert_width_multiplier | 0.5 | 0.75 |
| freeze_vision_encoder / train_expert_only | true / true | false / false |
| n_action_steps | 1 | 50 |
| scheduler_decay_steps | 30000 | 25000 |
| Tên camera đầu vào | `image`, `image2` | `camera1`, `camera2` |

Kết luận: đây là hai mô hình khác nhau, số liệu của checkpoint cũ không gộp được với checkpoint chính.

## 2. Recipe huấn luyện công bố

| Mục | Giá trị | Nguồn |
|---|---|---|
| Dataset | `lerobot/libero`, revision: null (không ghim) | [CHƯA KIỂM CHỨNG]: chỉ có trong `train_config.json`, file này không có trong cache |
| Số step / batch | 25.000 / 32 | [CHƯA KIỂM CHỨNG] (train_config.json). `config.json` có `scheduler_decay_steps: 25000`, khớp với 25.000 step nhưng không phải bằng chứng trực tiếp về số step |
| Optimizer | lr 1e-4, betas (0.9, 0.95), weight decay 1e-10, grad clip 10, warmup 1.000 step, decay về 2,5e-6 | `config.json` (`optimizer_*`, `scheduler_*`). Loại optimizer (AdamW), dạng lịch (cosine) và seed 1000: [CHƯA KIỂM CHỨNG] |
| Tham số được cập nhật | toàn bộ: `freeze_vision_encoder: false`, `train_expert_only: false`, `use_peft: false` | `config.json` |
| Số tham số | 450.046.176 tổng; 392.904.096 trainable. Phần không trainable: `lm_head` 47.308.800 và 9.833.280 trong `text_model` | `results/probe_report.json` → `2_parameters` |
| Augmentation ảnh | tắt | [CHƯA KIỂM CHỨNG] (train_config.json) |
| Kiến trúc | SmolVLM2-500M-Video-Instruct, 16 layer VLM, 16 layer expert, expert width x0,75, `cross_attn`, self-attn mỗi 2 layer, vision encoder 12 layer | `config.json`; `probe_report.json` → `modulelist_lengths` |
| Action | chunk_size 50, n_action_steps 50, 10 bước flow (Euler), action 7 chiều, state 8 chiều | `config.json`; `probe_report.json` → `6_inference_trace` |
| Env ghi trong train_config | control_mode relative, env.task libero_spatial, eval_freq 0 | [CHƯA KIỂM CHỨNG] (train_config.json) |

## 3. Bảng khác biệt giữa bản gốc và lần chạy

| Hạng mục | Bản gốc / docs | Lần chạy này | Nguồn / ghi chú |
|---|---|---|---|
| Phần cứng | [CHƯA KIỂM CHỨNG: không có trong file nào trên máy, lấy từ paper SmolVLA] | NVIDIA GeForce RTX 5070 Ti (sm_120, 16,6 GB), driver 580.178.04, CUDA 12.8, host `thanhlt-server` | `results/env_report.json`, `results/versions.txt` |
| LeRobot | [CHƯA KIỂM CHỨNG: config.json không ghi phiên bản LeRobot dùng lúc train] | 0.6.2, commit `72e1d9e5e76ee719bacfc78bc34d817bee5fb521` (nhánh main), cài editable; `git status` của repo sạch | `results/versions.txt`, `results/pip_freeze.txt` |
| PyTorch / Python | docs: torch >= 2.10, Python >= 3.12 | torch 2.11.0+cu128, torchvision 0.26.0+cu128, Python 3.12.16 | `results/env_report.json`, `pip_freeze.txt` |
| Mô phỏng | – | mujoco 3.8.1, robosuite 1.4.0, hf_libero 0.1.4, transformers 5.5.4, `MUJOCO_GL=egl` | `pip_freeze.txt`, `env_report.json` |
| Suite eval | paper: 4 suite (Spatial, Object, Goal, Long) | **2 suite**: `libero_10` (LIBERO-Long) và `libero_spatial`. Không chạy Object, Goal | `cmd.txt` của các job |
| Episode / task | paper: 10 trial/task; docs: 10 | B_main, F_spatial: 10 episode/job x 3 seed = **30 episode/task**. D_nas1, D_nas50: **50 episode/task**, 1 seed. F_spatial_abs, F_spatial_nas50: 10 episode/task, 1 seed. Mọi job đều chạy đủ số episode đã đặt (`n_episodes` trong eval_info.json = N_EP ở cả 100 job) | `eval_info.json` |
| Trạng thái đầu (init state) | [CHƯA KIỂM CHỨNG: paper không có trên máy] | Episode thứ i dùng init state số i của task (cố định, **không phụ thuộc seed**). Vì vậy 3 seed của B_main/F_spatial đều dùng **cùng 10 init state 0–9**; D dùng init state 0–49 | `lerobot/src/lerobot/envs/libero.py:173-175, 344-346, 437`. Kiểm chứng: 10 episode đầu của D_nas1 trùng từng episode với B_main seed 1000 ở 99/100 episode (chỉ khác task 8, episode 8) |
| Seed | docs: 3 (paper không nêu seed eval) | B_main, F_spatial: 1000, 1001, 1002. D_*, F_spatial_abs, F_spatial_nas50: chỉ 1000 | `cmd.txt` |
| Huấn luyện | paper (Sec 4.3): 100.000 step, batch 64, đa task trên 4 suite [CHƯA KIỂM CHỨNG] | Không train. Dùng checkpoint công bố (25.000 step / batch 32 theo train_config [CHƯA KIỂM CHỨNG]) | KHÁC: không kỳ vọng tái lập đúng điểm số paper |
| n_action_steps | paper (Sec 4.3): dự đoán lại sau mỗi action (ngầm hiểu = 1) [CHƯA KIỂM CHỨNG]; config checkpoint: 50 | **1** cho B_main, F_spatial, F_spatial_abs, D_nas1. **50** cho D_nas50, F_spatial_nas50. Luôn truyền tường minh `--policy.n_action_steps` | `cmd.txt`; eval.log in lại `'n_action_steps': 1` |
| control_mode | paper: delta end-effector (tương ứng relative) [CHƯA KIỂM CHỨNG] | **relative** cho mọi job trừ F_spatial_abs (**absolute**) | `cmd.txt`. Kết quả phép thử ở mục 3b |
| Giới hạn bước / episode | – | libero_10: 520 bước; libero_spatial: 280 bước (`episode_length: None` nên dùng mặc định của suite) | `libero.py:98-104`; eval.log in "Running rollout with at most 520/280 steps" |
| Batch eval | – | `--eval.batch_size=5` (5 env song song, async), `--env.max_parallel_tasks=1` | `cmd.txt`, eval.log |
| Đổi tên camera | – | `--rename_map` image→camera1, image2→camera2 cho mọi job hợp lệ | `cmd.txt` |
| Dataset khi chạy probe | lúc train: không ghim [CHƯA KIỂM CHỨNG] | `lerobot/libero` revision `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4` (1.693 episode, 273.465 frame, fps 10) | `results/probe_report.json` → `3_data_and_preprocess`, `args` |
| Số liệu paper, SmolVLA 0.45B (Table 2, p.10) | Spatial 90 / Object 96 / Goal 92 / Long 71 / TB 87,3 [CHƯA KIỂM CHỨNG: PDF paper không có trên máy] | **Spatial 79,0** (237/300) / Object – / Goal – / **Long 52,7** (158/300) / TB: không tính được (thiếu 2 suite) | Cấu hình: NAS=1, relative, 3 seed x 10 episode. Chênh lệch so với số paper: Spatial −11,0 điểm, Long −18,3 điểm |

### 3a. Kết quả theo task (tính lại từ eval_info.json)

CI 95% là khoảng Wilson trên tổng episode gộp (cùng công thức với `pc_success_ci95` trong eval_info.json).
Lưu ý: CI gộp coi 30 episode là độc lập, nhưng chúng chỉ phủ 10 init state (mục 3, dòng "Trạng thái đầu"), nên CI này hẹp hơn thực tế.

**B_main: libero_10, NAS=1, relative, seed 1000/1001/1002, 10 ep/seed**

| Task | Mô tả | s1000 | s1001 | s1002 | Gộp |
|---|---|---|---|---|---|
| 0 | put both the alphabet soup and the tomato sauce in the basket | 1/10 | 0/10 | 0/10 | 1/30 (3,3%) |
| 1 | put both the cream cheese box and the butter in the basket | 6/10 | 5/10 | 5/10 | 16/30 (53,3%) |
| 2 | turn on the stove and put the moka pot on it | 9/10 | 10/10 | 9/10 | 28/30 (93,3%) |
| 3 | put the black bowl in the bottom drawer of the cabinet and close it | 10/10 | 10/10 | 10/10 | 30/30 (100%) |
| 4 | put the white mug on the left plate and put the yellow and white mug on the right plate | 3/10 | 3/10 | 1/10 | 7/30 (23,3%) |
| 5 | pick up the book and place it in the back compartment of the caddy | 10/10 | 9/10 | 8/10 | 27/30 (90,0%) |
| 6 | put the white mug on the plate and put the chocolate pudding to the right of the plate | 3/10 | 2/10 | 4/10 | 9/30 (30,0%) |
| 7 | put both the alphabet soup and the cream cheese box in the basket | 5/10 | 6/10 | 5/10 | 16/30 (53,3%) |
| 8 | put both moka pots on the stove | 1/10 | 0/10 | 0/10 | 1/30 (3,3%) |
| 9 | put the yellow and white mug in the microwave and close it | 7/10 | 8/10 | 8/10 | 23/30 (76,7%) |
| **Tổng** | | 55/100 | 53/100 | 50/100 | **158/300 = 52,7%** (CI 47,0–58,2); độ lệch chuẩn giữa 3 seed 2,5 điểm |

**F_spatial: libero_spatial, NAS=1, relative, seed 1000/1001/1002, 10 ep/seed**

| Task | Mô tả | s1000 | s1001 | s1002 | Gộp |
|---|---|---|---|---|---|
| 0 | pick up the black bowl between the plate and the ramekin and place it on the plate | 10/10 | 9/10 | 10/10 | 29/30 (96,7%) |
| 1 | … next to the ramekin … | 8/10 | 8/10 | 8/10 | 24/30 (80,0%) |
| 2 | … from table center … | 10/10 | 10/10 | 10/10 | 30/30 (100%) |
| 3 | … on the cookie box … | 9/10 | 10/10 | 9/10 | 28/30 (93,3%) |
| 4 | … in the top drawer of the wooden cabinet … | 6/10 | 8/10 | 6/10 | 20/30 (66,7%) |
| 5 | … on the ramekin … | 0/10 | 0/10 | 0/10 | **0/30 (0%)** |
| 6 | … next to the cookie box … | 9/10 | 10/10 | 10/10 | 29/30 (96,7%) |
| 7 | … on the stove … | 9/10 | 8/10 | 9/10 | 26/30 (86,7%) |
| 8 | … next to the plate … | 8/10 | 8/10 | 7/10 | 23/30 (76,7%) |
| 9 | … on the wooden cabinet … | 9/10 | 9/10 | 10/10 | 28/30 (93,3%) |
| **Tổng** | | 78/100 | 80/100 | 79/100 | **237/300 = 79,0%** (CI 74,0–83,2); độ lệch chuẩn giữa 3 seed 1,0 điểm |

Mô tả task lấy từ `libero.libero.benchmark` (hf_libero 0.1.4). Task 1–9 của spatial đều có dạng "pick up the black bowl … and place it on the plate".

### 3b. Ablation (cặp episode cùng init state, cùng seed 1000)

Kiểm định McNemar chính xác (hai phía) trên các cặp episode cùng task và cùng init state.

| Phép thử | A | B | Episode chỉ A thành công / chỉ B thành công | p |
|---|---|---|---|---|
| D: libero_10, NAS 1 vs 50, 50 ep/task | NAS=1: 273/500 = **54,6%** (CI 50,2–58,9) | NAS=50: 239/500 = **47,8%** (CI 43,5–52,2) | 106 / 72 | 0,013 |
| F: libero_spatial, NAS 1 vs 50, 10 ep/task | NAS=1: 78/100 (seed 1000 của F_spatial) | NAS=50: 58/100 = **58,0%** (CI 48,2–67,2) | 23 / 3 | 8,8e-5 |
| E: libero_spatial, relative vs absolute, NAS=1, 10 ep/task | relative: 78/100 | absolute: 2/100 = **2,0%** (CI 0,6–7,0) | 76 / 0 | 2,7e-23 |

D theo task (thành công trên 50; NAS=1 / NAS=50): 0: 12/12, 1: 29/21, 2: 45/37, 3: 46/40, 4: 20/17, 5: 43/37, 6: 18/11, 7: 23/18, **8: 7/19**, 9: 30/27.
NAS=1 cao hơn hoặc bằng ở 9/10 task. **Ngoại lệ là task 8** (put both moka pots on the stove): NAS=50 thắng 19 so với 7, cả 12 cặp lệch đều nghiêng về NAS=50.

E: với `absolute`, chỉ task 0 và task 1 có 1/10 thành công, 8 task còn lại 0/10. Kết luận cho câu hỏi control_mode: checkpoint này phải chạy với `relative`.

Thời gian (tổng giây wall-clock của 10 job, `timing.csv`): D_nas1 6.309 s, D_nas50 1.336 s (NAS=1 chậm hơn 4,7 lần). Thời gian rollout mỗi episode (`overall.eval_ep_s`, có 5 env song song): NAS=1 trên libero_10 7,4–14,8 s; NAS=50 trên libero_10 1,8–2,7 s.

### 3c. Độ tái lập giữa hai lần chạy cùng cấu hình

B_main seed 1000 và 10 episode đầu của D_nas1 có cùng checkpoint, seed, NAS, control_mode và init state. Kết quả trùng nhau ở 99/100 episode (tổng thành công 55 so với 56). Episode duy nhất lệch là task 8, episode 8: B_main 0, D_nas1 1. Như vậy pipeline gần tất định nhưng **không tất định từng bit**: chạy lại cùng lệnh có thể lệch vài episode.

## 4. Nhật ký lỗi

Thời điểm lấy từ tên thư mục, mtime file hoặc dòng log (giờ +07).

| Thời điểm | Lỗi | Nguyên nhân | Cách sửa | Bằng chứng |
|---|---|---|---|---|
| 30/10 10:30, 10:45 | `Feature mismatch … Extra features: ['observation.images.image', 'observation.images.image2']` | env LIBERO trả camera `image`/`image2`, policy cần `camera1`/`camera2`; lệnh chưa có `--rename_map` | thêm `--rename_map` (image→camera1, image2→camera2) | `results/smoke_20261030_103053/eval.log`, `smoke_20261030_104519/eval.log` |
| 30/10 10:50 | `smoke_test.sh: line 70: lerobot-eval: command not found` | chạy khi chưa `conda activate lerobot` | bật env `lerobot` | `results/smoke_20261030_105033/eval.log` (`checkpoint_revision.txt` cũng rỗng) |
| 30/10 11:03 | Lần chạy dừng ở bước 477/520 của batch đầu, không có traceback, không có `eval_info.json`, không có `timing.txt` | log không ghi nguyên nhân dừng | chạy lại thành B_main | `results/smoke_20261030_110353/eval.log` (NAS=1, libero_10, task 0) |
| 30/10 11:09 | `AssertionError: environment got invalid action dimension -- expected 7, got 6` khi chạy `lerobot/smolvla_base` | `config.json` của smolvla_base (commit 5e8d12a): action **6** chiều, state **6** chiều, **3** camera (camera1/2/3). LIBERO cần action 7 chiều, state 8 chiều | không chạy base thẳng trên LIBERO được. Đối chứng thay bằng số baseline trong Table 2 của paper, ghi rõ là số của họ | `results/smoke_20261030_110908/eval.log`; `~/.cache/huggingface/hub/models--lerobot--smolvla_base/snapshots/5e8d12a…/config.json` |
| không rõ (trước 30/10 11:09) | `Feature mismatch` khi chạy smolvla_base | thiếu `--rename_map` | (sau đó gặp lỗi action 6 chiều ở dòng trên) | `results/demo_spatial_smolvla_base/eval.log` |
| 30/10 11:11 | `KeyboardInterrupt`, 3 giây sau khi bắt đầu | dừng tay. Lệnh này thiếu cờ NAS và CM (TAG mặc định `run`) | chạy lại với TAG=B_main, NAS=1, CM=relative | `results/runs/run/lerobot_smolvla_libero/task0_seed1000/eval.log`, `cmd.txt` |
| 30/10 ≤12:23 | 20 job B_main (seed 1001, 1002) kết thúc với **rc=127**, 0 giây, lặp 2 lượt (40 dòng) | rc=127 là mã "command not found" của shell. eval.log của các job này đã bị lần chạy lại ghi đè nên thông báo gốc không còn | chạy lại seed 1001, 1002 (12:26–13:11), tất cả rc=0 | `results/runs/B_main/timing.csv.bak` |

## 5. Việc cần ghi khi báo cáo kết quả

- **Smoke test cũ**: 0/1 thành công, 209,1 giây/episode, libero_10 task 0, checkpoint `HuggingFaceVLA/smolvla_libero` @6721902 (`results/smoke_20261029_103016`). `demo_tasks` (libero_10 task 0/1/2: 0/1, 1/1, 0/1) và `demo_spatial` (spatial task 0: 1/2, task 1: 2/2) cũng dùng checkpoint cũ này. Không gộp các số này với checkpoint chính.
- **Smoke test checkpoint chính trước khi chạy B**: libero_10 task 0, NAS=50, chạy 2 lần, mỗi lần 5 episode và mỗi lần 1/5 (`smoke_20261030_104640`, `smoke_20261030_105113`). libero_spatial task 0, NAS=1, 5 episode: 5/5 (`smoke_20261030_110519`).
- **Giới hạn bước của episode**: libero_spatial 280, libero_object 280, libero_goal 300, libero_10 520, libero_90 400. Các giá trị này là mặc định khi `episode_length` là None (`libero.py:98-104, 189-193`).
- **Cách env xác định thành công**: sau mỗi bước, env gọi `self._env.check_success()` (predicate của LIBERO). Episode kết thúc ngay khi thành công hoặc khi `done` (`libero.py:376-377`). Không đạt trước giới hạn bước thì tính là thất bại. Trước khi tính bước, env chạy `num_steps_wait` bước no-op để vật thể ổn định (`libero.py:351-352`).
- **Số episode thực tế mỗi job**: đã kiểm tra cả 100 job (6 thư mục runs, trừ `run` bị dừng tay), `n_episodes` trong eval_info.json luôn bằng N_EP (10 hoặc 50).
- **Video**: mỗi job chỉ lưu **10 video** (`lerobot_eval.py:823`: `max_episodes_rendered = 10`). Với D_nas1/D_nas50 (50 episode), 40 episode còn lại không có video. Phải tính điều này khi phân loại lỗi bằng video.
- **Seed không đổi cảnh ban đầu**: 3 seed dùng chung 10 init state (mục 3). Phải ghi rõ "30 episode trên 10 init state", không ghi "30 cảnh độc lập".

## 6. Lưu ý về file tổng hợp

- `runs/B_main/summary_per_job.csv`, `summary_per_task.csv`, `episode_log.csv` được tạo lúc 12:15, **trước khi** seed 1001/1002 chạy (12:26–13:11). Vì vậy chúng chỉ chứa seed 1000 (`n_seeds=1`, 100 dòng episode).
- `runs/F_spatial/*.csv` được tạo lúc 16:48, trước seed 1001/1002 (16:52–17:16). Các file này cũng chỉ chứa seed 1000.
- `runs/F_spatial_abs` và `runs/F_spatial_nas50` chưa có file summary.
- Cần chạy lại `python collect_results.py --tag <TAG>` cho B_main, F_spatial, F_spatial_abs, F_spatial_nas50 trước khi dùng các CSV này. Số trong tài liệu này không bị ảnh hưởng vì được tính thẳng từ eval_info.json.
