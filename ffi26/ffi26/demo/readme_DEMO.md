# Demo

`run_demo.py` replay lại một episode của eval gốc với một hoặc nhiều phương pháp, rồi ghi video.
Kết quả phải giống lần eval gốc: thành công hay thất bại, và số bước.

```bash
conda activate lerobot && cd ~/ffi26
python demo/run_demo.py --list                                     # các phương pháp, checkpoint, seed có eval gốc
python demo/run_demo.py --methods official,fpo_v4 --task 0 --init 4 --seed 5000
```

- Video và bảng ghi vào `demo/out/t{task}_init{k}_s{seed}/`.
- Dòng `[result] ... match=True` nghĩa là replay khớp với eval gốc.
- `--init`: task 0 và 8 dùng 0–49; các task khác dùng 0–19 và chỉ có seed 1000.
- Mỗi phương pháp mất khoảng 1–3 phút trên GPU.
- Checkpoint FPO không có trong repo. Muốn demo các phương pháp này, chạy train trước (`readme.md` ở gốc repo).
  Đường dẫn checkpoint của từng phương pháp nằm trong `methods.yaml`.
- `official` tự tải từ HuggingFace.

Video có sẵn, không cần GPU: `results/videos/readme_VIDEOS.md`.
