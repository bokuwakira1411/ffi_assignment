"""Trích khung hình điển hình cho mọi episode thất bại có video của một run.

Cách dùng:  python docs/failures/tools/extract_frames.py RUN
  RUN: tên thư mục trong ffi26/ffi26/results/runs (ví dụ B_main, D_nas1).

Kết quả:
  docs/failures/frames/RUN/tT_eE_sS.jpg  dải 4 khung: bước 10 | 1/3 | khung điển hình | khung cuối
  docs/failures/frames/RUN/index.csv     một dòng cho mỗi episode thất bại

Chỉ số chuyển động giống motion.py của failure_analysis.md mục 2.3: ffmpeg thu nhỏ 90x90 ảnh xám,
trung bình trị tuyệt đối hiệu hai khung liên tiếp.
Khung điển hình: nếu robot "dừng" (trung bình trượt 30 bước của chuyển động < 0,3 từ một bước k đến hết
episode) thì lấy bước k + 15, tức lúc robot vừa dừng hẳn; nếu không dừng thì lấy khung cuối.
"""
import csv, json, re, subprocess, sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
RUNS = REPO / "ffi26/ffi26/results/runs"
THR, WIN = 0.3, 30


def motion(video):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-vf", "scale=90:90",
                          "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True).stdout
    f = np.frombuffer(raw, np.uint8).reshape(-1, 90, 90).astype(np.float32)
    return np.abs(np.diff(f, axis=0)).mean(axis=(1, 2))


def settle_step(dif):
    if len(dif) < WIN:
        return None
    roll = np.convolve(dif, np.ones(WIN) / WIN, mode="valid")  # roll[i] = mean dif[i:i+WIN]
    below = roll < THR
    if not below[-1]:
        return None
    k = len(below) - 1
    while k > 0 and below[k - 1]:
        k -= 1
    return k


def read_frames(video, idxs):
    cap, out, i = cv2.VideoCapture(str(video)), {}, 0
    want = set(idxs)
    while want:
        ok, fr = cap.read()
        if not ok:
            break
        if i in want:
            out[i] = fr
            want.discard(i)
        i += 1
    return out


def main(run):
    outdir = REPO / "docs/failures/frames" / run
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for job in sorted((RUNS / run).glob("*/task*_seed*")):
        t = int(re.search(r"task(\d+)", job.name)[1]); s = int(re.search(r"seed(\d+)", job.name)[1])
        info = job / "eval_info.json"
        if not info.exists():
            continue
        m = json.loads(info.read_text())["per_task"][0]["metrics"]
        for e, (ok, vp) in enumerate(zip(m["successes"], m["video_paths"])):
            if ok:
                continue
            video = job / "videos" / Path(vp).parent.name / Path(vp).name
            dif = motion(video)
            n = len(dif) + 1
            k = settle_step(dif)
            key = min(k + 15, n - 1) if k is not None else n - 1
            idx = [10, n // 3, key, n - 1]  # bước 0–8: vật còn đang rơi vào chỗ khi env khởi tạo
            fr = read_frames(video, idx)
            tiles = []
            for lab, i in zip(["start", "1/3", "key", "end"], idx):
                im = cv2.resize(fr[i], (240, 240))
                cv2.putText(im, f"{lab} {i}", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                tiles.append(im)
            name = f"t{t}_e{e}_s{s}.jpg"
            cv2.imwrite(str(outdir / name), np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 80])
            rows.append(dict(run=run, task=t, seed=s, ep=e, n_frames=n,
                             last100_motion=f"{dif[-100:].mean():.3f}",
                             settle_step="" if k is None else k, key_step=key, image=name))
    with open(outdir / "index.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(run, "failures with video:", len(rows))


if __name__ == "__main__":
    main(sys.argv[1])
