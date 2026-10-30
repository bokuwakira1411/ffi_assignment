"""Ghép các dải khung của một run thành trang xem, mỗi trang tối đa 6 episode.
Cách dùng: python docs/failures/tools/sheet.py RUN OUTDIR [TASK ...]"""
import csv, sys
from pathlib import Path
import cv2, numpy as np
REPO = Path(__file__).resolve().parents[3]
run, out = sys.argv[1], Path(sys.argv[2]); tasks = set(sys.argv[3:])
d = REPO / "docs/failures/frames" / run
rows = [r for r in csv.DictReader(open(d / "index.csv")) if not tasks or r["task"] in tasks]
out.mkdir(parents=True, exist_ok=True)
by = {}
for r in rows: by.setdefault(r["task"], []).append(r)
for t, rs in by.items():
    for p in range(0, len(rs), 6):
        ims = []
        for r in rs[p:p + 6]:
            im = cv2.imread(str(d / r["image"]))
            cv2.putText(im, r["image"][:-4] + f" m={r['last100_motion']}", (250, 232), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            ims.append(im)
        cv2.imwrite(str(out / f"{run}_t{t}_p{p // 6}.jpg"), np.vstack(ims), [cv2.IMWRITE_JPEG_QUALITY, 75])
