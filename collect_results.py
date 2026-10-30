#!/usr/bin/env python
"""FFI26 - gom ket qua cua run_queue.sh thanh bang success rate + episode_log.csv.

CHUA CHAY THAT: dinh dang file output cua lerobot-eval 0.6.2 chua duoc kiem chung.
Script thu theo thu tu: (1) eval_info.json (tim cac list 'successes'/'video_paths'),
(2) neu khong co thi doc dong pc_success trong eval.log. Neu ca hai that bai, job duoc
ghi 'unparsed' de ban biet phai xem tay. Gui minh `ls -R` mot thu muc job de chinh script.

Usage:
  python collect_results.py --root ~/ffi26/results/runs --tag B_main
Outputs (trong <root>/<tag>/):
  summary_per_job.csv      moi job (task x seed): n_ep, n_success, success_rate, seconds
  summary_per_task.csv     trung binh qua seed, do lech giua cac seed
  episode_log.csv          moi episode 1 dong; 3 cot cuoi (stage_reached, failure_type, notes) dien tay
"""
import argparse
import csv
import json
import re
import statistics
from pathlib import Path

JOB_RE = re.compile(r"task(\d+)_seed(\d+)$")


def find_lists(obj, key, acc):
    """Tim de quy moi list co ten `key` trong JSON."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key and isinstance(v, list):
                acc.append(v)
            else:
                find_lists(v, key, acc)
    elif isinstance(obj, list):
        for v in obj:
            find_lists(v, key, acc)
    return acc


def parse_job(job: Path):
    info = next(iter(sorted(job.rglob("eval_info.json"))), None)
    succ, vids = [], []
    if info:
        data = json.loads(info.read_text())
        for lst in find_lists(data, "successes", []):
            succ += [bool(x) for x in lst]
        for lst in find_lists(data, "video_paths", []):
            vids += [str(x) for x in lst]
    if not succ:
        log = job / "eval.log"
        if log.exists():
            m = re.findall(r"pc_success['\"]?\s*[:=]\s*([0-9.]+)", log.read_text(errors="ignore"))
            if m:
                return {"succ": None, "pc": float(m[-1]), "vids": vids, "source": "eval.log (pc_success only)"}
        return None
    return {"succ": succ, "pc": 100 * sum(succ) / len(succ), "vids": vids, "source": "eval_info.json"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path.home() / "ffi26/results/runs"))
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    base = Path(args.root) / args.tag
    jobs = []
    for pol_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        for job in sorted(pol_dir.glob("task*_seed*")):
            m = JOB_RE.search(job.name)
            if m:
                jobs.append((pol_dir.name, int(m[1]), int(m[2]), job))
    if not jobs:
        raise SystemExit(f"khong thay job nao duoi {base}")

    rows, eps, per_task = [], [], {}
    for pol, t, seed, job in jobs:
        status = (job / "status.txt").read_text().strip() if (job / "status.txt").exists() else "no status"
        secs = re.search(r"seconds=(\d+)", status)
        done = (job / "DONE").exists()
        res = parse_job(job) if done else None
        n = len(res["succ"]) if res and res["succ"] else ""
        k = sum(res["succ"]) if res and res["succ"] else ""
        rows.append([pol, t, seed, done, n, k, f"{res['pc']:.1f}" if res else "unparsed",
                     secs[1] if secs else "", res["source"] if res else status])
        if res:
            per_task.setdefault((pol, t), {})[seed] = res["pc"]
            if res["succ"]:
                for i, s in enumerate(res["succ"]):
                    v = res["vids"][i] if i < len(res["vids"]) else ""
                    eps.append([args.tag, pol, t, seed, i, int(s), v, "", "", "", ""])

    def w(name, header, data):
        with open(base / name, "w", newline="") as f:
            cw = csv.writer(f)
            cw.writerow(header)
            cw.writerows(data)
        print("wrote", base / name)

    w("summary_per_job.csv", ["policy", "task", "seed", "done", "n_ep", "n_success", "success_pct", "seconds", "source"], rows)
    pt = []
    for (pol, t), d in sorted(per_task.items()):
        v = list(d.values())
        pt.append([pol, t, len(v), f"{statistics.mean(v):.1f}",
                   f"{statistics.stdev(v):.1f}" if len(v) > 1 else ""])
    w("summary_per_task.csv", ["policy", "task", "n_seeds", "mean_success_pct", "std_between_seeds"], pt)
    w("episode_log.csv", ["tag", "policy", "task", "seed", "episode", "success", "video",
                          "steps", "stage_reached", "failure_type", "notes"], eps)
    for pol in sorted({p for p, _ in per_task}):
        means = [statistics.mean(d.values()) for (p, _), d in per_task.items() if p == pol]
        print(f"{pol}: trung binh {len(means)} task = {statistics.mean(means):.1f}% (khong trong so)")
    bad = [r for r in rows if r[6] == "unparsed" or not r[3]]
    if bad:
        print(f"CHU Y: {len(bad)} job chua xong hoac khong parse duoc (xem summary_per_job.csv)")


if __name__ == "__main__":
    main()
