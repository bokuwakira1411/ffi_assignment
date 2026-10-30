"""Sinh bảng của docs/failures/RUN.md từ eval_info.json và docs/failures/labels/RUN.csv.

Cách dùng:  python docs/failures/tools/make_run_report.py RUN
Phần nhận xét viết tay nằm trong docs/failures/notes/RUN.md và được chép nguyên vào đầu file kết quả.
"""
import csv, json, re, sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RUNS = REPO / "ffi26/ffi26/results/runs"
CAUSES = ["F1", "F2", "F3", "F4", "F5", "F6", "F0"]
CAUSE_NAME = {
    "F1": "gắp hỏng", "F2": "làm đổ / kẹp sai tư thế", "F3": "đứng yên / lơ lửng không tiến triển",
    "F4": "đặt không đúng / gần đích", "F5": "nhầm vật / nhầm đích", "F6": "ra khỏi vùng làm việc",
    "F0": "không xác định",
}
EVID = {"T": "timeline", "K": "khung điển hình", "S": "suy luận"}


def success_table(run):
    per = defaultdict(lambda: [0, 0])
    seeds = set()
    for info in sorted((RUNS / run).glob("*/task*_seed*/eval_info.json")):
        t = int(re.search(r"task(\d+)", str(info))[1]); seeds.add(int(re.search(r"seed(\d+)", str(info))[1]))
        pt = json.loads(info.read_text())["per_task"][0]
        per[t][0] += pt["n_success"]; per[t][1] += pt["n_episodes"]
    return per, sorted(seeds)


def main(run):
    per, seeds = success_table(run)
    labels = list(csv.DictReader(open(REPO / f"docs/failures/labels/{run}.csv")))
    notes = REPO / f"docs/failures/notes/{run}.md"
    out = [notes.read_text().rstrip() + "\n" if notes.exists() else f"# {run}\n"]

    out.append("## Tỉ lệ thành công\n")
    out.append(f"Nguồn: `per_task[0].n_success` / `n_episodes` trong `eval_info.json` của từng job; seed: {', '.join(map(str, seeds))}.\n")
    out.append("| Task | Thành công | Tỉ lệ |\n|---|---|---|")
    K = N = 0
    for t in sorted(per):
        k, n = per[t]; K += k; N += n
        out.append(f"| {t} | {k}/{n} | {100 * k / n:.0f}% |")
    out.append(f"| **Tổng** | **{K}/{N}** | **{100 * K / N:.1f}%** |\n")

    out.append("## Nguyên nhân thất bại theo task (chỉ các episode có video)\n")
    out.append("Mỗi episode có đúng một nhãn nguyên nhân chính (định nghĩa ở `docs/failure_overview.md`). "
               "Giai đoạn: 0 = chưa xong bước con thứ nhất; 1 = xong bước con thứ nhất, hỏng ở bước sau; ? = không xác định.\n")
    out.append("| Task | Thất bại có video | " + " | ".join(CAUSES) + " | GĐ 0 | GĐ 1 | GĐ ? |")
    out.append("|---" * (len(CAUSES) + 5) + "|")
    by = defaultdict(list)
    for r in labels: by[int(r["task"])].append(r)
    tot_c, tot_s = Counter(), Counter()
    for t in sorted(by):
        c = Counter(r["cause"] for r in by[t]); s = Counter(r["stage"] for r in by[t])
        tot_c += c; tot_s += s
        out.append(f"| {t} | {len(by[t])} | " + " | ".join(str(c[x] or "") for x in CAUSES)
                   + f" | {s['0'] or ''} | {s['1'] or ''} | {s['?'] or ''} |")
    out.append(f"| **Tổng** | **{len(labels)}** | " + " | ".join(f"**{tot_c[x]}**" for x in CAUSES)
               + f" | **{tot_s['0']}** | **{tot_s['1']}** | **{tot_s['?']}** |\n")
    ev = Counter(r["evidence"] for r in labels)
    out.append("Mức bằng chứng: " + ", ".join(f"{EVID[k]} {v}" for k, v in sorted(ev.items())) + ".\n")

    out.append("## Danh sách từng episode thất bại\n")
    out.append("Ảnh: dải 4 khung (bước 10 | 1/3 episode | khung điển hình | khung cuối), sinh bởi "
               "`docs/failures/tools/extract_frames.py`. Chuyển động = trung bình 100 bước cuối.\n")
    for t in sorted(by):
        out.append(f"### Task {t}\n")
        out.append("| Episode | Seed | Giai đoạn | Nguyên nhân | Bằng chứng | Chuyển động | Ghi chú | Ảnh |\n|---|---|---|---|---|---|---|---|")
        for r in by[t]:
            img = r["image"].replace("../frames/", "frames/")
            out.append(f"| e{r['ep']} | {r['seed']} | {r['stage']} | {r['cause']} {CAUSE_NAME[r['cause']]} | "
                       f"{EVID[r['evidence']]} | {float(r['last100_motion']):.2f} | {r['note']} | [ảnh]({img}) |")
        out.append("")
    (REPO / f"docs/failures/{run}.md").write_text("\n".join(out))
    print(run, "tasks", len(per), "success", K, "/", N, "labelled", len(labels))


if __name__ == "__main__":
    main(sys.argv[1])
