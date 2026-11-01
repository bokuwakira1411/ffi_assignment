"""Write <output_dir>/index.md for the video replays (scripts/phase3_eval_video.py).

Reads the config (episode list + categories), episodes_{arm}.csv (selected episodes: outcome, labels,
events) and replay_check_{arm}.csv (every replayed episode vs the original run). The note column is
generated from the automatic labels (src/failure_labels.py) and events (src/reward_r2.py), not from
watching the video.

    python scripts/videos_index.py --config ~/ffi26/results/videos/configs/videos_v2.yaml
"""

import argparse
from pathlib import Path

import pandas as pd
import yaml

LABEL = {"A": "không gắp được vật mục tiêu", "B": "kẹt giữa hai subtask", "C": "chạm/gắp nhầm vật khác",
         "D": "làm hỏng tiến độ đã đạt", "E": "đứng yên (stall)", "O": "thất bại, không rõ cơ chế", "S": "thành công"}
CAT = {"oF_tS": "official thất bại, θ3 thành công", "oS_tF": "official thành công, θ3 thất bại",
       "FF": "cả hai thất bại", "SS": "cả hai thành công"}


def note(r) -> str:
    ev = r.events if isinstance(r.events, str) and r.events else "không có"
    head = f"thành công ở bước {r.length}" if r.success else f"thất bại, nhãn {r.primary}: {LABEL.get(r.primary, '')}"
    return f"{head}; subtask {r.subtask_completion:.1f}; sự kiện: {ev}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    out = Path(cfg["output_dir"]).expanduser()
    arms = [a for a in cfg["arms"] if (out / f"episodes_{a['name']}.csv").exists()]
    rd = lambda p: pd.read_csv(p, dtype=str, keep_default_na=False)  # noqa: E731 (bools stay "True"/"False")
    sel = pd.concat([rd(out / f"episodes_{a['name']}.csv") for a in arms])
    chk = pd.concat([rd(out / f"replay_check_{a['name']}.csv") for a in arms])
    for c in ("seed", "task", "init"):
        sel[c], chk[c] = sel[c].astype(int), chk[c].astype(int)
    sel["success"] = sel.success == "True"
    sel["subtask_completion"] = sel.subtask_completion.astype(float)
    cat = {(e["seed"], e["task"], e["init"]): e["cat"] for e in cfg["episodes"]}
    bad = chk[(chk.match != "") & (chk.match != "True")]

    L = ["# Video replay phần Improve", "",
         f"Config: `{args.config}`. Script: `improve/scripts/phase3_eval_video.py`. Mỗi video: agent view | wrist view, "
         "20 fps, header gồm nhánh, task, init state, seed eval (env seed = seed + init), bước, Φ, sự kiện reward_r2 "
         "(drop/slip/collision/timeout, hiện màu đỏ 2 s từ bước xảy ra).", "",
         "## Kiểm tra tính tất định", ""]
    for a in arms:
        c = chk[chk.arm == a["name"]]
        n_o = int((c.match != "").sum())
        if n_o == 0:
            L.append(f"- **{a['name']}**: {len(c)} episode chạy lại, không có bản gốc để so (episode mới).")
        else:
            nb = int(((c.match != "") & (c.match != "True")).sum())
            L.append(f"- **{a['name']}**: {len(c)} episode chạy lại, {n_o} có bản gốc; "
                     f"{'**' + str(nb) + ' KHÁC bản gốc**' if nb else 'tất cả khớp (success và length)'}.")
    if len(bad):
        L += ["", "⚠ Các episode khác bản gốc (success/length gốc → replay; `*` = episode có video):", "",
              "| nhánh | seed | task | init | gốc | replay |", "|---|---|---|---|---|---|"]
        for r in bad.itertuples():
            L.append(f"| {r.arm} | {r.seed} | {r.task} | {r.init}{' *' if r.selected == 'True' else ''} | "
                     f"{r.success_orig}/{r.length_orig} | {r.success_replay}/{r.length_replay} |")
    L += ["", "## Danh sách video", "",
          "Ghi chú tự động từ nhãn `src/failure_labels.py` và sự kiện `src/reward_r2.py` (sự kiện@bước), chưa xem video.", "",
          "| loại | seed | task | init | nhánh | file | kết quả | khớp bản gốc | ghi chú |",
          "|---|---|---|---|---|---|---|---|---|"]
    sel = sel.assign(order=[list(cat).index((r.seed, r.task, r.init)) for r in sel.itertuples()])
    for r in sel.sort_values(["order", "arm"], key=lambda s: s if s.name == "order" else
                             s.map({a["name"]: i for i, a in enumerate(cfg["arms"])})).itertuples():
        m = chk[(chk.arm == r.arm) & (chk.seed == r.seed) & (chk.task == r.task) & (chk.init == r.init)].match.iloc[0]
        ms = "không có bản gốc" if m == "" else ("có" if m == "True" else "**KHÔNG**")
        L.append(f"| {CAT[cat[(r.seed, r.task, r.init)]]} | {r.seed} | {r.task} | {r.init} | {r.arm} | [{r.file}]({r.file}) | "
                 f"{'thành công' if r.success else 'thất bại'} | {ms} | {note(r)} |")
    (out / "index.md").write_text("\n".join(L) + "\n")
    print(f"[index] {out / 'index.md'}: {len(sel)} videos, {len(bad)} mismatches")


if __name__ == "__main__":
    main()
