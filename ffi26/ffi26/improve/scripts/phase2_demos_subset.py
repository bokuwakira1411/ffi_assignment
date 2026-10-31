"""Write the target-task demos of lerobot/libero into a new LeRobot dataset, with the task
string suffixed " Advantage: positive".

Why not dataset_tools.delete_episodes: the hub copy of lerobot/libero (rev a1aaacb7) has no
'meta/episodes/chunk_index' column in meta/episodes, which delete_episodes needs (KeyError in
dataset_tools.py:99). So frames are decoded with PyAV from the episode's segment
[from_timestamp, to_timestamp) of the concatenated video file and re-encoded. Re-encoding adds
one more AV1 compression generation (known limitation).

    python scripts/phase2_demos_subset.py --config configs/phase2_r1.yaml
"""

import argparse
import glob
from pathlib import Path

import av
import numpy as np
import pandas as pd
import yaml
from lerobot.datasets.lerobot_dataset import LeRobotDataset

POS = " Advantage: positive"
CAMS = ["observation.images.image", "observation.images.image2"]
FEATURES = {
    **{c: {"dtype": "video", "shape": [256, 256, 3], "names": ["height", "width", "channel"]} for c in CAMS},
    "observation.state": {"dtype": "float32", "shape": [8], "names": ["state"]},
    "action": {"dtype": "float32", "shape": [7], "names": ["actions"]},
}


def decode_segment(path: Path, t0: float, t1: float, n: int, fps: float) -> np.ndarray:
    """Decode the n frames with timestamps in [t0, t1) (+ half-frame tolerance)."""
    tol = 0.5 / fps
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        c.seek(int(max(t0 - 1.0, 0) / s.time_base), stream=s, backward=True, any_frame=False)
        out = []
        for f in c.decode(s):
            t = float(f.pts * s.time_base)
            if t < t0 - tol:
                continue
            if t >= t1 - tol or len(out) == n:
                break
            out.append(f.to_ndarray(format="rgb24"))
    arr = np.stack(out)
    assert len(arr) == n, f"{path} [{t0},{t1}) decoded {len(arr)} frames, expected {n}"
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    b = cfg["build"]
    src = Path(b["demo_root"]).expanduser()
    out = Path(b["demos_out"]).expanduser()
    # round 2: optional keys (all libero_10 tasks, no suffix); defaults = round-1 behaviour
    texts = b.get("demos_task_texts", b["target_task_texts"])
    suffix = b.get("demos_suffix", POS)
    repo_id = b.get("demos_repo_id", "local/ffi26_demos_targets")
    info = pd.read_json(src / "meta/info.json", typ="series")
    fps = float(info["fps"])
    tasks = pd.read_parquet(src / "meta/tasks.parquet")
    keep = {int(tasks.loc[t, "task_index"]): t for t in texts}

    data = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(src / "data/*/*.parquet")))])
    data = data[data.task_index.isin(keep)]
    eps = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(src / "meta/episodes/*/*.parquet")))])
    eps = eps[eps.episode_index.isin(data.episode_index.unique())].sort_values("episode_index")
    print(f"keeping {len(eps)} demo episodes, {len(data)} frames", flush=True)

    ds = LeRobotDataset.create(repo_id=repo_id, fps=int(fps), features=FEATURES,
                               root=out, robot_type="panda", use_videos=True)
    for _, e in eps.iterrows():
        ei = int(e.episode_index)
        rows = data[data.episode_index == ei].sort_values("frame_index")
        n = len(rows)
        assert n == int(e.length)
        frames = {}
        for c in CAMS:
            vp = src / info["video_path"].format(
                video_key=c, chunk_index=int(e[f"videos/{c}/chunk_index"]), file_index=int(e[f"videos/{c}/file_index"]))
            frames[c] = decode_segment(vp, float(e[f"videos/{c}/from_timestamp"]), float(e[f"videos/{c}/to_timestamp"]), n, fps)
        task = keep[int(rows.task_index.iloc[0])] + suffix
        for k, (_, r) in enumerate(rows.iterrows()):
            ds.add_frame({
                CAMS[0]: frames[CAMS[0]][k], CAMS[1]: frames[CAMS[1]][k],
                "observation.state": np.asarray(r["observation.state"], dtype=np.float32),
                "action": np.asarray(r["action"], dtype=np.float32),
                "task": task,
            })
        ds.save_episode()
        print(f"[demo] src episode {ei} -> {ds.meta.total_episodes - 1} ({n} frames) {task!r}", flush=True)
    ds.finalize()
    print(f"[done] {ds.meta.total_episodes} episodes, {ds.meta.total_frames} frames")


if __name__ == "__main__":
    main()
