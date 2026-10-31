"""Download lerobot/libero (training data of lerobot/smolvla_libero, per its train_config.json) into data/."""
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
dst = Path.home() / "ffi26/improve/data/lerobot_libero"
sha = HfApi().dataset_info("lerobot/libero").sha
print("revision", sha, flush=True)
snapshot_download("lerobot/libero", repo_type="dataset", local_dir=str(dst), revision=sha)
(dst / "REVISION.txt").write_text(sha + "\n")
print("DONE", dst)
