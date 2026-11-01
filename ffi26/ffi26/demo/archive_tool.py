"""Build and verify archives from explicit file manifests. NEVER deletes or moves source files.

    python demo/archive_tool.py build  --archive A1_research_results --dest ~/ffi26_archives
    python demo/archive_tool.py build  --archive A2_visualization     --dest ~/ffi26_archives
    python demo/archive_tool.py verify --zip ~/ffi26_archives/A1_research_results.zip
    python demo/archive_tool.py plan   --archive A3_reproducibility   (sizes only, no zip written)
    python demo/archive_tool.py restore --zip <zip> [--only <path prefix>]   (extracts into the ORIGINAL locations under
                                                                             ~/ffi26, refuses to overwrite existing files)

Paths inside the zip are relative to ~/ffi26 (the repo data root, symlink to ~/Lan_K68AI/temp/ffi26/ffi26), except
entries starting with "repo/", which are relative to ~/Lan_K68AI/temp. A manifest (path, size, sha256) is written next to
the zip as <name>.manifest.csv and stored inside the zip as MANIFEST.csv. Media (mp4/png/jpg/npz/safetensors/pt) are
stored with low compression (zip level 1; npz still shrinks ~2x), text with level 6.
Verification: zipfile.testzip (CRC of every member), manifest == members (names, sizes), counts and total bytes, and a
sha256 comparison of up to 200 members extracted to a temporary directory next to the zip.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import os
import random
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

H = Path.home() / "ffi26"
REPO = Path.home() / "Lan_K68AI/temp"
MEDIA = (".mp4", ".png", ".jpg", ".jpeg", ".npz", ".safetensors", ".pt", ".parquet")

# --- archive definitions (glob patterns relative to ~/ffi26, or repo/<path> relative to the git repo) -------------------
ARCHIVES = {
    "A1_research_results": {
        "include": [
            "results/round2/**/*.csv", "results/round2/**/*.md", "results/round2/**/*.json", "results/round2/**/*.yaml",
            "results/runs/**/eval_info.json", "results/runs/**/cmd.txt", "results/runs/**/eval.log",
            "results/runs/**/status.txt", "results/runs/**/checkpoint_revision.txt", "results/runs/**/*.csv",
            "results/*.json", "results/*.txt", "results/*.yml",
            "improve/outputs/*/iter*/train/*.json", "improve/outputs/*/iter*/train/*.csv",
            "improve/outputs/*/iter*/*.json", "improve/outputs/*/iter*/*.csv",
            "improve/outputs/*/iter_summary.csv", "improve/outputs/*/selection.json", "improve/outputs/*/loop.log",
            "improve/outputs/*/timing.txt", "improve/outputs/*/configs/*.yaml",
            "improve/outputs/*/eval_s*/*/task*/*.meta.json",
            "improve/outputs/r2/eval*/*/task*/*.meta.json", "improve/outputs/eval_baseline/task*/*.meta.json",
            "improve/outputs/r2/round1/*.csv", "improve/outputs/r2/round1/*.json", "improve/outputs/r2/*.log",
            "improve/outputs/r2/*.txt", "improve/outputs/r2/*.csv",
            "improve/outputs/failure_labels*/**/*.csv", "improve/outputs/failure_labels*/**/*.md",
            "improve/outputs/task0_diag/*.csv", "improve/outputs/task0_diag/*.md", "improve/outputs/task0_diag/*.log",
            "improve/configs/**/*", "improve/*.md",
            "repo/docs/**/*.md", "repo/readme.md", "repo/differences.md", "repo/report/main.tex",
            "repo/report/figures/figure_data.json",
        ],
        "exclude": ["*.npz", "*.safetensors", "*.pt", "*.mp4"],
    },
    "A2_visualization": {
        "include": [
            "results/videos/**/*", "results/runs/**/*.mp4", "results/round2/detector_check/**/*",
            "improve/outputs/baseline/frames/*", "repo/docs/failures/frames/**/*", "repo/report/figures/*",
            "repo/report/main.pdf",
        ],
        "exclude": ["results/videos/*/raw/*/*/*/*.npz", "results/videos/raw/**/*.npz"],
    },
    "A3_reproducibility": {   # planned only (sizes); to be written to a disk with enough space after approval
        "include": [
            "improve/outputs/fpo/iter*/decisions/*", "improve/outputs/fpo_v2/iter*/decisions/*",
            "improve/outputs/fpo_v3/iter[1-9]/decisions/*", "improve/outputs/fpo_v4/iter[1-9]/decisions/*",
            "improve/outputs/fpo/iter[013]/train/pretrained_model/*",
            "improve/outputs/fpo_v[234]/iter[012]/train/pretrained_model/*",
            "improve/outputs/r2/train_*/checkpoints/001000/**/*",
            "improve/outputs/r2/train_*/checkpoints/0*/training_state/**/*",
            "improve/outputs/fpo/smoke/**/*", "improve/outputs/fpo_v2_aborted/**/*",
            "improve/outputs/fpo_v2_costtest/**/*",
        ],
        "exclude": [],
    },
}


def expand(name: str) -> list[tuple[str, Path]]:
    spec = ARCHIVES[name]
    out: dict[str, Path] = {}
    for pat in spec["include"]:
        root, rel = (REPO, pat[5:]) if pat.startswith("repo/") else (H, pat)
        prefix = "repo/" if pat.startswith("repo/") else ""
        for p in root.glob(rel):
            if p.is_file() and not p.is_symlink() and not any(part == "__pycache__" for part in p.parts):
                # do not follow directory symlinks (iter0 of v3/v4 links to v2's data)
                rel_p = p.relative_to(root)
                if any((root / anc).is_symlink() for anc in rel_p.parents if str(anc) != "."):
                    continue
                arc = prefix + str(p.relative_to(root))
                if any(fnmatch.fnmatch(arc, e) or fnmatch.fnmatch(p.name, e) for e in spec["exclude"]):
                    continue
                out[arc] = p
    return sorted(out.items())


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def build(name: str, dest: Path):
    files = expand(name)
    dest.mkdir(parents=True, exist_ok=True)
    zp, mp = dest / f"{name}.zip", dest / f"{name}.manifest.csv"
    if zp.exists():
        sys.exit(f"{zp} exists; refusing to overwrite")
    total = sum(p.stat().st_size for _, p in files)
    need = total * 1.05
    free = shutil.disk_usage(dest).free
    print(f"[{name}] {len(files)} files, {total / 1e9:.3f} GB; free at dest {free / 1e9:.1f} GB")
    if need > free * 0.8:
        sys.exit("not enough free space at destination (need < 80% of free)")
    rows = []
    with zipfile.ZipFile(zp, "w", allowZip64=True) as z:
        for arc, p in files:
            lvl = 1 if p.suffix.lower() in MEDIA else 6
            z.write(p, arc, compress_type=zipfile.ZIP_DEFLATED, compresslevel=lvl)
            rows.append({"path": arc, "size": p.stat().st_size, "sha256": sha256(p)})
        with open(mp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["path", "size", "sha256"])
            w.writeheader(); w.writerows(rows)
        z.write(mp, "MANIFEST.csv")
    print(f"[{name}] wrote {zp} ({zp.stat().st_size / 1e9:.3f} GB, ratio {total / max(zp.stat().st_size, 1):.2f}x)")
    verify(zp)


def verify(zp: Path):
    mp = zp.parent / (zp.stem + ".manifest.csv")
    man = {r["path"]: r for r in csv.DictReader(open(mp))}
    with zipfile.ZipFile(zp) as z:
        bad = z.testzip()
        info = {i.filename: i for i in z.infolist() if i.filename != "MANIFEST.csv"}
        ok_names = set(info) == set(man)
        ok_sizes = all(info[k].file_size == int(man[k]["size"]) for k in man if k in info)
        tot_zip = sum(i.file_size for i in info.values())
        tot_man = sum(int(r["size"]) for r in man.values())
        # source still present and unchanged (size)
        def src(k):
            return REPO / k[5:] if k.startswith("repo/") else H / k
        missing_src = [k for k in man if not src(k).exists()]
        changed_src = [k for k in man if src(k).exists() and src(k).stat().st_size != int(man[k]["size"])]
        sample = random.Random(0).sample(sorted(man), min(200, len(man)))
        with tempfile.TemporaryDirectory(dir=zp.parent) as td:
            mism = []
            for k in sample:
                z.extract(k, td)
                if sha256(Path(td) / k) != man[k]["sha256"]:
                    mism.append(k)
    rep = {"zip": str(zp), "crc_ok": bad is None, "names_match_manifest": ok_names, "sizes_match": ok_sizes,
           "n_files": len(man), "bytes_uncompressed": tot_man, "bytes_in_zip_members": tot_zip,
           "zip_bytes": zp.stat().st_size, "ratio": round(tot_man / zp.stat().st_size, 3),
           "sample_extracted": len(sample), "sample_sha_mismatch": len(mism),
           "sources_missing": len(missing_src), "sources_changed_size": len(changed_src)}
    for k, v in rep.items():
        print(f"  {k}: {v}")
    ok = rep["crc_ok"] and ok_names and ok_sizes and not mism and tot_man == tot_zip
    print("  VERIFIED" if ok else "  VERIFICATION FAILED")
    (zp.parent / (zp.stem + ".verify.txt")).write_text("\n".join(f"{k}: {v}" for k, v in rep.items()) +
                                                         ("\nVERIFIED\n" if ok else "\nFAILED\n"))
    return ok


def plan(name: str):
    files = expand(name)
    total = sum(p.stat().st_size for _, p in files)
    groups: dict[str, int] = {}
    for arc, p in files:
        key = "/".join(arc.split("/")[:4])
        groups[key] = groups.get(key, 0) + p.stat().st_size
    for k, v in sorted(groups.items(), key=lambda x: -x[1]):
        print(f"{v / 1e9:8.2f} GB  {k}")
    print(f"[{name}] {len(files)} files, {total / 1e9:.2f} GB (uncompressed)")


def restore(zp: Path, only: str | None):
    with zipfile.ZipFile(zp) as z:
        for n in z.namelist():
            if n == "MANIFEST.csv" or (only and not n.startswith(only)):
                continue
            dst = REPO / n[5:] if n.startswith("repo/") else H / n
            if dst.exists():
                print(f"[skip, exists] {dst}")
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            with z.open(n) as s, open(dst, "wb") as d:
                shutil.copyfileobj(s, d)
            print(f"[restored] {dst}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "verify", "plan", "restore"])
    ap.add_argument("--archive")
    ap.add_argument("--dest", default=str(Path.home() / "ffi26_archives"))
    ap.add_argument("--zip")
    ap.add_argument("--only")
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.archive, Path(a.dest).expanduser())
    elif a.cmd == "verify":
        sys.exit(0 if verify(Path(a.zip).expanduser()) else 1)
    elif a.cmd == "plan":
        plan(a.archive)
    else:
        restore(Path(a.zip).expanduser(), a.only)
