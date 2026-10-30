#!/usr/bin/env python
"""FFI26 Day-1 environment check.

Verifies, in order:
  1. PyTorch sees the GPU and can actually run kernels (catches the Blackwell
     "no kernel image is available" error that older wheels give on RTX 50xx).
  2. MuJoCo can render offscreen with the selected backend (EGL by default).
  3. lerobot, SmolVLA policy class and LIBERO can be imported.
  4. lerobot-eval / lerobot-train are on PATH.

Writes a JSON report you can drop into the "setup" section of the report.
Exit code is non-zero if a hard requirement fails.
"""
import argparse
import json
import os
import platform
import shutil
import sys
import traceback

os.environ.setdefault("MUJOCO_GL", "egl")  # must be set before importing mujoco

report = {"python": sys.version.split()[0], "platform": platform.platform(),
          "MUJOCO_GL": os.environ["MUJOCO_GL"], "checks": {}}
hard_fail = False


def record(name, ok, detail="", hard=True):
    global hard_fail
    report["checks"][name] = {"ok": bool(ok), "detail": str(detail)}
    mark = "OK  " if ok else ("FAIL" if hard else "WARN")
    print(f"[{mark}] {name}: {detail}")
    if not ok and hard:
        hard_fail = True


# ---- 1. PyTorch + GPU -------------------------------------------------------
try:
    import torch

    report["torch"] = torch.__version__
    report["torch_cuda"] = torch.version.cuda
    avail = torch.cuda.is_available()
    record("cuda_available", avail, f"torch {torch.__version__}, cuda {torch.version.cuda}")
    if avail:
        props = torch.cuda.get_device_properties(0)
        cap = f"sm_{props.major}{props.minor}"
        report["gpu"] = {"name": props.name, "capability": cap,
                         "memory_gb": round(props.total_memory / 1e9, 1),
                         "arch_list": torch.cuda.get_arch_list()}
        in_list = cap in torch.cuda.get_arch_list() or any(
            a.startswith(f"compute_{props.major}{props.minor}") for a in torch.cuda.get_arch_list())
        record("arch_supported", in_list,
               f"{props.name} is {cap}; wheel arch list = {torch.cuda.get_arch_list()}", hard=False)
        try:
            x = torch.randn(512, 512, device="cuda")
            y = (x @ x).sum().item()
            xb = x.to(torch.bfloat16)
            yb = (xb @ xb).float().sum().item()
            record("gpu_kernels_fp32_bf16", True, f"matmul ok ({y:.1f}, {yb:.1f})")
        except Exception as e:  # noqa: BLE001
            record("gpu_kernels_fp32_bf16", False,
                   f"{e}  -> wrong PyTorch build for this GPU? use the cu128 (or newer) wheel")
except Exception as e:  # noqa: BLE001
    record("import_torch", False, e)

# ---- 2. Offscreen rendering -------------------------------------------------
try:
    import mujoco

    xml = ("<mujoco><worldbody><light pos='0 0 3'/>"
           "<geom type='sphere' size='0.1'/></worldbody></mujoco>")
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, 64, 64)
    mujoco.mj_forward(model, data)
    renderer.update_scene(data)
    img = renderer.render()
    record("mujoco_offscreen_render", img.shape == (64, 64, 3),
           f"backend={os.environ['MUJOCO_GL']} frame shape {img.shape}")
except Exception as e:  # noqa: BLE001
    record("mujoco_offscreen_render", False,
           f"{e}  -> try MUJOCO_GL=osmesa (slow) or ask admin for libEGL")

# ---- 3. Library imports -----------------------------------------------------
try:
    import lerobot

    record("import_lerobot", True, getattr(lerobot, "__version__", "version attr missing"),
           hard=True)
    report["lerobot"] = getattr(lerobot, "__version__", None)
except Exception as e:  # noqa: BLE001
    record("import_lerobot", False, e)

try:
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: F401

    record("import_smolvla_policy", True, "SmolVLAPolicy importable")
except Exception as e:  # noqa: BLE001
    record("import_smolvla_policy", False, e)

try:
    import libero.libero  # noqa: F401

    record("import_libero", True, "libero importable")
except Exception as e:  # noqa: BLE001
    traceback.print_exc()
    record("import_libero", False,
           f"{type(e).__name__}: {e}. LIBERO may ask interactive questions on its first import "
           "(e.g. about a dataset path). Run:  python -c 'import libero.libero'  "
           "in a normal terminal, answer N, then rerun this script.")

# ---- 4. CLI tools -----------------------------------------------------------
for tool in ("lerobot-eval", "lerobot-train"):
    path = shutil.which(tool)
    record(f"cli_{tool}", path is not None, path or "not on PATH (is the env activated?)")

# ---- write report -----------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--out", default="env_report.json")
args = parser.parse_args()
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
with open(args.out, "w") as f:
    json.dump(report, f, indent=2)
print(f"\nreport written to {args.out}")
sys.exit(1 if hard_fail else 0)