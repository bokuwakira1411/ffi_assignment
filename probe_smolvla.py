#!/usr/bin/env python
"""FFI26 - SmolVLA probe: what goes in, what is inside, what the loss is, how actions are produced.

Writes a JSON report (for the 'Reproduce: explain the pipeline' section) and prints tables.
Nothing is trained: weights are only read (a checksum before/after proves it).

Sections (each fails independently; a failure is recorded in the report, the rest still run):
  0  versions / device
  1  load policy + key config values
  2  parameters per block (exact module names, no guessing) + ModuleList lengths (layer counts)
  3  data: read a few samples of the dataset, run the checkpoint's own pre-processor
  4  pipeline trace: tensor shapes after each stage (images, tokens, prefix/suffix lengths)
  5  loss: one forward pass, no update, at fixed flow times, vs a 'predict zero velocity' baseline
  6  inference: denoise steps per action chunk, latency, peak VRAM

Written against the LeRobot `main` source read on 30/10/2026 (policies/smolvla/modeling_smolvla.py,
policies/common/flow_matching.py, scripts/lerobot_train.py). It was NOT run here (no GPU / no Hub
access in the authoring sandbox). Your installed commit (see results/versions.txt) may differ, so
read each section's error message if one fails.

Usage:
  conda activate lerobot
  python probe_smolvla.py --dataset-revision <commit-or-tag-of-lerobot/libero>
  python probe_smolvla.py --skip-data          # only sections 0-2
"""
import argparse
import contextlib
import json
import os
import statistics
import time
import traceback

import torch
import torch.nn as nn

REPORT = {"sections": {}}


@contextlib.contextmanager
def section(name):
    print(f"\n=== {name} ===")
    out = {}
    REPORT["sections"][name] = out
    try:
        yield out
        out["ok"] = True
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        out["ok"] = False
        out["error"] = f"{type(e).__name__}: {e}"
        print(f"[FAIL] {name}: {out['error']}")


def shapes(d):
    return {k: list(v.shape) for k, v in d.items() if torch.is_tensor(v)}


def group_params(named_params, depth):
    groups = {}
    for name, p in named_params:
        key = ".".join(name.split(".")[:depth])
        g = groups.setdefault(key, {"params": 0, "trainable": 0})
        g["params"] += p.numel()
        g["trainable"] += p.numel() if p.requires_grad else 0
    return dict(sorted(groups.items(), key=lambda kv: -kv[1]["params"]))


def print_groups(groups, total):
    for k, v in groups.items():
        print(f"  {v['params'] / 1e6:9.2f} M  ({100 * v['params'] / total:5.1f}%)  trainable {v['trainable'] / 1e6:8.2f} M  {k}")


def checksum(model):
    with torch.no_grad():
        return float(sum(p.double().sum() for p in model.parameters()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="lerobot/smolvla_libero")
    ap.add_argument("--dataset", default="lerobot/libero")
    ap.add_argument("--dataset-revision", default=None,
                    help="PIN THIS (commit/tag). The checkpoint's train_config has revision=null.")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--n-batches", type=int, default=3)
    ap.add_argument("--skip-data", action="store_true")
    ap.add_argument("--out", default=os.path.expanduser("~/ffi26/results/probe_report.json"))
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    REPORT["args"] = vars(args)

    # ---- 0 ------------------------------------------------------------------------------------
    with section("0_versions") as out:
        out["torch"] = torch.__version__
        out["device"] = device
        if device == "cuda":
            out["gpu"] = torch.cuda.get_device_name(0)
        try:
            import lerobot
            out["lerobot"] = getattr(lerobot, "__version__", "unknown")
        except Exception as e:  # noqa: BLE001
            out["lerobot"] = f"import failed: {e}"
        try:
            from huggingface_hub import HfApi
            out["policy_commit_sha"] = HfApi().model_info(args.policy).sha
        except Exception as e:  # noqa: BLE001
            out["policy_commit_sha"] = f"unavailable: {e}"
        if args.dataset_revision is None:
            out["WARNING"] = "dataset revision not pinned; record which revision you actually used"
        print(json.dumps(out, indent=2))

    # ---- 1 ------------------------------------------------------------------------------------
    policy = None
    with section("1_load_policy_and_config") as out:
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        policy = SmolVLAPolicy.from_pretrained(args.policy).to(device).eval()
        cfg = policy.config
        keys = ["chunk_size", "n_action_steps", "num_steps", "max_state_dim", "max_action_dim",
                "resize_imgs_with_padding", "tokenizer_max_length", "vlm_model_name", "num_vlm_layers",
                "num_expert_layers", "expert_width_multiplier", "self_attn_every_n_layers",
                "attention_mode", "freeze_vision_encoder", "train_expert_only", "train_state_proj",
                "use_cache", "min_period", "max_period", "empty_cameras"]
        out["config"] = {k: (getattr(cfg, k, "MISSING") if not isinstance(getattr(cfg, k, None), (tuple, set))
                             else list(getattr(cfg, k))) for k in keys}
        out["input_features"] = {k: {"type": str(v.type), "shape": list(v.shape)} for k, v in cfg.input_features.items()}
        out["output_features"] = {k: {"type": str(v.type), "shape": list(v.shape)} for k, v in cfg.output_features.items()}
        out["normalization_mapping"] = {str(k): str(v) for k, v in cfg.normalization_mapping.items()}
        print(json.dumps(out, indent=2, default=str))

    if policy is None:
        return finish(args)

    # ---- 2 ------------------------------------------------------------------------------------
    with section("2_parameters") as out:
        named = list(policy.model.named_parameters())
        total = sum(p.numel() for _, p in named)
        trainable = sum(p.numel() for _, p in named if p.requires_grad)
        out["total_params"], out["trainable_params"] = total, trainable
        print(f"total {total / 1e6:.1f} M, trainable (as loaded) {trainable / 1e6:.1f} M")
        out["by_top_level_child_of_model"] = group_params(named, 1)
        out["by_depth_4"] = group_params(named, 4)
        print("-- top-level children of policy.model (exact):")
        print_groups(out["by_top_level_child_of_model"], total)
        print("-- depth 4 (find the vision encoder / text layers / expert layers here):")
        print_groups(out["by_depth_4"], total)
        out["modulelist_lengths"] = {n: len(m) for n, m in policy.model.named_modules()
                                     if isinstance(m, nn.ModuleList) and n}
        print("-- ModuleList lengths (layer counts):")
        for n, k in out["modulelist_lengths"].items():
            print(f"  {k:4d}  {n}")

    if args.skip_data:
        return finish(args)

    # ---- 3 ------------------------------------------------------------------------------------
    from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

    batches = []
    with section("3_data_and_preprocess") as out:
        try:
            from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
        except ImportError:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
        try:
            from lerobot.policies import make_pre_post_processors
        except ImportError:
            from lerobot.policies.factory import make_pre_post_processors
        import numpy as np
        from torch.utils.data import default_collate

        cfg = policy.config
        meta = LeRobotDatasetMetadata(args.dataset, revision=args.dataset_revision)
        idx = list(getattr(cfg, "action_delta_indices", None) or range(cfg.chunk_size))
        ds = LeRobotDataset(args.dataset, revision=args.dataset_revision,
                            delta_timestamps={"action": [i / meta.fps for i in idx]})
        out["dataset"] = {"repo_id": args.dataset, "revision": args.dataset_revision, "fps": meta.fps,
                          "frames": ds.num_frames, "episodes": ds.num_episodes,
                          "features": list(meta.features.keys()), "camera_keys": list(meta.camera_keys)}
        n = args.batch_size * args.n_batches
        picks = np.linspace(0, len(ds) - 1, n).astype(int)  # spread over the dataset
        items = [ds[int(i)] for i in picks]
        raw = [default_collate(items[j * args.batch_size:(j + 1) * args.batch_size]) for j in range(args.n_batches)]
        out["raw_batch_shapes"] = shapes(raw[0])
        out["tasks_sampled"] = sorted({str(t) for it in items for t in [it.get("task", "?")]})[:12]

        # Rename map copied from the published train_config.json of lerobot/smolvla_libero.
        rename = {"observation.images.image": "observation.images.camera1",
                  "observation.images.image2": "observation.images.camera2"}
        pre = None
        attempts = [{"device_processor": {"device": device},
                     "rename_observations_processor": {"rename_map": rename}},
                    {"device_processor": {"device": device}}, None]
        for ov in attempts:
            try:
                kw = dict(policy_cfg=cfg, pretrained_path=args.policy)
                if ov:
                    kw["preprocessor_overrides"] = ov
                pre, _post = make_pre_post_processors(**kw)
                out["processor_overrides_used"] = list(ov) if ov else []
                break
            except Exception as e:  # noqa: BLE001
                out.setdefault("processor_attempt_errors", []).append(f"{type(e).__name__}: {e}")
        if pre is None:
            raise RuntimeError(f"could not build the pre-processor: {out.get('processor_attempt_errors')}")
        for r in raw:
            b = {}
            for k, v in r.items():
                if torch.is_tensor(v) and v.dtype == torch.uint8:
                    v = v.to(torch.float32) / 255.0  # same as lerobot_train._preprocess_dataset_batch
                b[rename.get(k, k)] = v
            batches.append(pre(b))
        out["processed_batch_shapes"] = shapes(batches[0])
        print(json.dumps(out, indent=2, default=str))

    if not batches:
        return finish(args)
    batch = batches[0]

    # ---- 4 ------------------------------------------------------------------------------------
    with section("4_pipeline_trace") as out, torch.no_grad():
        images, img_masks = policy.prepare_images(dict(batch))
        state = policy.prepare_state(batch)
        out["images_after_prepare (B,3,H,W, range [-1,1])"] = [list(i.shape) for i in images]
        out["state_after_pad"] = list(state.shape)
        emb = policy.model.vlm_with_expert.embed_image(images[0])
        out["tokens_per_image"], out["image_embed_dim"] = emb.shape[1], emb.shape[2]
        lang_tokens, lang_masks = batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK]
        out["language_tokens"] = list(lang_tokens.shape)
        pe, _pm, _pa = policy.model.embed_prefix(images, img_masks, lang_tokens, lang_masks, state=state)
        out["prefix_len (images + language + 1 state token)"] = pe.shape[1]
        actions = policy.prepare_action(batch)
        out["action_chunk_target (B, chunk, max_action_dim)"] = list(actions.shape)
        t = torch.full((actions.shape[0],), 0.5, device=actions.device)
        se, _sm, _sa = policy.model.embed_suffix(actions, t)
        out["suffix_len (one token per chunk step)"] = se.shape[1]
        print(json.dumps(out, indent=2))

    # ---- 5 ------------------------------------------------------------------------------------
    with section("5_loss_one_forward_no_update") as out, torch.no_grad():
        policy.eval()
        before = checksum(policy)
        adim = policy.config.action_feature.shape[0]
        g = torch.Generator(device=device).manual_seed(0)
        by_t, base = {}, []
        for t in (0.1, 0.3, 0.5, 0.7, 0.9):
            vals = []
            for b in batches:
                a = policy.prepare_action(b)
                noise = torch.randn(a.shape, generator=g, device=a.device, dtype=a.dtype)
                tt = torch.full((a.shape[0],), t, device=a.device)
                loss, _ = policy.forward(dict(b), noise=noise, time=tt)
                vals.append(float(loss))
                if t == 0.5:
                    pad = b.get("action_is_pad")
                    u = (noise - a)[:, :, :adim]
                    if pad is not None:
                        u = u * (~pad).unsqueeze(-1)
                    base.append(float((u ** 2).mean()))
            by_t[str(t)] = statistics.mean(vals)
        torch.manual_seed(0)
        out["loss_by_flow_time (t=1 is pure noise)"] = by_t
        out["loss_default_time_sampling"] = statistics.mean(float(policy.forward(dict(b))[0]) for b in batches)
        out["zero_velocity_baseline_mse (approx, padding mask applied if present)"] = statistics.mean(base)
        out["weights_unchanged"] = abs(checksum(policy) - before) == 0.0
        out["note"] = ("Loss = MSE between predicted velocity and (noise - action). Convention NOISE_AT_ONE: "
                       "x_t = t*noise + (1-t)*action. Batches: %d x %d samples spread over the dataset."
                       % (len(batches), args.batch_size))
        print(json.dumps(out, indent=2))

    # ---- 6 ------------------------------------------------------------------------------------
    with section("6_inference_trace") as out:
        sample = {k: (v[:1] if torch.is_tensor(v) else v) for k, v in batch.items()}
        calls = {"n": 0}
        orig = policy.model.denoise_step

        def counted(*a, **k):
            calls["n"] += 1
            return orig(*a, **k)

        policy.model.denoise_step = counted
        try:
            for _ in range(2):  # warmup
                policy.reset()
                policy.predict_action_chunk(dict(sample))
            calls["n"] = 0
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            times = []
            for _ in range(5):
                policy.reset()
                if device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                chunk = policy.predict_action_chunk(dict(sample))
                if device == "cuda":
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
        finally:
            del policy.model.denoise_step  # remove the instance-level wrapper
        out["action_chunk_shape (B, chunk, action_dim)"] = list(chunk.shape)
        out["denoise_calls_per_chunk"] = calls["n"] // 5
        out["median_latency_ms_per_chunk (batch 1)"] = round(1000 * statistics.median(times), 1)
        if device == "cuda":
            out["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        n_as = policy.config.n_action_steps
        out["policy_forward_calls_per_100_env_steps"] = round(100 / n_as, 1)
        out["note"] = (f"Euler integration from noise (t=1) to action (t=0) with num_steps={policy.config.num_steps}; "
                       f"a new chunk is generated every n_action_steps={n_as} env steps.")
        print(json.dumps(out, indent=2))

    finish(args)


def finish(args):
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(REPORT, f, indent=2, default=str)
    bad = [k for k, v in REPORT["sections"].items() if not v.get("ok")]
    print(f"\nreport written to {args.out}" + (f"  | FAILED sections: {bad}" if bad else "  | all sections ok"))


if __name__ == "__main__":
    main()
