# SmolVLA on LIBERO — Reproduce, Analyse, Study, Improve

Checkpoint `lerobot/smolvla_libero` on LeRobot, 1 GPU with 16 GB. Main setting: libero_10, `n_action_steps = 10`.

![Agent view: official vs FPO v4 on the same episodes](docs/images/official_vs_v4.png)

*Official vs FPO v4 (ours) on the same episodes, agent camera. These two episodes, where v4 succeeds and official fails, were
chosen for illustration (v4 also loses some episodes to official); statistics over all 300 episodes are in [Results](#results).
Videos: [Videos](#videos). Figure: `docs/images/make_readme_fig.py`.*

## Videos

> **Result videos — official fails, FPO v4 succeeds (same initial state and seed):**
> https://github.com/bokuwakira1411/ffi_assignment/tree/main/ffi26/ffi26/results/videos/4_official_fail_v4_success

More side-by-side videos of the methods: `ffi26/ffi26/results/videos/` (e.g. `3_v4_paired/`).

## Results

### Reproduce

| Suite | `n_action_steps` | Episodes | Success |
|---|---|---|---|
| libero_10 | 10 | 3 seeds × 10 tasks × 50 init | **72.8%** (paper: 71%, not verified) |
| libero_10 | 1 | 3 seeds × 10 tasks × 10 init | 52.7% |
| libero_spatial | 1 | 3 seeds × 10 tasks × 10 init | 79.0% |

Source: `docs/reproduce.md`.

### Improve: RL on task 0 and task 8 of libero_10

Protocol:

- All methods run on the same initial states and seeds, so they can be compared episode by episode (paired).
- Trained tasks (task 0, task 8): 50 init × 3 seeds (3000 / 4000 / 5000) per task, 300 episodes in total.
- Forgetting (the other 8 tasks): 20 init per task, seed 1000, 160 episodes in total.

| Method | Change from previous version | Task 0 | Task 8 | Task 0 + 8 | Other 8 tasks |
|---|---|---|---|---|---|
| official | original checkpoint | 40.7% | 40.0% | 40.3% | 79.4% |
| FPO v1 | PPO-clip with ratio computed from the flow-matching loss, on-policy, 4 rounds | 41.3% | 24.0% | 32.7% | 69.4% |
| FPO v2 | fp32 expert, batch 128, BC loss on successful rollouts | 33.3% | 50.7% | 42.0% | 67.5% |
| FPO v3 | + demo replay against forgetting, per-task advantage normalization | 30.7% | 58.7% | 44.7% | 71.9% |
| **FPO v4** | + per-segment advantage weighting, no mean subtraction | 40.0% | 53.3% | **46.7%** | **78.1%** |

- **v4 is the only method significantly better than official** (paired McNemar, p = 0.025). The gain comes from task 8
  (p = 0.002); task 0 is on par with official.
- v4 shows almost no forgetting: 78.1% vs 79.4% (p = 0.83). v1, v2 and v3 are all clearly worse than official on the other
  8 tasks (p = 0.014 / 0.003 / 0.017).
- v4 beats v3 overall, but not significantly (p = 0.55). v4 brings task 0 back to the official level (v3: 30.7%).
- v1 reached 57% in the screening round (table below), but this did not hold up on seed 2000. In the table above, v1 is
  worse than official (p = 0.03).

Source: `ffi26/ffi26/results/round2/fpo_v4/pooled.md` (official, v2, v3, v4), `fpo_v2/seed*/episodes.csv` (v1).

#### Screening round: A1, A2 (offline), FPO v1

Before FPO, two offline approaches were tried. Both use rollouts of the original checkpoint, score them with a progress-based
reward, and compute advantages within groups of rollouts that share the same initial state.

- **A1** (RECAP-style): label each segment with a positive / negative advantage, then fine-tune conditioned on the label.
  At test time the positive label is used. Two ways of injecting the label: a prompt suffix, or a learned embedding fed to
  the action expert.
- **A2** (AWR): advantage-weighted flow-matching fine-tuning, no labels.

The protocol differs from the table above, so the numbers are not directly comparable: seed 1000, 50 init per task,
bf16 expert; other 8 tasks × 20 init.

| Method | Task 0 + 8 | Other 8 tasks |
|---|---|---|
| official | 44/100 | 126/160 |
| A1, prompt suffix, positive label | 31/100 | 88/160 |
| A1, embedding, positive label | 38/100 | 118/160 |
| A2 AWR, 2000 steps | 43/100 | 128/160 |
| A2 AWR, 6000 steps (best: step 2000) | 47/100 | 121/160 |
| FPO v1 θ3 | 57/100 | 122/160 |
| FPO v1 θ3, seed 2000 (official: 45/100) | 46/100 | — |

- A1: the label has no effect. Choosing the positive label is even 6 percentage points worse than the negative label
  (p = 0.29); the prompt suffix also hurts the other tasks.
- A2: on par with official at every checkpoint (p ≥ 0.68).
- Since neither offline approach improved on the original checkpoint, the work moved to on-policy RL (FPO).

Source: `ffi26/ffi26/results/round2/summary.md`, `readme_ROUND2.md`.

**DSRL (not run, expectations only).** DSRL keeps SmolVLA frozen and learns a small policy that chooses the initial noise ε
of flow matching.

- Expected: since SmolVLA's weights are not changed, there is no forgetting of other tasks, and no gradient through the VLA
  is needed.
- Expected limitation: it can only select among behaviours that SmolVLA can already produce from noise.
- This repository has no results for DSRL yet.

Details: `docs/analyse.md`, `docs/study.md`, `ffi26/ffi26/results/round2/readme_ROUND2.md`.

## Structure

```
docs/                     Reproduce / Analyse / Study, failure analysis
differences.md            differences from the paper, bug log
setup_lerobot.sh, check_env.py, smoke_test.sh     installation
run_queue.sh, collect_results.py                  Reproduce evaluation
ffi26/ffi26/
  improve/                Improve code (src/, scripts/), evaluation outputs
  results/runs/           Reproduce results
  results/round2/         configs + Improve result tables
  results/videos/         videos comparing the methods
  demo/                   run a demo of one checkpoint
```

## Installation

The scripts use the path `~/ffi26`, so create the symlink first:

```bash
git clone <repo> && cd <repo>
ln -s "$PWD/ffi26/ffi26" ~/ffi26
LEROBOT_REF=72e1d9e5e76ee719bacfc78bc34d817bee5fb521 bash setup_lerobot.sh   # conda env `lerobot`, LeRobot, LIBERO
conda activate lerobot && python check_env.py && bash smoke_test.sh
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
```

The `improve/scripts/*.sh` scripts call `source /opt/miniconda3/etc/profile.d/conda.sh`. If conda is installed elsewhere,
edit this line.

## Re-running Reproduce

```bash
TAG=D_nas10 NAS=10 CM=relative N_EP=50 SEEDS="1000 1001 1002" bash run_queue.sh
python collect_results.py --tag D_nas10
```

Results are in `~/ffi26/results/runs/D_nas10/`.

## Re-running Improve (FPO v3, v4)

```bash
cd ~/ffi26/improve && R=~/ffi26/results/round2/configs
# 1. demonstration data (once)
python scripts/download_demos.py
python scripts/phase2_demos_subset.py --config $R/round1.yaml
python scripts/fpo_build_demo_cache.py --config $R/fpo_v3.yaml
# 2. v2 (v3 and v4 reuse the round-0 rollouts and the official evaluation of v2)
bash scripts/fpo_v2_loop.sh
# 3. v3, v4 (about 3 hours each)
bash scripts/fpo_v3_loop.sh
bash scripts/fpo_v4_loop.sh
```

Each loop: train 4 rounds → select the checkpoint → evaluate → table `results/round2/fpo_v{3,4}/pooled.md`.
Checkpoints are saved in `improve/outputs/fpo_v{3,4}/iter3/train/pretrained_model` (not included in the repository).
When re-run, the scripts skip steps that are already done.

Quick demo of one checkpoint: `ffi26/ffi26/demo/readme_DEMO.md`.
