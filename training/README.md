# Training

Train the Body and Hand experts from paired simulator observations and executable controller references. Run these commands from the repository root. This recipe uses one public AppleToPlate demonstration and the supplied fixed-base scene; its held split tests initial robot joint perturbations, not unseen trajectories, object placements, or table materials.

## 1. Setup

Create the environment and install the package:

```bash
conda create -n movens-interact python=3.10 -y
conda activate movens-interact
pip install -e ".[train]"
source scripts/download_models.sh --models base backbone
source simulation/prepare_assets.sh
```

The training pipeline expects these local files:

| Path | Content |
| --- | --- |
| `checkpoints/Movens-Zero-XL/model.safetensors` | Original XL Body and Qwen weights |
| `checkpoints/Qwen2.5-Omni-3B/` | Qwen configuration, tokenizer, and processor assets |
| `checkpoints/Movens-Interact/body_init.pt` | Optional fitted Body initialization, supplied locally |
| `configs/body.json` | Matching Body architecture and motion configuration |
| `configs/body_normalization.json` | Original 40D Body normalization |
| `assets/scene.xml`, `assets/scene.json`, `assets/robot.json` | Fixed scene, joint mapping, and reset protocol |

The public downloader provides the original base checkpoint and Qwen processor assets. The fitted `body_init.pt` and final `adapter.pt` remain local and are not included in this code release. The fresh-clone commands below use `--no-body-init` to train from the original Movens-Zero weights. This differs from the initialization used by the current fitted local policy; evaluate the resulting checkpoint separately.

The scripts use one CUDA device, eight CPU threads, BF16 model autocast, FP32 trainable parameters, and a per-process CUDA memory fraction of `0.35` for training. FK, physical losses, and angular metrics run in FP32. Qwen caching uses a memory fraction of `0.30` and releases its model when the process exits.

For headless rendering, select the available MuJoCo backend before collection:

```bash
export MUJOCO_GL=egl
```

## 2. Prepare data

Prepare the pinned episode from the public [GR00T-N1.7-AppleToPlate dataset](https://huggingface.co/datasets/nvidia/GR00T-N1.7-AppleToPlate/tree/d89c126a713c6632432a607c12661546ff4d6ea9):

```bash
source training/prepare_data.sh
```

This downloads only episode 0 Parquet and three small metadata files, checks fixed SHA-256 values, preserves the originals under `data/apple_raw/<revision>/`, and writes `data/apple_prepared/episode_000000.npz` with its manifest. Reuse an existing raw cache with `--raw-dir`; add `--offline` to forbid downloads. `--download-video` optionally fetches the original ego video for inspection. The new training observations come from the simulator, so that real video is not used as the training image stream.

The original data provide measured joint positions and desired joint commands. The preparation step maps body joints to the original Isaac order and hand joints to index, middle, thumb order. It linearly resamples measured Body joints and desired Hand commands to 50 Hz. Body references therefore come from measured real joint trajectories, not the original expert's body commands. Cross-source joint-order assumptions are recorded in the manifest; equal finger command pairs and constant right-hand targets prevent full dynamic identification.

There is no measured real root pose in these data. Preparation does not estimate or fabricate one. Collection uses the known fixed simulator base pose for both observed and desired absolute-body arrays. This reference is valid only after successful execution in the supplied scene.

## 3. Collect paired simulator teachers

```bash
source training/collect.sh
```

The collector first checks the published nominal reset against the prepared initial joint state and validates every reference without rendering. It then repeats each successful simulation to collect RGB, requiring identical physical trajectories with and without rendering. It refuses to overwrite an existing collection; choose another `--output` for a new experiment.

The default collection contains one nominal and three perturbed training episodes plus one held initial-joint perturbation. The legs, fixed root, apple, plate, table, and camera stay fixed. Arm and waist perturbations are at most `0.005` radians; hand perturbations are at most `0.002` radians. All five episodes share the same desired reference trajectory.

At observation tick `t`, the next interval executes `reference[t+1]` with a 50 Hz zero-order hold over ten 500 Hz physics steps. The loader trains `observation[t]` against `reference[t+1:t+31]`. RGB is recorded from the actual current state at 10 Hz. Successful labels require a supported grasp and lift followed by stable placement on the plate, with contact and force checks that exclude robot or table support after release.

The output is `data/apple_teacher/index.json`, with these per-episode arrays:

| Field | Shape | Meaning |
| --- | --- | --- |
| `timestamps` | `(T,)` | 50 Hz observation times |
| `body_obs_abs` | `(T,36)` | Actual body29, root xyz, root quaternion wxyz |
| `body_ref_abs` | `(T,36)` | Desired controller body29 and the known fixed root |
| `hand_obs` | `(T,14)` | Actual canonical hand joint positions |
| `hand_ref` | `(T,14)` | Desired canonical hand commands |
| `rgb` | `(N,360,480,3)` | Lossless current-ego RGB in a separate NPZ |
| `frame_indices`, RGB `timestamps` | `(N,)` | Exact observed source tick for each RGB frame |

Use `source training/collect.sh --preflight-only --output data/apple_teacher_preflight` for physics validation without image collection. This output is intentionally incomplete and cannot be used for caching or training.

## 4. Cache frozen visual context

```bash
source training/cache.sh
```

The cache encodes each simulator RGB frame with the original frozen Qwen weights and the prompt `Move the apple to the plate.`. A single observed frame is duplicated for the temporal patching interface; it never uses a later image. Training selects the latest cached timestamp at or before the current observation time.

Each `data/apple_context/episode_XXXXXX.npz` contains FP16 `context[N,L,2048]`, timestamps, frame indices, and the source RGB checksum. The manifest records the dataset, scene, base checkpoint, processor configuration, prompt, and output checksums. Existing files are reused only when those identities match. No Qwen weights are optimized during precision training.

## 5. Train

Run a CPU data and gradient-wiring check first:

```bash
source training/train.sh --no-body-init --check-data-only --output runs/apple_data_check
```

This validates successful paired teachers, split separation, all file identities, causal image lookup, known fixed-root consistency, and the FK gradient path. It checks that an analytical flow velocity recovers the clean target and that committed-prefix velocity gradients are zero. It loads no model weights and performs no optimizer updates.

Start the first precision stage from the original base:

```bash
source training/train.sh --no-body-init --output runs/apple_from_base
```

Defaults are 300 Hand-only warmup updates followed by 3,000 joint precision updates, batch size 8, Body LR `5e-5`, Hand LR `2e-4`, and seed `20261101`. Warmup freezes the complete Body expert and checks that its fixed probe output remains bitwise unchanged. The joint stage trains the final three Body blocks and the action input, observation, and output projections. The original Qwen and all other Body modules stay frozen.

Hand reads the paired clean teacher Body plan during training. Evaluation reports Hand conditioned on both the teacher plan (`B_ref`) and the generated plan (`B_hat`). Both experts use the same randomly chosen 0–10-frame committed prefix; prefix and padded frames have separate masks. The policy still predicts 30 frames, executes 20, and retains 10 for the next chunk. Future-progress features use the same known 20-second budget as inference, not the recorded demonstration's end time.

The flow convention is `X=(1-sigma)*A+sigma*noise`, velocity `noise-A`, and clean estimate `X-sigma*velocity`. Physical losses add wrist position and orientation error, body joint error, soft joint-limit penalties, and Hand command error. Default weights are:

| Loss | Weight |
| --- | --- |
| Wrist position, metres squared | `5000` |
| Wrist orientation, chordal rotation error | `10` |
| Body joint error, radians squared | `5` |
| Soft joint-limit error, radians squared | `10` |
| Hand physical command error, radians squared | `5` |
| Left wrist relative to right wrist | `2` |

Body uses the unchanged original 40D normalizer. Hand statistics use training episodes only. Hand-state ranges have a minimum width of `0.1` radians; action ranges retain their measured training bounds. Constant action channels remain recorded as protocol metadata. Raw generative metrics do not project those channels to constants, and zero right-hand targets do not establish a learned right-hand skill.

## 6. Resume and refine

After the first stage reaches global step 3,300, continue for another 3,000 updates in a new output directory:

```bash
source training/train.sh \
  --no-body-init \
  --resume runs/apple_from_base/checkpoint_last.pt \
  --joint-steps 6000 \
  --body-lr-scale 0.5 --hand-lr-scale 0.25 \
  --cosine-decay --lr-min-scale 0.2 \
  --output runs/apple_from_base_refined
```

`--joint-steps` is the total number of joint-stage updates, not an increment. With 300 warmup updates, `6000` ends at global step 6,300. The resume file restores both experts, Adam state, random-number state, and progress. It must match the original data, cache, scene, normalization, architecture, seed, batch settings, and loss recipe.

The LR multipliers apply to the original base LRs after optimizer restoration; they never multiply an already decayed LR. The cosine interval is fixed from the continuation's starting step to its total target step. To resume an interrupted continuation in the same directory, repeat its exact schedule flags and target step count.

Evaluation runs every 500 updates by default and reports per-wrist RMS, P95 and maximum position and angular errors, body joint RMSE, and separate left/right Hand MAE and RMSE. The best snapshot is selected only from the nominal training trajectory using the predeclared score:

```text
max(left_wrist_RMS_m, right_wrist_RMS_m) / 0.01 + left_hand_B_hat_MAE_rad / 0.03
```

The held split is reported independently and does not select the checkpoint. Open-loop fit metrics do not establish closed-loop task success; run the separate simulator policy evaluation on the selected adapter.

### Optional fitted Body initialization

When a fitted Body-only initializer is supplied locally, place it at `checkpoints/Movens-Interact/body_init.pt` and run:

```bash
source training/train.sh \
  --body-init checkpoints/Movens-Interact/body_init.pt \
  --output runs/apple_precision
source training/train.sh \
  --resume runs/apple_precision/checkpoint_last.pt \
  --joint-steps 6000 \
  --body-lr-scale 0.5 --hand-lr-scale 0.25 \
  --cosine-decay --lr-min-scale 0.2 \
  --output runs/apple_precision_refined
```

The existing local workspace has this initializer, so this fitted recipe remains usable there. It contains Body weights from the earlier single-demonstration fitting stage, without Hand weights or an optimizer. The current fitted local policy used this initialization, 300 Hand warmup updates, 3,000 joint updates, and another 3,000 refinement updates. It is distinct from the fresh-clone base initialization above. Resume uses the saved optimizer checkpoint and does not reload the initializer.

For inference or simulation, supply a locally trained checkpoint explicitly:

```bash
source inference/predict.sh --checkpoint runs/apple_from_base_refined/model_best_nominal.pt
source simulation/rollout.sh --checkpoint runs/apple_from_base_refined/model_best_nominal.pt --output outputs/rollout
```

`model_delta.pt`, `model_best_nominal.pt`, and `checkpoint_last.pt` are supported policy inputs. Only `checkpoint_last.pt` includes optimizer state for training resume. To use the default inference path, copy a chosen local policy checkpoint to `checkpoints/Movens-Interact/adapter.pt`.

## Outputs

| Path | Content |
| --- | --- |
| `data/apple_prepared/manifest.json` | Pinned source, mapping, and reference semantics |
| `data/apple_teacher/index.json` | Successful paired teachers, reset protocols, split, and hashes |
| `data/apple_context/manifest.json` | Frozen Qwen cache identities and causal protocol |
| `runs/<run>/config.json` | Training recipe, trainable modules, provenance, LR schedule |
| `runs/<run>/hand_normalization.json` | Raw train bounds, expanded state bounds, constant channels |
| `runs/<run>/data_check.json` | CPU data and FK integration checks |
| `runs/<run>/loss.jsonl` | Per-update losses, gradients, and LRs |
| `runs/<run>/metrics.jsonl` | Nominal fit and held-jitter evaluations |
| `runs/<run>/model_delta.pt` | Latest dual-expert inference adapter |
| `runs/<run>/model_best_nominal.pt` | Best nominal-fit inference adapter |
| `runs/<run>/checkpoint_last.pt` | Latest adapter, optimizer, RNG state, and resume metadata |
| `runs/<run>/training_report.json` | Final step count, continuation, memory, best snapshot |

Body adapter tensors contain absolute parameter values for the listed modules, not additive differences. The base checkpoint and original Body normalizer remain required at inference. Keep the adapter, configuration, and identity metadata together.

## Common options

| Command | Option | Default |
| --- | --- | --- |
| Prepare | `--raw-dir`, `--output` | `data/apple_raw`, `data/apple_prepared` |
| Collect | `--prepared`, `--scene-setup`, `--robot-config` | Prepared manifest and supplied asset JSONs |
| Cache | `--manifest`, `--output`, `--batch-size` | Teacher index, `data/apple_context`, `8` |
| Train | `--manifest`, `--context-dir`, `--scene-xml` | Teacher index, context directory, `assets/scene.xml` |
| Train | `--body-init`, `--no-body-init` | Optional local initializer; use `--no-body-init` without it |
| Train | `--base-checkpoint` | Original XL weights |
| Train | `--hand-warmup-steps`, `--joint-steps` | `300`, `3000` |
| Train | `--batch-size`, `--stride` | `8`, `10` |
| Train | `--eval-every`, `--save-every`, `--eval-batches` | `500`, `500`, `0` (all windows) |
| Train | `--condition-dropout`, `--cfg-scale` | `0.1`, `1.0` |
| Train | `--body-lr-scale`, `--hand-lr-scale`, `--lr-min-scale` | `1`, `1`, `0.1` |

All four Python programs accept `--help`. Their shell launchers activate `movens-interact`, switch to the repository root, and forward all command-line arguments.
