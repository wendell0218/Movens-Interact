# Simulation

This directory runs a G1 tabletop manipulation policy in MuJoCo. The default scene fixes the robot base and drives 29 body joints plus 14 hand joints with position servos. Body root-motion channels are predicted but are not executed by this controller. This is separate from SONIC whole-body tracking.

## Prepare assets

Activate the environment described in the [main README](../README.md), then run:

```bash
source simulation/prepare_assets.sh
```

The downloader verifies 63 meshes and textures against pinned SHA256 digests. The download is approximately 37.4 MB. To reuse an existing GR00T-WholeBodyControl checkout without downloading:

```bash
source simulation/prepare_assets.sh --cache /path/to/GR00T-WholeBodyControl --offline
```

Binary assets are fetched directly from upstream and stored under `assets/vendor/`. Their source revision and checksums are recorded in `assets/manifest.json`; see [asset notices](../assets/NOTICE.md). The scene uses relative asset paths.

## Run the policy

```bash
source simulation/rollout.sh --checkpoint runs/apple_from_base/model_best_nominal.pt --seed 42 --output outputs/rollout
```

Supply a locally trained task checkpoint with `--checkpoint`, as above. The fitted
task weights are kept locally; the [training guide](../training/README.md) covers
training from the public backbone. To use the defaults, copy your task checkpoint
to the adapter path below. Default model locations are:

```text
checkpoints/Movens-Interact/adapter.pt
checkpoints/Movens-Zero-XL/model.safetensors
checkpoints/Qwen2.5-Omni-3B/
```

Use `--checkpoint`, `--base-checkpoint`, `--backbone` or `--device` to override them. Use a separate output directory for each model, seed or scene.

The policy observes the current ego RGB image and actual body/hand state. It predicts 30 frames at 50 Hz, executes 20 frames and preserves a 10-frame committed prefix for the next plan. Body and hand targets share one execution clock. Each target is held for ten 500 Hz physics steps. Initialization uses the exact `qpos43`, `ctrl43` and 100-step settling protocol in `assets/robot.json`.

Physics pauses while visual encoding and Body/Hand inference run. The output is a slow closed-loop policy diagnostic, not a real-time deployment result. Object poses are used by the simulator and evaluation metrics; they are not provided to the policy.

For a CPU interface check that holds the current robot pose without loading a policy:

```bash
source simulation/rollout.sh --interface-smoke --no-video --seconds 0.4 --output outputs/interface_smoke
```

Its report deliberately leaves policy success unset.

For a saved teacher reset, only the requested reset protocol is read:

```bash
source simulation/rollout.sh --checkpoint runs/apple_from_base/model_best_nominal.pt --reset-index data/apple_teacher/index.json --reset-episode 4 --output outputs/held_reset
```

## Outputs and success checks

`rollout_report.json` contains scene/reset identity, seed, inference time, tracking errors and support verification. The original source-scene hash and portable scene-file hash are separate: rewriting asset paths changes file bytes while preserving physical parameters.

`plan_*.npz` contains predicted Body/Hand plans, measured input state, anchor, prefix diagnostics and visual context. `observed_ego_*.png` contains the actual causal policy images. `physical_states.npz` contains every executed `qpos`, `qvel` and `ctrl`; `policy_measurements.npz` and `support_force_measurements.npz` contain tracking, object contact and world-coordinate force records.

`support_verified_success` requires a left-hand contact lift at least 2 cm above the settled height for at least 0.2 seconds, followed by one second of stable plate support. The final window requires continuous plate contact, no table/robot/other support, low apple speed and height variation, conservative plate containment, and mean plate vertical force within 5% of the apple's weight. The configured containment radius is for the supplied apple and plate sizes. These checks describe this simulated scene, not a real-world success rate.

`rollout.mp4` shows the live policy execution alongside its ego view. `--no-video` disables video encoding while keeping policy observations and numerical records.

## Render saved states

```bash
source simulation/render.sh --run-dir outputs/rollout --view overview
source simulation/render.sh --run-dir outputs/rollout --view close --frame -1
source simulation/render.sh --run-dir outputs/rollout --view topdown --frame -1
```

Rendering uses saved physical states and changes only the camera view. It does not run a new policy, integrate physics or substitute a demonstration.

## Controlled robustness cases

```bash
source simulation/randomize.sh --output outputs/randomization
source simulation/evaluate.sh --checkpoint runs/apple_from_base/model_best_nominal.pt --index outputs/randomization/index.json --output outputs/robustness
```

The 12 predefined cases are the baseline, four apple XY offsets of ±1 cm, four plate XY offsets of ±2 cm, two table appearance changes, and one combined case. Table variants change only visual material; friction, collision, inertia, camera, robot reset and servo settings remain fixed. XML-requested and post-settle object poses are recorded separately because a free apple can roll during settling.

`evaluate.sh` creates the default variant set automatically when `outputs/randomization/index.json` is missing. Run `randomize.sh` separately when inspecting the cases first or supplying a custom scene.

Evaluation keeps one checkpoint and sampling seed across cases and writes JSON/CSV summaries. Missing or failed runs are explicit. One run per case is a controlled comparison, not an estimate over independent trials.

```bash
source simulation/evaluate.sh --cases baseline table_dark_gray --dry-run
```

## Custom scenes and headless rendering

`--scene-setup /path/to/scene.json` accepts an external setup. The JSON identifies the XML file, its checksum, robot configuration, camera, free apple body/joint and plate body. Paths inside setup JSON are resolved relative to that JSON file. Keep the G1 joint/actuator names and reference units compatible with `assets/robot.json`. The default controlled-variant generator additionally requires the supplied object-body and table-material names.

The launch scripts select EGL. On a Linux host with Mesa, software rendering can be selected explicitly before launch:

```bash
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
export EGL_PLATFORM=surfaceless
export LIBGL_ALWAYS_SOFTWARE=1
```

The approximate torso camera is not calibrated to a physical robot camera. Different rendering backends can produce small image differences.
