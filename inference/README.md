# Offline prediction

This entry predicts one paired Body/Hand action chunk from a current ego image and robot observation. Install the environment and public backbone files using the [main README](../README.md), then supply a locally trained task checkpoint:

```bash
source inference/predict.sh --checkpoint runs/apple_from_base/model_best_nominal.pt
```

The bundled `examples/ego.png` and `examples/observation.npz` are a matched simulated observation. The command writes `outputs/prediction/plan.npz` and `summary.json`. The fitted task weights are kept locally; see the [training guide](../training/README.md) to create a checkpoint from the public backbone.

## Inputs

```bash
source inference/predict.sh \
  --checkpoint runs/apple_from_base/model_best_nominal.pt \
  --image /path/to/current_ego.png \
  --observation /path/to/current_observation.npz \
  --frame 0 \
  --seed 42 \
  --output outputs/prediction
```

The observation archive contains:

| Key | Shape | Meaning |
| --- | --- | --- |
| `body_abs` | `1 × 36` | IsaacLab-order body joints29, root XYZ3, root quaternion WXYZ4 |
| `hand_state` | `1 × 14` | Measured left and right hand joints |
| `frame` | scalar, optional | Number of already executed 50 Hz frames |

Each hand uses `index0, index1, middle0, middle1, thumb0, thumb1, thumb2`; left precedes right. Angles are in radians and position is in meters. Use a current image and measured robot state from the same observation time. The frame index provides progress within the preset 20-second budget; it must not encode an unknown completion time.

Model defaults are `checkpoints/Movens-Interact/adapter.pt`, `checkpoints/Movens-Zero-XL/model.safetensors` and `checkpoints/Qwen2.5-Omni-3B/`. Override them with `--checkpoint`, `--base-checkpoint` and `--backbone`; `--device` defaults to `cuda:0`.

## Plan outputs

The plan contains `body_absolute[30,36]`, `hand_absolute[30,14]`, normalized Body/Hand predictions, raw hand predictions, the heading anchor, measured input state and timing diagnostics. Sampling uses ten flow steps and the checkpoint's CFG configuration. The fitted checkpoint uses CFG 1.

Execute 20 synchronized Body/Hand frames. At the next observation, pass the entire previous archive with `--previous-plan` and set `--frame 20`. The model re-encodes the retained ten-frame body prefix into the current anchor and keeps the hand prefix on the same physical frames. Continue with newly captured observations for later chunks.

Offline prediction does not execute physics or verify object contact. Use [simulation rollout](../simulation/README.md) to evaluate the learned policy with a free apple and actual contact dynamics.

## Python API

```python
from movens_interact.model import MovensInteractModel

model = MovensInteractModel.from_pretrained(
    "runs/apple_from_base/model_best_nominal.pt",
    device="cuda:0",
)
plan = model.predict(ego_rgb, body_abs, hand_state, frame=0, seed=42)
```

Pass the returned plan as `previous_plan` when requesting the next chunk. Physical joint-limit clipping and control execution belong to the controller, outside this API.
