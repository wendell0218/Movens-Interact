from pathlib import Path

from safetensors import safe_open


def checkpoint_model_config(path: str | Path) -> dict[str, int]:
    with safe_open(str(path), framework="pt", device="cpu") as weights:
        keys = list(weights.keys())
        prefix = "action_header." if any(k.startswith("action_header.") for k in keys) else "action_expert."
        projection = weights.get_slice(prefix + "action_proj_in.ac_proj.2.weight").get_shape()
        output = weights.get_slice(prefix + "action_proj_out.linear.weight").get_shape()
        state = weights.get_slice(prefix + "obs_proj._obs_proc.1.weight").get_shape()
        position = weights.get_slice(prefix + "action_proj_in.dec_pos").get_shape()
        blocks = {int(k.split("transformer_blocks.", 1)[1].split(".")[0]) for k in keys if k.startswith(prefix + "transformer_blocks.")}
    width, dimension = projection
    if output != [dimension, width] or state[0] != width or blocks != set(range(len(blocks))):
        raise ValueError("Checkpoint has an inconsistent action expert architecture")
    if width % 64 or dimension not in (39, 40):
        raise ValueError(f"Unsupported action expert: width={width}, dimension={dimension}")
    return dict(hidden_dim=width, num_blocks=len(blocks), num_heads=width // 64,
                action_dim=dimension, state_dim=state[1], action_horizon=position[0])
