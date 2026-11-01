from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from .backbone import MovensZeroModel
from .hand import HandExpert, sample_flow
from .normalization import MotionNormalizer
from .progress import build_future_progress
from .root40 import (
    choose_heading_anchor,
    encode_root40,
    recover_motion_abs_from_root40_action,
    reencode_root40_prefix,
)


ROOT = Path(__file__).resolve().parents[1]


def _path(value):
    value = Path(value).expanduser()
    return value.resolve() if value.is_absolute() or value.exists() else ROOT / value


def _normalizer(values):
    return MotionNormalizer(
        values["action"]["q01"], values["action"]["q99"],
        values["state"]["q01"], values["state"]["q99"],
        normalize_state_enabled=values.get("normalize_state", True),
        use_norm_mask=values.get("use_norm_mask", False),
        action_norm_masks=values.get("action_norm_masks"),
    )


class MovensInteractModel:
    action_horizon = 30
    execution_horizon = 20
    retained_prefix = 10
    progress_total_frames = 1001
    contact_protocol = "isaac_leg_delta_v2"

    def __init__(self, base, hand_expert, checkpoint, adapter_checkpoint):
        self.base = base
        self.body_expert = base.action_expert
        self.hand_expert = hand_expert
        self.checkpoint = checkpoint
        self.adapter_checkpoint = Path(adapter_checkpoint)
        self.device = base.runtime_device
        self.body_normalizer = _normalizer(checkpoint["body_normalization"])
        self.hand_normalizer = _normalizer(checkpoint["hand_normalization"])
        self.constant_hand_actions = np.abs(self.hand_normalizer.action_max - self.hand_normalizer.action_min) < 1e-8
        self.cfg_scale = float(checkpoint["config"]["training"].get("cfg_scale", 1.0))
        self.steps = 10
        self.seed = 42
        self.prompt = checkpoint.get("data_provenance", {}).get("context_prompt", "Move the apple to the plate.")

    @classmethod
    def from_pretrained(
        cls,
        adapter_checkpoint,
        base_checkpoint=None,
        backbone=None,
        device="cuda:0",
    ):
        adapter_checkpoint = _path(adapter_checkpoint)
        checkpoint = torch.load(adapter_checkpoint, map_location="cpu", weights_only=False)
        if checkpoint.get("format") != "movens_interact_dual_expert_v1":
            raise ValueError("Unsupported Movens-Interact checkpoint format")
        config = json.loads((ROOT / "configs/body.json").read_text())
        config["model"].update(checkpoint["config"]["body_config"])
        model_config = config["model"]
        if (model_config["action_dim"], model_config["state_dim"], model_config["action_horizon"], model_config["action_exec_horizon"]) != (40, 40, 30, 20):
            raise ValueError("The policy requires Body40 with a30/20/10 action timeline")
        base_checkpoint = _path(base_checkpoint or checkpoint.get("base_checkpoint", "checkpoints/Movens-Zero-XL/model.safetensors"))
        backbone = _path(backbone or "checkpoints/Qwen2.5-Omni-3B")
        runtime_device = torch.device(device)
        dtype = torch.bfloat16 if runtime_device.type == "cuda" else torch.float32
        base = MovensZeroModel.from_pretrained(base_checkpoint, backbone, config,
            device=runtime_device, torch_dtype=dtype)
        modules = checkpoint.get("body_trainable_modules",
            checkpoint["config"].get("body_trainable_modules", ["transformer_blocks.6", "action_proj_out"]))
        for name in modules:
            base.action_expert.get_submodule(name).float()
        loaded = base.action_expert.load_state_dict(checkpoint["body_delta_state_dict"], strict=False)
        if loaded.unexpected_keys:
            raise ValueError(f"Unexpected Body adapter parameters: {loaded.unexpected_keys}")
        base.requires_grad_(False).eval()
        hand = HandExpert(**checkpoint["config"]["hand_config"]).to(runtime_device).float()
        hand.load_state_dict(checkpoint["hand_state_dict"], strict=True)
        hand.requires_grad_(False).eval()
        return cls(base, hand, checkpoint, adapter_checkpoint)

    def _synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.inference_mode()
    def predict(self, ego_rgb, body_abs, hand_state, frame, previous_plan=None, seed=None):
        ego_rgb = np.asarray(ego_rgb)
        body_abs = np.asarray(body_abs, dtype=np.float32)
        hand_state = np.asarray(hand_state, dtype=np.float32)
        if body_abs.shape not in [(36,), (1, 36)] or hand_state.shape not in [(14,), (1, 14)]:
            raise ValueError("Provide one current Body36 pose and one current Hand14 state")
        body_abs = body_abs.reshape(1, 36)
        hand_state = hand_state.reshape(1, 14)
        if ego_rgb.ndim != 3 or ego_rgb.shape[-1] != 3 or ego_rgb.dtype != np.uint8:
            raise ValueError("ego_rgb must be one current HxWx3 uint8 RGB image")
        if not np.isfinite(body_abs).all() or not np.isfinite(hand_state).all():
            raise ValueError("Current body and hand states must be finite")
        if frame < 0 or int(frame) != frame or frame % self.execution_horizon:
            raise ValueError("frame must be a nonnegative multiple of20")
        frame = int(frame)
        plan_id = frame // self.execution_horizon
        anchor = choose_heading_anchor(body_abs)
        _, state40, _ = encode_root40(np.empty((0, 36), dtype=np.float32), body_abs,
            anchor, contact_protocol=self.contact_protocol)
        state_norm = self.body_normalizer.normalize_state(state40)
        hand_norm = self.hand_normalizer.normalize_state(hand_state)
        body_prefix = np.zeros((self.action_horizon, 40), dtype=np.float32)
        hand_prefix = np.zeros((self.action_horizon, 14), dtype=np.float32)
        prefix_length = 0
        if previous_plan is not None:
            if int(previous_plan["first_physical_frame"]) + self.execution_horizon - 1 != frame:
                raise ValueError("previous_plan must be the immediately preceding common Body/Hand plan")
            raw_prefix = reencode_root40_prefix(previous_plan["body_raw"],
                previous_plan["measured_body"], previous_plan["anchor"], body_abs, anchor,
                start_index=self.execution_horizon, length=self.retained_prefix,
                contact_protocol=self.contact_protocol)
            body_prefix[:self.retained_prefix] = self.body_normalizer.normalize_action(raw_prefix)
            hand_prefix[:self.retained_prefix] = previous_plan["hand_execution_normalized"][self.execution_horizon:]
            prefix_length = self.retained_prefix
        infer_started = time.perf_counter()
        inputs = self.base._prepare_inputs([self.prompt], "video", None,
            [np.stack([ego_rgb, ego_rgb])], video_sample_fps=30.0, video_size_pixels=50176)
        with self.base._autocast():
            context = self.base._extract_context(inputs).float()
        context_mask = inputs["attention_mask"].float()
        self._synchronize()
        context_seconds = time.perf_counter() - infer_started
        progress = torch.from_numpy(build_future_progress(frame, self.progress_total_frames, self.action_horizon)[None]).to(self.device)
        state_t = torch.from_numpy(state_norm[None]).to(self.device)
        hand_t = torch.from_numpy(hand_norm[None]).to(self.device)
        lengths = torch.tensor([prefix_length], device=self.device)
        seed = self.seed if seed is None else int(seed)
        body_started = time.perf_counter()
        with self.base._autocast():
            body_sample = sample_flow(self.body_expert, state_t, context, context_mask,
                progress=progress, prefix=torch.from_numpy(body_prefix[None]).to(self.device),
                prefix_lengths=lengths, steps=self.steps, seed=seed + plan_id * 2,
                cfg_scale=self.cfg_scale)
        self._synchronize()
        body_seconds = time.perf_counter() - body_started
        hand_started = time.perf_counter()
        with self.base._autocast():
            hand_sample = sample_flow(self.hand_expert, hand_t, context, context_mask,
                body_plan=body_sample, prefix=torch.from_numpy(hand_prefix[None]).to(self.device),
                prefix_lengths=lengths, steps=self.steps, seed=seed + plan_id * 2 + 1,
                cfg_scale=self.cfg_scale)
        self._synchronize()
        hand_seconds = time.perf_counter() - hand_started
        body_hat = body_sample[0].float().cpu().numpy()
        hand_hat = hand_sample[0].float().cpu().numpy()
        latency = time.perf_counter() - infer_started
        if not np.isfinite(body_hat).all() or not np.isfinite(hand_hat).all():
            raise FloatingPointError("Nonfinite Body or Hand prediction")
        body_error = float(np.max(np.abs(body_hat[:prefix_length] - body_prefix[:prefix_length]))) if prefix_length else 0.0
        hand_error = float(np.max(np.abs(hand_hat[:prefix_length] - hand_prefix[:prefix_length]))) if prefix_length else 0.0
        if max(body_error, hand_error) >= 1e-6:
            raise RuntimeError("The committed Body/Hand prefix changed")
        body_raw = self.body_normalizer.denormalize_action(body_hat)
        body_absolute = recover_motion_abs_from_root40_action(body_raw, body_abs, anchor)
        hand_absolute = self.hand_normalizer.denormalize_action(hand_hat)
        hand_absolute_raw = hand_absolute.copy()
        hand_absolute[:, self.constant_hand_actions] = self.hand_normalizer.action_min[self.constant_hand_actions]
        hand_execution_norm = self.hand_normalizer.normalize_action(hand_absolute)
        return {
            "plan_id": plan_id, "first_physical_frame": frame + 1,
            "horizon": self.action_horizon, "execute": self.execution_horizon,
            "prefix_length": prefix_length, "anchor": anchor,
            "body_raw": body_raw, "body_normalized": body_hat, "body_absolute": body_absolute,
            "hand_normalized": hand_hat, "hand_absolute": hand_absolute,
            "hand_absolute_raw": hand_absolute_raw, "hand_execution_normalized": hand_execution_norm,
            "measured_body": body_abs, "measured_hand": hand_state,
            "body_state_normalized": state_norm, "hand_state_normalized": hand_norm,
            "context": context.to(torch.float16).cpu().numpy(),
            "context_mask": context_mask.cpu().numpy(), "prompt": self.prompt,
            "context_seconds": context_seconds, "body_seconds": body_seconds,
            "hand_seconds": hand_seconds, "total_inference_seconds": latency,
            "prefix_body_max_abs_error": body_error, "prefix_hand_max_abs_error": hand_error,
        }
