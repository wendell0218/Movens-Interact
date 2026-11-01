import contextlib
import math

import torch
from torch import nn
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler


class HandExpert(nn.Module):
    def __init__(self, hidden_dim=256, num_layers=3, num_heads=8,
                 context_dim=2048, action_horizon=30):
        super().__init__()
        self.action_dim, self.action_horizon = 14, action_horizon
        self.input_proj = nn.Linear(14, hidden_dim)
        self.state_proj = nn.Linear(14, hidden_dim)
        self.body_plan_encoder = nn.Sequential(nn.Linear(40, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.context_proj = nn.Sequential(nn.Linear(context_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.time_proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.register_buffer("time_frequencies", torch.exp(-math.log(10000) * torch.arange(hidden_dim // 2) / (hidden_dim // 2 - 1)))
        self.hand_position = nn.Parameter(torch.randn(1, action_horizon, hidden_dim) * 0.02)
        self.body_position = nn.Parameter(torch.randn(1, action_horizon, hidden_dim) * 0.02)
        self.memory_type = nn.Parameter(torch.randn(3, hidden_dim) * 0.02)
        layer = nn.TransformerDecoderLayer(hidden_dim, num_heads, hidden_dim * 4,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerDecoder(layer, num_layers, norm=nn.LayerNorm(hidden_dim))
        self.output_proj = nn.Linear(hidden_dim, 14)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, noisy_hand, sigma, context, hand_state, body_plan,
                context_mask=None, body_valid=None):
        batch, horizon, dimensions = noisy_hand.shape
        if dimensions != 14 or horizon != self.action_horizon or body_plan.shape != (batch, horizon, 40):
            raise ValueError("Hand expects noise[B,30,14] and clean normalized Body[B,30,40]")
        if sigma.ndim == 1:
            sigma = sigma[:, None].expand(batch, horizon)
        views = context[:, 0] if context.ndim == 4 else context
        hand_state = hand_state.reshape(batch, -1, 14)
        phase = sigma.float()[..., None] * 1000 * self.time_frequencies
        time = self.time_proj(torch.cat((phase.cos(), phase.sin()), dim=-1))
        tokens = self.input_proj(noisy_hand) + self.hand_position + time
        body_tokens = self.body_plan_encoder(body_plan) + self.body_position + self.memory_type[0]
        state_tokens = self.state_proj(hand_state) + self.memory_type[1]
        context_tokens = self.context_proj(views) + self.memory_type[2]
        memory = torch.cat((body_tokens, state_tokens, context_tokens), dim=1)
        if context_mask is None:
            context_mask = torch.ones(views.shape[:2], dtype=torch.bool, device=views.device)
        if body_valid is None:
            body_valid = torch.ones((batch, horizon), dtype=torch.bool, device=views.device)
        memory_valid = torch.cat((body_valid.bool(), torch.ones(hand_state.shape[:2], dtype=torch.bool, device=views.device), context_mask.bool()), dim=1)
        decoded = self.transformer(tokens, memory, tgt_key_padding_mask=~body_valid.bool(), memory_key_padding_mask=~memory_valid)
        return self.output_proj(decoded)


@torch.inference_mode()
def sample_flow(expert, state, context, context_mask, progress=None, body_plan=None,
                prefix=None, prefix_lengths=None, steps=10, seed=0, body_valid=None,
                cfg_scale=1.0, cfg_interval=(0.2, 0.8)):
    device, batch = state.device, len(state)
    horizon, dim = expert.action_horizon, expert.action_dim
    generator = torch.Generator(device=device).manual_seed(seed)
    samples = torch.randn(batch, horizon, dim, device=device, generator=generator)
    lengths = torch.zeros(batch, dtype=torch.long, device=device) if prefix_lengths is None else torch.as_tensor(prefix_lengths, device=device).reshape(batch)
    if torch.any(lengths < 0) or torch.any(lengths >= horizon):
        raise ValueError("Invalid committed prefix length")
    mask = torch.arange(horizon, device=device)[None] < lengths[:, None]
    if prefix is None:
        if torch.any(mask):
            raise ValueError("Committed actions are required for a nonempty prefix")
        prefix = torch.zeros_like(samples)
    if prefix.shape != samples.shape:
        raise ValueError("prefix must have full [B,30,D] shape")
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(steps)
    for index, timestep in enumerate(scheduler.timesteps):
        samples = torch.where(mask[..., None], prefix, samples)
        sigma = torch.full((batch, horizon), float(scheduler.sigmas[index]), device=device).masked_fill(mask, 0)
        autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext()
        with autocast:
            if body_plan is None:
                conditional = expert(samples, sigma * 1000, context, state, progress, context_mask=context_mask)
            else:
                conditional = expert(samples, sigma, context, state, body_plan, context_mask, body_valid)
            scale = cfg_scale if cfg_interval is None or cfg_interval[0] <= float(scheduler.sigmas[index]) <= cfg_interval[1] else 1.0
            if scale != 1.0:
                if body_plan is None:
                    unconditional = expert(samples, sigma * 1000, context, state, progress, context_mask=torch.zeros_like(context_mask))
                else:
                    unconditional = expert(samples, sigma, context, state, body_plan, torch.zeros_like(context_mask), body_valid)
                conditional = unconditional + scale * (conditional - unconditional)
        samples = scheduler.step(conditional.float(), timestep, samples.float()).prev_sample
    return torch.where(mask[..., None], prefix, samples).float()
