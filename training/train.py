import argparse
import hashlib
import json
import math
import random
import struct
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
from safetensors import safe_open

from movens_interact.kinematics import G1Kinematics
from movens_interact.hand import HandExpert, sample_flow
from movens_interact.backbone import _ActionExpert
from movens_interact.normalization import MotionNormalizer
from movens_interact.progress import build_future_progress
from movens_interact.root40 import encode_root40

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collate_windows(items, device):
    batch = {name: torch.as_tensor(np.stack([item[name] for item in items]), device=device) for name in ["body_action", "body_state", "hand_action", "hand_state", "progress", "valid"]}
    length = max(len(item["context"]) for item in items)
    context = np.zeros((len(items), 1, length, 2048), dtype=np.float16)
    context_mask = np.zeros((len(items), length), dtype=bool)
    for i, item in enumerate(items):
        count = len(item["context"])
        context[i, 0, :count] = item["context"]
        context_mask[i, :count] = item["context_mask"]
    batch["context"] = torch.as_tensor(context, device=device)
    batch["context_mask"] = torch.as_tensor(context_mask, device=device)
    return batch


class SimWindows:
    def __init__(self, index_path, context_dir, body_normalizer, hand_normalizer, split, stride=10, episode_ids=None):
        self.path = Path(index_path)
        index = json.loads(self.path.read_text())
        if index["fps"] != 50:
            raise ValueError("Expected a 50 Hz paired simulation teacher")
        self.episodes, self.windows = [], []
        self.body_norm, self.hand_norm = body_normalizer, hand_normalizer
        for row in index["episodes"]:
            if row["split"] != split or not row["success"] or not row["teacher_success"]:
                continue
            if not row["rgb_ready"]:
                raise ValueError("Successful teacher is missing paired ego observations")
            if episode_ids is not None and row["episode_index"] not in episode_ids:
                continue
            if row["root_source"] != "sim_known_fixed_base":
                raise ValueError("Precision training requires a known simulation root")
            if sha256_file(self.path.parent / row["trajectory"]) != row["trajectory_sha256"]:
                raise ValueError("Paired teacher trajectory checksum changed")
            with np.load(self.path.parent / row["trajectory"], allow_pickle=False) as archive:
                episode = {key: archive[key] for key in ["timestamps", "body_obs_abs", "body_ref_abs", "hand_obs", "hand_ref"]}
            with np.load(Path(context_dir) / f"episode_{row['episode_index']:06d}.npz", allow_pickle=False) as archive:
                episode["context"] = archive["context"]
                episode["context_times"] = archive["timestamps"].astype(np.float64)
                episode["context_frames"] = archive["frame_indices"]
                if str(archive["source_sha256"]) != row["rgb_sha256"]:
                    raise ValueError("Context cache and teacher RGB identity differ")
            if sha256_file(self.path.parent / row["rgb"]) != row["rgb_sha256"]:
                raise ValueError("Teacher RGB checksum changed")
            count = len(episode["timestamps"])
            if episode["body_obs_abs"].shape != (count, 36) or episode["body_ref_abs"].shape != (count, 36) or episode["hand_obs"].shape != (count, 14) or episode["hand_ref"].shape != (count, 14):
                raise ValueError("Paired observation/reference shapes disagree")
            if not all(np.isfinite(value).all() for value in episode.values()):
                raise ValueError("Nonfinite teacher/cache values")
            if np.max(np.abs(np.diff(episode["timestamps"]) - .02)) > 1e-6 or np.any(np.diff(episode["context_times"]) < 0):
                raise ValueError("Invalid teacher/context time grid")
            if episode["context"].ndim != 3 or episode["context"].shape[-1] != 2048 or episode["context_times"][0] > episode["timestamps"][0] + 1e-8:
                raise ValueError("Missing initial causal Qwen context")
            if np.max(np.abs(episode["body_obs_abs"][:, 29:36] - episode["body_obs_abs"][0, 29:36])) > 1e-6 or np.max(np.abs(episode["body_ref_abs"][:, 29:36] - episode["body_obs_abs"][0, 29:36])) > 1e-6:
                raise ValueError("This pilot's FK root must be the same known fixed base")
            episode["episode_index"] = row["episode_index"]
            episode["perturbation_seed"] = row["perturbation_seed"]
            slot = len(self.episodes)
            self.episodes.append(episode)
            self.windows.extend((slot, frame) for frame in range(0, count - 1, stride))
        if not self.windows:
            raise ValueError(f"No successful paired teachers for split {split}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, item):
        slot, frame = self.windows[item]
        ep = self.episodes[slot]
        indices = frame + 1 + np.arange(30)
        valid = indices < len(ep["timestamps"])
        selected = np.minimum(indices, len(ep["timestamps"]) - 1)
        current = ep["body_obs_abs"][frame:frame + 1]
        reference = ep["body_ref_abs"][selected]
        body_action, body_state, _ = encode_root40(reference, current, contact_protocol="isaac_leg_delta_v2")
        timestamp = float(ep["timestamps"][frame])
        cached = int(np.searchsorted(ep["context_times"], timestamp, side="right") - 1)
        if cached < 0 or ep["context_times"][cached] > timestamp + 1e-9:
            raise ValueError("Future ego context would leak into the policy input")
        return {"body_action": self.body_norm.normalize_action(body_action), "body_state": self.body_norm.normalize_state(body_state),
            "hand_action": self.hand_norm.normalize_action(ep["hand_ref"][selected]),
            "hand_state": self.hand_norm.normalize_state(ep["hand_obs"][frame:frame + 1]),
            "progress": build_future_progress(frame, 1001, 30), "valid": valid,
            "context": ep["context"][cached], "context_mask": np.ones(ep["context"].shape[1], dtype=bool),
            "body_ref_joints": reference[:, :29].astype(np.float32), "hand_ref": ep["hand_ref"][selected].astype(np.float32),
            "root_xyz": current[0, 29:32].astype(np.float32), "root_quat": current[0, 32:36].astype(np.float32),
            "episode_index": ep["episode_index"], "frame": frame, "timestamp": timestamp, "context_timestamp": float(ep["context_times"][cached])}


def precise_batch(items, device):
    batch = collate_windows(items, device)
    for key in ["body_ref_joints", "hand_ref", "root_xyz", "root_quat"]:
        batch[key] = torch.as_tensor(np.stack([item[key] for item in items]), device=device, dtype=torch.float32)
    return batch


def physical_action(values, lower, upper):
    width = upper - lower
    return torch.where(width.abs() >= 1e-8, (values.float() + 1) * .5 * width + lower, values.float())


@torch.inference_mode()
def evaluate_precise(body, hand, fk, dataset, args, bounds, label):
    hand.eval()
    count = len(dataset) if args.eval_batches == 0 else min(len(dataset), args.eval_batches * args.batch_size)
    indices = np.linspace(0, len(dataset) - 1, count).astype(int)
    position_errors, angle_errors, joint_errors = [], [], []
    hand_errors = {"B_ref": [], "B_hat": []}
    for offset in range(0, count, args.batch_size):
        batch = precise_batch([dataset[int(i)] for i in indices[offset:offset + args.batch_size]], args.device)
        seed = args.seed + 500000 + offset
        b_hat = sample_flow(body, batch["body_state"], batch["context"], batch["context_mask"], progress=batch["progress"], steps=10, seed=seed, cfg_scale=args.cfg_scale)
        h_ref = sample_flow(hand, batch["hand_state"], batch["context"], batch["context_mask"], body_plan=batch["body_action"], body_valid=batch["valid"], steps=10, seed=seed + 1, cfg_scale=args.cfg_scale)
        h_hat = sample_flow(hand, batch["hand_state"], batch["context"], batch["context_mask"], body_plan=b_hat, body_valid=batch["valid"], steps=10, seed=seed + 1, cfg_scale=args.cfg_scale)
        with torch.autocast("cuda", enabled=False):
            joints = physical_action(b_hat[..., :29], bounds["body_lo"], bounds["body_hi"])
            p, r = fk(joints.float(), batch["root_xyz"], batch["root_quat"])
            target_p, target_r = fk(batch["body_ref_joints"], batch["root_xyz"], batch["root_quat"])
            valid = batch["valid"]
            position_errors.append((p - target_p).norm(dim=-1)[valid].cpu().numpy())
            relative = r.transpose(-1, -2) @ target_r
            cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) * .5).clamp(-1, 1)
            angle_errors.append(torch.acos(cosine)[valid].cpu().numpy() * (180 / np.pi))
            joint_errors.append((joints - batch["body_ref_joints"])[valid].cpu().numpy())
            for name, prediction in [("B_ref", h_ref), ("B_hat", h_hat)]:
                raw = physical_action(prediction, bounds["hand_lo"], bounds["hand_hi"])
                hand_errors[name].append((raw - batch["hand_ref"])[valid].cpu().numpy())
    position = np.concatenate(position_errors)
    angle = np.concatenate(angle_errors)
    joints = np.concatenate(joint_errors)
    report = {"split": label, "episode_ids": [e["episode_index"] for e in dataset.episodes], "windows": count,
        "sample_steps": 10, "raw_outputs_projected": False, "wrist_names": fk.metadata["wrist_names"],
        "wrist_position_rms_m": np.sqrt(np.mean(position ** 2, axis=0)).tolist(),
        "wrist_position_p95_m": np.percentile(position, 95, axis=0).tolist(), "wrist_position_max_m": position.max(axis=0).tolist(),
        "wrist_angle_rms_deg": np.sqrt(np.mean(angle ** 2, axis=0)).tolist(), "wrist_angle_p95_deg": np.percentile(angle, 95, axis=0).tolist(), "wrist_angle_max_deg": angle.max(axis=0).tolist(),
        "body_joint_rmse_rad": float(np.sqrt(np.mean(joints ** 2))), "hand": {},
        "boundary": "Known fixed scene and shared desired teacher trajectory; held initial-state jitter is not unseen-task/trajectory generalization or closed-loop success."}
    for name, chunks in hand_errors.items():
        error = np.concatenate(chunks)
        report["hand"][name] = {"left_mae_rad": float(np.abs(error[:, :7]).mean()), "left_rmse_rad": float(np.sqrt((error[:, :7] ** 2).mean())),
            "right_mae_rad": float(np.abs(error[:, 7:]).mean()), "right_rmse_rad": float(np.sqrt((error[:, 7:] ** 2).mean()))}
    report["diagnostic_thresholds"] = {"wrist_rms_below_1cm": bool(np.all(np.array(report["wrist_position_rms_m"]) < .01)),
        "wrist_p95_below_2cm": bool(np.all(np.array(report["wrist_position_p95_m"]) < .02)),
        "wrist_angle_rms_below_10deg": bool(np.all(np.array(report["wrist_angle_rms_deg"]) < 10)),
        "left_hand_Bhat_mae_below_003rad": report["hand"]["B_hat"]["left_mae_rad"] < .03}
    hand.train()
    return report


def make_lr_schedule(args, start_step, total, resume):
    previous = resume.get("lr_schedule") if resume else None
    same_run = bool(resume) and Path(resume["config"]["training"]["output"]).resolve() == Path(args.output).resolve()
    if previous and same_run:
        requested = {"body_lr_scale": args.body_lr_scale, "hand_lr_scale": args.hand_lr_scale,
            "cosine_decay": args.cosine_decay, "lr_min_scale": args.lr_min_scale}
        if any(previous[key] != value for key, value in requested.items()) or previous["end_step"] != total:
            raise ValueError("Same-run LR schedule is fixed; repeat its flags or choose a new output directory")
        return dict(previous), True
    return {"base_body_lr": args.body_lr, "base_hand_lr": args.hand_lr,
        "body_lr_scale": args.body_lr_scale if resume else 1.0, "hand_lr_scale": args.hand_lr_scale if resume else 1.0,
        "cosine_decay": bool(args.cosine_decay and resume), "lr_min_scale": args.lr_min_scale,
        "start_step": start_step, "end_step": total, "source_checkpoint": args.resume,
        "source_step": start_step, "scale_reference": "original base LR; never loaded/current LR",
        "step_semantics": "start_step completed updates; first new update factor1, final update factor=min if cosine"}, False


def scheduled_lrs(schedule, step):
    factor = 1.0
    if schedule["cosine_decay"]:
        progress = min(1.0, max(0.0, (step - schedule["start_step"]) / max(schedule["end_step"] - schedule["start_step"] - 1, 1)))
        factor = schedule["lr_min_scale"] + (1 - schedule["lr_min_scale"]) * .5 * (1 + math.cos(math.pi * progress))
    return schedule["base_hand_lr"] * schedule["hand_lr_scale"] * factor, schedule["base_body_lr"] * schedule["body_lr_scale"] * factor, factor


def nominal_selection_score(metrics):
    return max(metrics["wrist_position_rms_m"]) / .01 + metrics["hand"]["B_hat"]["left_mae_rad"] / .03



def save_precise(output, body, hand, optimizer, config, hand_stats, body_stats, base_identity, warmstart_info, modules, fk_metadata, step, stage, joint_active, probe, best_nominal=None, best_only=False):
    prefixes = tuple(name + "." for name in modules)
    payload = {"format": "movens_interact_dual_expert_v1", "experiment": "apple_precision", "config": config,
        "body_trainable_modules": modules, "body_delta_state_dict": {k: v.detach().cpu() for k, v in body.state_dict().items() if k.startswith(prefixes)},
        "body_delta_semantics": "absolute module parameter values, not additive differences; includes Body initialization even before Body updates",
        "hand_state_dict": {k: v.detach().cpu() for k, v in hand.state_dict().items()}, "hand_normalization": hand_stats,
        "body_normalization": body_stats, "base_identity": base_identity, "base_checkpoint": base_identity["path"], "base_sha256": base_identity["recorded_sha256"],
        "warmstart": warmstart_info, "kinematics_metadata": fk_metadata, "data_provenance": config["data_provenance"],
        "step": step, "stage": stage, "optimizer_has_body_group": joint_active, "body_frozen_probe_reference": probe,
        "old_seven_tasks_evaluated": False, "closed_loop_success_measured": False}
    payload.update(lr_schedule=config["lr_schedule"], continuation=config["continuation"], best_nominal=best_nominal,
        body_normalization_sha256=base_identity["body_normalization_sha256"], context_prompt=config["data_provenance"]["context_prompt"])
    if best_only:
        temporary = output / "model_best_nominal.pt.tmp"
        torch.save(payload, temporary)
        temporary.replace(output / "model_best_nominal.pt")
        (output / "best_nominal_metrics.json").write_text(json.dumps(best_nominal, indent=2) + "\n")
        return
    temporary = output / "model_delta.pt.tmp"
    torch.save(payload, temporary)
    temporary.replace(output / "model_delta.pt")
    payload["optimizer_state_dict"] = optimizer.state_dict()
    payload["rng_state"] = {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(), "numpy": np.random.get_state(), "python": random.getstate()}
    temporary = output / "checkpoint_last.pt.tmp"
    torch.save(payload, temporary)
    temporary.replace(output / "checkpoint_last.pt")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Precision pilot using paired actual sim state / commanded reference")
    parser.add_argument("--manifest", default="data/apple_teacher/index.json")
    parser.add_argument("--context-dir", default="data/apple_context")
    parser.add_argument("--body-init", default="checkpoints/Movens-Interact/body_init.pt")
    parser.add_argument("--no-body-init", action="store_true")
    parser.add_argument("--base-checkpoint", default="checkpoints/Movens-Zero-XL/model.safetensors")
    parser.add_argument("--base-sha256", default="d1e6a44092938ce641d3bec43138df03e2cdb2bb0ebf1fb64840f1635de1dd3a")
    parser.add_argument("--body-config", default="configs/body.json")
    parser.add_argument("--scene-xml", default="assets/scene.xml")
    parser.add_argument("--body-normalization", default="configs/body_normalization.json")
    parser.add_argument("--output", default="runs/apple_precision")
    parser.add_argument("--resume")
    parser.add_argument("--hand-warmup-steps", type=int, default=300)
    parser.add_argument("--joint-steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--body-lr", type=float, default=5e-5)
    parser.add_argument("--hand-lr", type=float, default=2e-4)

    parser.add_argument("--body-lr-scale", type=float, default=1.0)
    parser.add_argument("--hand-lr-scale", type=float, default=1.0)
    parser.add_argument("--cosine-decay", action="store_true")
    parser.add_argument("--lr-min-scale", type=float, default=.1)
    parser.add_argument("--hand-state-min-width", type=float, default=.1)
    parser.add_argument("--wrist-position-weight", type=float, default=5000.)
    parser.add_argument("--wrist-orientation-weight", type=float, default=10.)
    parser.add_argument("--joint-error-weight", type=float, default=5.)
    parser.add_argument("--soft-limit-weight", type=float, default=10.)
    parser.add_argument("--hand-physical-weight", type=float, default=5.)
    parser.add_argument("--left-wrist-weight", type=float, default=2.)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=0, help="0 evaluates every window")
    parser.add_argument("--condition-dropout", type=float, default=.1)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--check-data-only", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.stride < 1 or args.hand_warmup_steps < 0 or args.joint_steps < 0 or args.hand_state_min_width < .1 or args.eval_batches < 0 or min(args.eval_every, args.save_every) < 1:
        raise ValueError("Invalid precision-pilot training settings")
    if not all(math.isfinite(value) and value > 0 for value in [args.body_lr_scale, args.hand_lr_scale, args.lr_min_scale]) or args.lr_min_scale > 1:
        raise ValueError("LR multipliers must be finite and positive; min-scale must be <=1")
    if not args.resume and (args.body_lr_scale != 1 or args.hand_lr_scale != 1 or args.cosine_decay):
        raise ValueError("LR scaling/decay is an explicit resume-only option")
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "checkpoint_last.pt").exists() and not args.resume and not args.check_data_only:
        raise FileExistsError("Select --resume or a new output directory")
    index_path = Path(args.manifest)
    index = json.loads(index_path.read_text())
    if not index["collection_complete"]:
        raise ValueError("Wait for complete paired teacher collection before training/cache identity checks")
    context_path = Path(args.context_dir) / "manifest.json"
    context_info = json.loads(context_path.read_text())
    if not context_info["complete"]:
        raise ValueError("Complete the frozen Qwen cache before training")
    index_hash = hashlib.sha256(index_path.read_bytes()).hexdigest()
    if context_info["teacher_index_sha256"] != index_hash or context_info["prompt"] != "Move the apple to the plate.":
        raise ValueError("Teacher/Qwen context identity mismatch")
    if context_info["body_config_sha256"] != sha256_file(args.body_config):
        raise ValueError("Body configuration differs from the frozen Qwen cache configuration")
    names, hand_names = index["body_joint_names"], index["hand_joint_names"]
    scene_hash = hashlib.sha256(Path(args.scene_xml).read_bytes()).hexdigest()
    if scene_hash != index["scene_sha256"]:
        raise ValueError("Teacher scene differs from FK scene")
    if context_info["scene_sha256"] != scene_hash:
        raise ValueError("Qwen cache and teacher scene differ")
    cached_episodes = {row["episode_index"]: row for row in context_info["episodes"]}
    for row in index["episodes"]:
        if row["success"] and row["teacher_success"]:
            cached = cached_episodes[row["episode_index"]]
            if cached["source_sha256"] != row["rgb_sha256"] or sha256_file(Path(args.context_dir) / cached["path"]) != cached["sha256"]:
                raise ValueError("Frozen context file checksum or source identity changed")
    body_stats = json.loads(Path(args.body_normalization).read_text())
    body_norm = MotionNormalizer.from_file(args.body_normalization)
    train_obs, train_ref, train_ids = [], [], []
    for row in index["episodes"]:
        if row["split"] == "train" and row["success"] and row["teacher_success"]:
            with np.load(index_path.parent / row["trajectory"], allow_pickle=False) as ep:
                train_obs.append(ep["hand_obs"])
                train_ref.append(ep["hand_ref"])
            train_ids.append(row["episode_index"])
    if not train_ids:
        raise ValueError("No successful training teacher")
    hand_stats = {"action": {}, "state": {}, "normalize_state": True, "use_norm_mask": False,
        "normalization_version": "bounded_state_width", "state_min_width_rad": args.hand_state_min_width,
        "raw_train_minmax": {}, "expanded_state_mask": [], "constant_dimensions": {},
        "canonical_joint_names": hand_names, "statistics_train_episode_ids": train_ids,
        "target_source": "exact paired simulator controller hand reference", "constant_action_execution_projection": "optional runtime protocol; raw sampled metrics remain unprojected"}
    for key, values in [("state", np.concatenate(train_obs)), ("action", np.concatenate(train_ref))]:
        lo, hi = values.min(0), values.max(0)
        hand_stats["raw_train_minmax"][key] = {"min": lo.tolist(), "max": hi.tolist()}
        if key == "state":
            expand = hi - lo < args.hand_state_min_width
            center = (lo + hi) * .5
            width = np.maximum(hi - lo, np.float32(args.hand_state_min_width))
            lo, hi = np.where(expand, center - width * .5, lo), np.where(expand, center + width * .5, hi)
            hand_stats["expanded_state_mask"] = expand.tolist()
        hand_stats[key] = {"q01": lo.tolist(), "q99": hi.tolist()}
        hand_stats["constant_dimensions"][key] = np.flatnonzero(np.abs(hi - lo) < 1e-8).tolist()
    (output / "hand_normalization.json").write_text(json.dumps(hand_stats, indent=2) + "\n")
    hand_norm = MotionNormalizer.from_file(output / "hand_normalization.json")
    train_data = SimWindows(args.manifest, args.context_dir, body_norm, hand_norm, "train", args.stride)
    val_data = SimWindows(args.manifest, args.context_dir, body_norm, hand_norm, "val", args.stride)
    nominal = SimWindows(args.manifest, args.context_dir, body_norm, hand_norm, "train", args.stride, episode_ids=[train_ids[0]])
    val_ids = [ep["episode_index"] for ep in val_data.episodes]
    if set(train_ids) & set(val_ids):
        raise ValueError("Teacher train/val episode leakage")
    fk = G1Kinematics(args.scene_xml, names).to(dtype=torch.float32)
    model = mujoco.MjModel.from_xml_path(args.scene_xml)
    joint_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in names])
    if np.any(joint_ids < 0):
        raise ValueError("FK/teacher joint names differ")
    check = {"dataset": str(index_path), "train_ids": train_ids, "val_ids": val_ids, "train_windows": len(train_data), "val_windows": len(val_data),
        "root_source": "sim_known_fixed_base", "targets": "obs[t] -> desired controller reference[t+1:t+31]", "reference_is_measured_trajectory": False,
        "context": "latest cache timestamp <= observed tick", "normalization_version": hand_stats["normalization_version"], "scene_sha256": scene_hash,
        "fk_metadata": fk.metadata, "generalization_scope": "same desired trajectory, held initial proprioception jitter"}
    for dataset in [train_data, val_data]:
        for selection in [0, len(dataset) - 1]:
            item = dataset[selection]
            assert item["context_timestamp"] <= item["timestamp"] + 1e-9
            with torch.no_grad():
                p, r = fk(torch.from_numpy(item["body_ref_joints"][None]), torch.from_numpy(item["root_xyz"]), torch.from_numpy(item["root_quat"]))
            assert p.shape == (1, 30, 2, 3) and r.shape == (1, 30, 2, 3, 3) and torch.isfinite(p).all()

    test = precise_batch([train_data[0]], "cpu")
    sigma = torch.full((1, 30), .5)
    sigma[:, :5] = 0
    noise = torch.randn(test["body_action"].shape, generator=torch.Generator().manual_seed(args.seed))
    noisy = (1 - sigma[..., None]) * test["body_action"] + sigma[..., None] * noise
    oracle_clean = noisy - sigma[..., None] * (noise - test["body_action"])
    velocity = (noise - test["body_action"] + .02).detach().requires_grad_(True)
    clean = noisy - sigma[..., None] * velocity
    q_test = physical_action(clean[..., :29], torch.tensor(body_norm.action_min[:29]), torch.tensor(body_norm.action_max[:29]))
    p_test, r_test = fk(q_test.float(), test["root_xyz"], test["root_quat"])
    with torch.no_grad():
        p_ref, r_ref = fk(test["body_ref_joints"], test["root_xyz"], test["root_quat"])
    test_valid = test["valid"].clone()
    test_valid[:, :5] = False
    integration_loss = (((p_test - p_ref).square().sum((-1, -2)) + (r_test - r_ref).square().sum((-1, -2, -3))) * test_valid).sum() / test_valid.sum()
    integration_loss.backward()
    assert torch.isfinite(velocity.grad).all() and velocity.grad[:, 5:].abs().max() > 0 and velocity.grad[:, :5].abs().max() == 0
    assert (oracle_clean - test["body_action"]).abs().max() < 2e-6
    check["cpu_auxiliary_integration_check"] = {"oracle_clean_max_abs_error": float((oracle_clean - test["body_action"]).abs().max()),
        "fk_velocity_gradient_finite_nonzero": True, "prefix_velocity_gradient_exact_zero": True,
        "scope": "synthetic velocity applied to real teacher labels; no model loading or parameter updates"}
    (output / "data_check.json").write_text(json.dumps(check, indent=2) + "\n")
    if args.check_data_only:
        print(json.dumps(check), flush=True)
        return
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("Precision pilot requires the prepared CUDA environment")
    torch.cuda.set_per_process_memory_fraction(.35, torch.device(args.device))
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()
    fk = fk.to(args.device)
    resume = torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
    warm = None if args.no_body_init or resume else torch.load(args.body_init, map_location="cpu", weights_only=False)
    body_config_file = json.loads(Path(args.body_config).read_text())
    body_config = body_config_file.get("model", body_config_file)
    initializer = resume if resume else warm
    architecture = {key: value for key, value in body_config.items() if key != "backbone"}
    if initializer and {key: value for key, value in initializer["config"]["body_config"].items() if key != "backbone"} != architecture:
        raise ValueError("Body architecture differs from the initialization checkpoint")
    base_path = Path(args.base_checkpoint)
    base_hash = sha256_file(base_path)
    if base_hash != args.base_sha256 or context_info["checkpoint_sha256"] != base_hash:
        raise ValueError("Base checkpoint or frozen Qwen cache identity mismatch")
    norm_hash = hashlib.sha256(Path(args.body_normalization).read_bytes()).hexdigest()
    if initializer and norm_hash != initializer["base_identity"]["body_normalization_sha256"]:
        raise ValueError("Original Body40 normalization changed")
    with base_path.open("rb") as source:
        header_length = struct.unpack("<Q", source.read(8))[0]
        header_hash = hashlib.sha256(source.read(header_length)).hexdigest()
    if initializer and (header_hash != initializer["base_identity"]["header_sha256"] or base_hash != initializer["base_identity"]["recorded_sha256"]):
        raise ValueError("Original Body checkpoint header changed")
    weights = {}
    with safe_open(str(base_path), framework="pt", device="cpu") as source:
        for name in source.keys():
            if name.startswith("action_header."):
                weights[name.removeprefix("action_header.")] = source.get_tensor(name)
    with torch.device("meta"):
        body = _ActionExpert(body_config)
    body.load_state_dict(weights, strict=True, assign=True)
    body.to(device=args.device, dtype=torch.bfloat16)
    body.action_proj_in.float().load_state_dict({name.removeprefix("action_proj_in."): value for name, value in weights.items() if name.startswith("action_proj_in.")}, strict=True)
    del weights
    modules = [f"transformer_blocks.{i}" for i in range(body.num_blocks - 3, body.num_blocks)] + ["action_proj_in", "obs_proj", "action_proj_out"]
    originally_trainable = {name for name, p in body.named_parameters() if p.requires_grad}
    for name in modules:
        body.get_submodule(name).float()
    if initializer:
        result = body.load_state_dict(initializer["body_delta_state_dict"], strict=False)
        if result.unexpected_keys:
            raise ValueError("Unexpected Body initialization keys")
        if any(not name.startswith(tuple(module + "." for module in modules)) for name in initializer["body_delta_state_dict"]):
            raise ValueError("Body initialization contains parameters outside the configured adapter modules")
    base_identity = {"path": str(base_path), "recorded_sha256": base_hash, "header_sha256": header_hash, "body_normalization_sha256": norm_hash}
    warmstart_info = {"path": args.resume or (args.body_init if warm else None), "step": initializer["step"] if initializer else 0, "body_only": not bool(resume), "hand_fresh": not bool(resume), "source": "body_init" if warm else ("resume" if resume else "original_movens_zero")}
    body.requires_grad_(False).eval()
    trainable_names = [name for name, p in body.named_parameters() if name in originally_trainable and name.startswith(tuple(module + "." for module in modules))]
    hand_config = {"hidden_dim": 256, "num_layers": 3, "num_heads": 8, "context_dim": 2048, "action_horizon": 30}
    hand = HandExpert(**hand_config).to(args.device)
    optimizer = torch.optim.Adam(hand.parameters(), lr=args.hand_lr)
    joint_active, start_step = False, 0
    if resume:
        if resume["hand_normalization"] != hand_stats or resume["data_provenance"]["teacher_index_sha256"] != index_hash or resume["data_provenance"]["context_manifest_sha256"] != hashlib.sha256(context_path.read_bytes()).hexdigest():
            raise ValueError("Resume normalization/data/context mismatch")
        if resume["body_trainable_modules"] != modules:
            raise ValueError("Resume trainable Body modules differ")
        previous = resume["config"]["training"]
        for key in ["seed", "batch_size", "stride", "hand_warmup_steps", "body_lr", "hand_lr", "condition_dropout", "cfg_scale", "wrist_position_weight", "wrist_orientation_weight", "joint_error_weight", "soft_limit_weight", "hand_physical_weight", "left_wrist_weight"]:
            if previous[key] != vars(args)[key]:
                raise ValueError(f"Resume training setting differs: {key}")
        hand.load_state_dict(resume["hand_state_dict"], strict=True)
        body.load_state_dict(resume["body_delta_state_dict"], strict=False)
        if resume["optimizer_has_body_group"]:
            for name, p in body.named_parameters():
                p.requires_grad_(name in trainable_names)
            optimizer.add_param_group({"params": [p for p in body.parameters() if p.requires_grad], "lr": args.body_lr})
            joint_active = True
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        start_step = resume["step"]
        warmstart_info.update(hand_fresh=False, resumed_from=args.resume)
    total = args.hand_warmup_steps + args.joint_steps
    if total <= start_step:
        raise ValueError("New total step count must exceed the checkpoint step")
    lr_schedule, inherited_schedule = make_lr_schedule(args, start_step, total, resume)
    loaded_lrs = [group["lr"] for group in optimizer.param_groups]

    hand_lr, body_lr, _ = scheduled_lrs(lr_schedule, start_step)
    optimizer.param_groups[0]["lr"] = hand_lr
    if joint_active:
        optimizer.param_groups[1]["lr"] = body_lr
    continuation = {"parent_checkpoint": str(Path(args.resume).resolve()) if args.resume else None,
        "resume_start_step": start_step, "new_total_step": total, "additional_steps": total - start_step,
        "loaded_optimizer_lrs": loaded_lrs, "initial_effective_hand_lr": hand_lr, "initial_effective_body_lr": body_lr,
        "fixed_schedule_inherited": inherited_schedule}
    best_nominal = resume.get("best_nominal") if resume and inherited_schedule else None
    if best_nominal and not (output / "model_best_nominal.pt").exists():
        raise FileNotFoundError("Same-run best weights are missing; cannot reconstruct an earlier best from last weights")
    bounds = {"body_lo": torch.tensor(body_norm.action_min[:29], device=args.device), "body_hi": torch.tensor(body_norm.action_max[:29], device=args.device),
        "hand_lo": torch.tensor(hand_norm.action_min, device=args.device), "hand_hi": torch.tensor(hand_norm.action_max, device=args.device),
        "limit_lo": torch.tensor(model.jnt_range[joint_ids, 0], dtype=torch.float32, device=args.device), "limit_hi": torch.tensor(model.jnt_range[joint_ids, 1], dtype=torch.float32, device=args.device)}
    config = {"experiment": "apple_precision", "training": vars(args), "body_config": body_config, "hand_config": hand_config,
        "body_trainable_modules": modules, "body_trainable_parameter_names": trainable_names,
        "body_trainable_parameter_count": sum(p.numel() for name, p in body.named_parameters() if name in trainable_names),
        "precision": "bf16 model autocast; trainable Body modules and Hand parameters FP32; input40 FP32; FK/physical losses and angular metrics explicitly outside autocast FP32",
        "optimizer": "Adam, weight_decay=0",
        "objective": "flow v=epsilon-A; clean_hat=noisy-sigma*v; FK wrist position m^2 and chordal orientation; physical joints/soft limits rad^2; common committed prefix excluded",
        "data_provenance": {"teacher_index": str(index_path), "teacher_index_sha256": index_hash, "data_revision": context_info["data_revision"],
            "context_manifest": str(context_path), "context_manifest_sha256": hashlib.sha256(context_path.read_bytes()).hexdigest(), "context_prompt": context_info["prompt"],
            "root_source": "sim_known_fixed_base", "root_known_in_sim": True, "real_robot_root_measured": False,
            "body_target_source": "paired simulator desired reference, derived from original measured body joints and validated only in this fixed scene", "hand_target_source": "same teacher exact hand reference",
            "contact_channels": "original isaac_leg_delta_v2 codec heuristic on controller reference; not measured contact labels",
            "train_episode_ids": train_ids, "val_episode_ids": val_ids, "nominal_train_fit_episode_ids": [train_ids[0]],
            "scene_xml": args.scene_xml, "scene_sha256": scene_hash, "progress_budget_seconds": 20,
            "validation_scope": "same desired trajectory held initial-state jitter, no unseen trajectory generalization claim", "old_task_replay": False}}
    config.update(lr_schedule=lr_schedule, continuation=continuation,
        nominal_selection={"formula": "max(wrist_position_rms_m)/0.01 + left_hand_Bhat_mae_rad/0.03", "split": "nominal_train_fit", "held_val_used_for_selection": False})
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "base_identity.json").write_text(json.dumps(base_identity, indent=2) + "\n")
    probe_batch = precise_batch([val_data[0]], args.device)
    probe_gen = torch.Generator(device=args.device).manual_seed(args.seed + 900000)
    probe_noise = torch.randn(probe_batch["body_action"].shape, device=args.device, generator=probe_gen)
    probe_sigma = torch.full((1, 30), .5, device=args.device)
    probe_sigma[:, :5] = 0
    probe_noisy = (1 - probe_sigma[..., None]) * probe_batch["body_action"] + probe_sigma[..., None] * probe_noise
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        probe = body(probe_noisy, probe_sigma * 1000, probe_batch["context"], probe_batch["body_state"], probe_batch["progress"], context_mask=probe_batch["context_mask"]).float().cpu()
    if resume:
        if not joint_active and not torch.equal(probe, resume["body_frozen_probe_reference"]):
            raise ValueError("Frozen warmstart Body changed on resume")
        probe = resume["body_frozen_probe_reference"]
    weights40 = torch.ones(40, device=args.device) / 40
    for selection, multiplier in [(slice(29, 31), 3), (slice(31, 37), 10), (slice(37, 38), 15), (slice(38, 40), .1)]:
        weights40[selection] *= multiplier
    wrist_weights = torch.tensor([args.left_wrist_weight, 1.0], device=args.device)
    if resume and "rng_state" in resume:
        torch.set_rng_state(resume["rng_state"]["torch"])
        torch.cuda.set_rng_state_all(resume["rng_state"]["cuda"])
        np.random.set_state(resume["rng_state"]["numpy"])
        random.setstate(resume["rng_state"]["python"])
    began = time.perf_counter()
    hand.train()
    total = args.hand_warmup_steps + args.joint_steps

    if resume and best_nominal is None:
        baseline = evaluate_precise(body, hand, fk, nominal, args, bounds, "nominal_train_fit")
        baseline.update(step=start_step, stage="resume_baseline")
        score = nominal_selection_score(baseline)
        best_nominal = {"score": score, "step": start_step, "formula": config["nominal_selection"]["formula"], "metrics": baseline}
        with (output / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(baseline) + "\n")
        save_precise(output, body, hand, optimizer, config, hand_stats, body_stats, base_identity, warmstart_info, modules, fk.metadata, start_step, "resume_baseline", joint_active, probe, best_nominal, best_only=True)
        print(json.dumps({"resume_baseline_best_score": score, "source_step": start_step, "lr_schedule": lr_schedule}), flush=True)
    for step in range(start_step, total):
        if step >= args.hand_warmup_steps and not joint_active:
            for name, p in body.named_parameters():
                p.requires_grad_(name in trainable_names)
            optimizer.add_param_group({"params": [p for p in body.parameters() if p.requires_grad], "lr": args.body_lr})
            joint_active = True

        hand_lr, body_lr, lr_factor = scheduled_lrs(lr_schedule, step)
        optimizer.param_groups[0]["lr"] = hand_lr
        if joint_active:
            optimizer.param_groups[1]["lr"] = body_lr
        stage = "joint_precision" if joint_active else "hand_warmup"
        selected = np.random.default_rng(args.seed + step * 104729).integers(len(train_data), size=args.batch_size)
        batch = precise_batch([train_data[int(i)] for i in selected], args.device)
        generator = torch.Generator(device=args.device).manual_seed(args.seed + step * 1000003)
        delay = torch.minimum(torch.randint(0, 11, (args.batch_size,), device=args.device, generator=generator), batch["valid"].sum(1) - 1)
        prefix = torch.arange(30, device=args.device)[None] < delay[:, None]
        valid = batch["valid"] & ~prefix
        sigma_b = torch.rand(args.batch_size, 1, device=args.device, generator=generator).expand(-1, 30).masked_fill(prefix, 0)
        sigma_h = torch.rand(args.batch_size, 1, device=args.device, generator=generator).expand(-1, 30).masked_fill(prefix, 0)
        eps_b = torch.randn(batch["body_action"].shape, device=args.device, generator=generator)
        eps_h = torch.randn(batch["hand_action"].shape, device=args.device, generator=generator)
        noisy_b = (1 - sigma_b[..., None]) * batch["body_action"] + sigma_b[..., None] * eps_b
        noisy_h = (1 - sigma_h[..., None]) * batch["hand_action"] + sigma_h[..., None] * eps_h
        context_mask = batch["context_mask"].clone()
        context_mask[torch.rand(args.batch_size, device=args.device, generator=generator) < args.condition_dropout] = False
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            velocity_h = hand(noisy_h, sigma_h, batch["context"], batch["hand_state"], batch["body_action"].detach(), context_mask, batch["valid"])
            velocity_b = body(noisy_b, sigma_b * 1000, batch["context"], batch["body_state"], batch["progress"], context_mask=context_mask) if joint_active else None
        with torch.autocast("cuda", enabled=False):
            hand_flow = (((velocity_h.float() - (eps_h - batch["hand_action"])) ** 2).mean(-1) * valid).sum() / valid.sum()
            clean_h = noisy_h.float() - sigma_h[..., None] * velocity_h.float()
            raw_h = physical_action(clean_h, bounds["hand_lo"], bounds["hand_hi"])
            hand_physical = (((raw_h - batch["hand_ref"]) ** 2).mean(-1) * valid).sum() / valid.sum()
            body_flow = hand_flow.new_zeros(())
            position_loss = hand_flow.new_zeros(())
            orientation_loss = hand_flow.new_zeros(())
            joint_loss = hand_flow.new_zeros(())
            limit_loss = hand_flow.new_zeros(())
            if joint_active:
                body_flow = (((velocity_b.float() - (eps_b - batch["body_action"])) ** 2 * weights40).sum(-1) * valid).sum() / valid.sum()
                clean_b = noisy_b.float() - sigma_b[..., None] * velocity_b.float()
                joints = physical_action(clean_b[..., :29], bounds["body_lo"], bounds["body_hi"])
                position, rotation = fk(joints.float(), batch["root_xyz"], batch["root_quat"])
                with torch.no_grad():
                    ref_position, ref_rotation = fk(batch["body_ref_joints"], batch["root_xyz"], batch["root_quat"])
                denominator = valid.sum() * wrist_weights.sum()
                position_loss = (((position - ref_position).square().sum(-1) * wrist_weights) * valid[..., None]).sum() / denominator
                orientation_loss = (((rotation - ref_rotation).square().sum((-1, -2)) * .5 * wrist_weights) * valid[..., None]).sum() / denominator
                joint_loss = (((joints - batch["body_ref_joints"]).square().mean(-1)) * valid).sum() / valid.sum()
                violation = torch.relu(bounds["limit_lo"] - joints).square() + torch.relu(joints - bounds["limit_hi"]).square()
                limit_loss = (violation.mean(-1) * valid).sum() / valid.sum()
            loss = hand_flow + args.hand_physical_weight * hand_physical + body_flow + args.wrist_position_weight * position_loss + args.wrist_orientation_weight * orientation_loss + args.joint_error_weight * joint_loss + args.soft_limit_weight * limit_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite precision loss at step {step}")
        loss.backward()
        hand_grad = torch.nn.utils.clip_grad_norm_(hand.parameters(), 1.0)
        body_grad = torch.nn.utils.clip_grad_norm_([p for p in body.parameters() if p.requires_grad], 1.0) if joint_active else torch.tensor(0.)
        optimizer.step()
        completed = step + 1
        record = {"step": completed, "stage": stage, "loss": float(loss.detach()), "hand_flow": float(hand_flow.detach()), "hand_physical_mse": float(hand_physical.detach()),
            "body_flow": float(body_flow.detach()), "wrist_position_m2": float(position_loss.detach()), "wrist_orientation_chordal": float(orientation_loss.detach()),
            "joint_physical_rad2": float(joint_loss.detach()), "soft_limit_rad2": float(limit_loss.detach()), "body_grad_norm": float(body_grad), "hand_grad_norm": float(hand_grad), "elapsed_seconds": time.perf_counter() - began}
        record.update(hand_lr=hand_lr, body_lr=body_lr if joint_active else 0.0, lr_decay_factor=lr_factor)
        with (output / "loss.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        if completed == 1 or completed % 25 == 0:
            print(json.dumps(record), flush=True)
        if completed == args.hand_warmup_steps:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                after = body(probe_noisy, probe_sigma * 1000, probe_batch["context"], probe_batch["body_state"], probe_batch["progress"], context_mask=probe_batch["context_mask"]).float().cpu()
            regression = {"step": completed, "warmstart_body_bitwise_equal": torch.equal(after, probe), "max_abs_difference": float((after - probe).abs().max())}
            (output / "frozen_body_regression.json").write_text(json.dumps(regression, indent=2) + "\n")
            if not regression["warmstart_body_bitwise_equal"]:
                raise RuntimeError("Frozen warmstart Body changed during Hand warmup")
        if completed % args.eval_every == 0 or completed == total:
            for label, dataset in [("nominal_train_fit", nominal), ("held_initial_jitter_val", val_data)]:
                metrics = evaluate_precise(body, hand, fk, dataset, args, bounds, label)
                metrics.update(step=completed, stage=stage)
                if label == "nominal_train_fit":
                    score = nominal_selection_score(metrics)
                    metrics["nominal_selection_score"] = score
                    if best_nominal is None or score < best_nominal["score"]:
                        best_nominal = {"score": score, "step": completed, "formula": config["nominal_selection"]["formula"], "metrics": metrics}
                        save_precise(output, body, hand, optimizer, config, hand_stats, body_stats, base_identity, warmstart_info, modules, fk.metadata, completed, stage, joint_active, probe, best_nominal, best_only=True)
                with (output / "metrics.jsonl").open("a") as handle:
                    handle.write(json.dumps(metrics) + "\n")
                print(json.dumps({"evaluation": metrics}), flush=True)
        if completed % args.save_every == 0 or completed in (args.hand_warmup_steps, total):

            save_precise(output, body, hand, optimizer, config, hand_stats, body_stats, base_identity, warmstart_info, modules, fk.metadata, completed, stage, joint_active, probe, best_nominal)
    summary = {"experiment": "apple_precision", "steps": total, "root_source": "sim_known_fixed_base", "paired_reference_targets": True,
        "body_trainable_modules": modules, "wall_seconds": time.perf_counter() - began, "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
        "weights": str(output / "model_delta.pt"), "closed_loop_task_success_measured": False, "old_seven_tasks_evaluated": False}
    summary.update(parent_checkpoint=continuation["parent_checkpoint"], resume_start_step=start_step, additional_steps=total-start_step,
        lr_schedule=lr_schedule, best_nominal_score=best_nominal["score"] if best_nominal else None,
        best_nominal_step=best_nominal["step"] if best_nominal else None, best_nominal_snapshot=str(output / "model_best_nominal.pt"))
    (output / "training_report.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
