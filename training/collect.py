import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from simulation.common import AppleSimulation, support_summary


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def simulate(prepared, reset, args, episode_index, split, seed, render):
    simulator = AppleSimulation(scene_setup=args.scene_setup, robot_config=args.robot_config, reset_protocol=reset)
    try:
        body_names, hand_names = list(simulator.body_names), list(simulator.hand_names)
        if prepared["body_joint_names"].tolist() != body_names or prepared["hand_joint_names"].tolist() != hand_names:
            raise ValueError("Prepared reference and simulator joint order differ")
        reference = np.clip(prepared["references43"], simulator.joint_limits[:, 0], simulator.joint_limits[:, 1])
        body, hands, samples, apple_xyz = [], [], [], []
        images, image_times, image_frames = [], [], []
        current, hand = simulator.observe()
        root = current[0, 29:36].copy()
        for frame, timestamp in enumerate(prepared["timestamps"]):
            current, hand = simulator.observe()
            body.append(current[0].copy())
            hands.append(hand[0].copy())
            apple_xyz.append(simulator.data.xpos[simulator.apple_id].copy())
            if render and frame % 5 == 0:
                images.append(simulator.ego_rgb())
                image_times.append(timestamp)
                image_frames.append(frame)
            if frame + 1 < len(prepared["timestamps"]):
                samples.append(simulator.step(reference[frame + 1]))
        summary = support_summary(samples, simulator.baseline_apple_z, simulator.apple_weight_N)
        success = bool(summary["support_verified_success"])
        body, hands = np.asarray(body, dtype=np.float32), np.asarray(hands, dtype=np.float32)
        if not np.isfinite(body).all() or not np.isfinite(hands).all():
            raise ValueError("Nonfinite simulator observation")
        output = Path(args.output)
        trajectory_path = output / f"episodes/episode_{episode_index:06d}.npz"
        rgb_path = output / f"rgb/episode_{episode_index:06d}.npz"
        if render:
            with np.load(trajectory_path, allow_pickle=False) as previous:
                if not np.array_equal(previous["body_obs_abs"], body) or not np.array_equal(previous["hand_obs"], hands):
                    raise RuntimeError("Rendering changed the paired teacher physics")
        np.savez_compressed(trajectory_path, timestamps=prepared["timestamps"], body_obs_abs=body,
            body_ref_abs=np.c_[reference[:, :29], np.repeat(root[None], len(reference), axis=0)].astype(np.float32),
            hand_obs=hands, hand_ref=reference[:, 29:].astype(np.float32),
            body_joint_names=np.asarray(body_names), hand_joint_names=np.asarray(hand_names),
            root_source=np.asarray("sim_known_fixed_base"), apple_xyz=np.asarray(apple_xyz),
            interval_contacts=np.asarray([[sample["contacts"][key] for key in ["plate", "table", "robot", "other"]] for sample in samples]),
            interval_plate_force_z=np.asarray([sample["forces"]["plate"][2] for sample in samples]))
        if render:
            np.savez_compressed(rgb_path, rgb=np.asarray(images, dtype=np.uint8), timestamps=np.asarray(image_times, dtype=np.float64), frame_indices=np.asarray(image_frames, dtype=np.int64))
        row = {"episode_index": episode_index, "split": split,
            "trajectory": str(trajectory_path.relative_to(output)), "rgb": str(rgb_path.relative_to(output)),
            "trajectory_sha256": sha256_file(trajectory_path), "rgb_sha256": sha256_file(rgb_path) if render else None,
            "frames": len(body), "success": success, "teacher_success": success, "rgb_ready": render,
            "rgb_frames": len(images), "root_source": "sim_known_fixed_base",
            "control_profile": "zoh_50hz_target_next_tick", "perturbation_seed": seed, "reset_protocol": reset}
        report = dict(summary, episode_index=episode_index, split=split, seed=seed, render_rgb=render,
            reference_joint_clip_count=int(np.count_nonzero(np.abs(reference - prepared["references43"]) > 1e-10)))
        (output / f"reports/episode_{episode_index:06d}_{'rgb' if render else 'preflight'}.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"episode_index": episode_index, "render_rgb": render, "teacher_success": success}), flush=True)
        return row
    finally:
        simulator.close()


def main():
    parser = argparse.ArgumentParser(description="Collect paired actual simulator observations and validated controller references")
    parser.add_argument("--prepared", default="data/apple_prepared/manifest.json")
    parser.add_argument("--scene-setup", default="assets/scene.json")
    parser.add_argument("--robot-config", default="assets/robot.json")
    parser.add_argument("--output", default="data/apple_teacher")
    parser.add_argument("--train-jitter-seeds", type=int, nargs="+", default=[20261101, 20261102, 20261103])
    parser.add_argument("--val-jitter-seed", type=int, default=20261104)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    if (output / "index.json").exists():
        raise FileExistsError("Choose a fresh collection output directory")
    for directory in [output, output / "episodes", output / "rgb", output / "reports"]:
        directory.mkdir(parents=True, exist_ok=True)
    prepared_manifest = Path(args.prepared)
    preparation = json.loads(prepared_manifest.read_text())
    prepared_path = prepared_manifest.parent / preparation["prepared"]
    if sha256_file(prepared_path) != preparation["prepared_sha256"]:
        raise ValueError("Prepared reference checksum changed")
    with np.load(prepared_path, allow_pickle=False) as archive:
        prepared = {name: archive[name] for name in archive.files}
    scene_path = Path(args.scene_setup)
    scene = json.loads(scene_path.read_text())
    scene_xml = scene_path.parent / scene["scene"]
    robot = json.loads(Path(args.robot_config).read_text())
    nominal_reset = robot["reset_protocol"]
    if np.max(np.abs(np.asarray(nominal_reset["qpos43"]) - prepared["initial_qpos43"])) > 1e-7:
        raise ValueError("Published nominal reset differs from the prepared source initial joint state")
    probe = AppleSimulation(scene_setup=args.scene_setup, robot_config=args.robot_config)
    try:
        limits = probe.joint_limits.copy()
        names = probe.joint_names
        clipped_first = np.clip(prepared["references43"][0], limits[:, 0], limits[:, 1])
        if np.max(np.abs(np.asarray(nominal_reset["ctrl43"]) - clipped_first)) > 1e-7:
            raise ValueError("Published initial control differs from the clipped reference")
    finally:
        probe.close()
    if set(args.train_jitter_seeds) & {args.val_jitter_seed} or len(set(args.train_jitter_seeds)) != len(args.train_jitter_seeds):
        raise ValueError("Train and validation perturbation seeds must be distinct")
    plans = [(0, "train", None)] + [(i + 1, "train", seed) for i, seed in enumerate(args.train_jitter_seeds)]
    plans.append((len(plans), "val", args.val_jitter_seed))
    index = {"format": "movens_interact_paired_sim_teacher_v1", "collection_complete": False,
        "data_revision": f"sim_teacher_from_apple_{preparation['revision']}_episode0",
        "scene_xml": str(scene_xml), "scene_sha256": sha256_file(scene_xml),
        "scene_setup": args.scene_setup, "scene_setup_sha256": sha256_file(scene_path),
        "robot_config": args.robot_config, "robot_config_sha256": sha256_file(args.robot_config),
        "prepared_manifest": args.prepared, "prepared_manifest_sha256": sha256_file(prepared_manifest),
        "fps": 50, "physics_fps": 500, "ego_fps": 10, "root_source": "sim_known_fixed_base",
        "body_joint_names": prepared["body_joint_names"].tolist(), "hand_joint_names": prepared["hand_joint_names"].tolist(),
        "body_reference_source": preparation["body_reference_source"], "hand_reference_source": preparation["hand_reference_source"],
        "reference_timing": "Observe state[t], then apply desired reference[t+1] for ten 0.002-second physics steps; train against reference[t+1:t+31].",
        "rgb_protocol": "Render actual current observation ticks 0,5,10,...; cache lookup never selects a future timestamp.",
        "split_boundary": "The held split changes only initial robot joint perturbation. All episodes share one trajectory and fixed object positions/material; this is not new-trajectory or placement generalization.",
        "camera_calibrated_to_real": False, "reset_protocol": nominal_reset, "episodes": []}
    for episode_index, split, seed in plans:
        reset = copy.deepcopy(nominal_reset)
        delta = np.zeros(43)
        if seed is not None:
            generator = np.random.default_rng(seed)
            for joint, name in enumerate(names):
                if joint >= 29:
                    delta[joint] = generator.uniform(-.002, .002)
                elif "waist" in name or any(word in name for word in ["shoulder", "elbow", "wrist"]):
                    delta[joint] = generator.uniform(-.005, .005)
        initial = np.asarray(nominal_reset["qpos43"]).copy()
        changed = delta != 0
        initial[changed] = np.clip(initial[changed] + delta[changed], limits[changed, 0], limits[changed, 1])
        reset.update(qpos43=initial.tolist(), perturbation_seed=seed, requested_joint_delta43=delta.tolist(),
            effective_joint_delta43=(initial - nominal_reset["qpos43"]).tolist(), objects_and_root_perturbed=False)
        row = simulate(prepared, reset, args, episode_index, split, seed, False)
        index["episodes"].append(row)
        (output / "index.json").write_text(json.dumps(index, indent=2) + "\n")
        if not row["teacher_success"]:
            raise RuntimeError("Teacher preflight failed; inspect the saved report before changing the experiment")
    if args.preflight_only:
        return
    for position, (episode_index, split, seed) in enumerate(plans):
        row = simulate(prepared, index["episodes"][position]["reset_protocol"], args, episode_index, split, seed, True)
        if not row["teacher_success"]:
            raise RuntimeError("Rendered teacher failed the same physical success criteria")
        index["episodes"][position] = row
        (output / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    index.update(collection_complete=True, successful_train_episodes=len(args.train_jitter_seeds) + 1,
        successful_val_episodes=1, scene_unchanged=sha256_file(scene_xml) == index["scene_sha256"])
    if not index["scene_unchanged"]:
        raise ValueError("Scene changed during collection")
    (output / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(json.dumps({"collection_complete": True, "train_episodes": index["successful_train_episodes"], "val_episodes": 1}), flush=True)


if __name__ == "__main__":
    main()
