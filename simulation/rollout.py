import argparse
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from simulation.common import AppleSimulation, support_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/Movens-Interact/adapter.pt")
    parser.add_argument("--base-checkpoint")
    parser.add_argument("--backbone")
    parser.add_argument("--scene-setup", default="assets/scene.json")
    parser.add_argument("--robot-config")
    parser.add_argument("--reset-index", type=Path)
    parser.add_argument("--reset-episode", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--render-fps", type=float, default=6)
    parser.add_argument("--output", type=Path, default=Path("outputs/rollout"))
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--interface-smoke", action="store_true")
    args = parser.parse_args()
    if args.seconds <= 0 or args.render_fps <= 0:
        parser.error("seconds and render-fps must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    reset = None
    if args.reset_index:
        index = json.loads(args.reset_index.read_text())
        rows = [row for row in index["episodes"] if row["episode_index"] == args.reset_episode]
        if len(rows) != 1:
            raise ValueError("Reset episode must occur exactly once in the index")
        reset = rows[0]["reset_protocol"]
    sim = AppleSimulation(args.scene_setup, args.robot_config, reset)
    model = None
    model_load_seconds = 0.0
    if not args.interface_smoke:
        from movens_interact.model import MovensInteractModel

        started = time.perf_counter()
        model = MovensInteractModel.from_pretrained(args.checkpoint, base_checkpoint=args.base_checkpoint, backbone=args.backbone, device=args.device)
        model_load_seconds = time.perf_counter() - started
    writer = None
    if not args.no_video:
        if not shutil.which("ffmpeg"):
            raise FileNotFoundError("FFmpeg is required for video output; install it or use --no-video")
        writer = subprocess.Popen([shutil.which("ffmpeg"), "-y", "-loglevel", "error", "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", "960x360", "-framerate", str(args.render_fps), "-i", "-", "-an", "-c:v", "libx264", "-crf", "22", "-pix_fmt", "yuv420p", str(args.output / "rollout.mp4")], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    frame = 0
    total_frames = int(round(args.seconds * 50))
    next_render = 0.0
    previous = None
    plans, samples, records, observations = [], [], [], []
    failure = None
    tracking = np.zeros(43)
    clipping = np.zeros(43, dtype=np.int64)
    snapshots = 0
    wall_start = time.perf_counter()
    while frame < total_frames and failure is None:
        body, hand = sim.observe()
        plan_id = len(plans)
        observation_time = float(sim.data.time)
        if args.interface_smoke:
            plan = {"body_absolute": np.repeat(body, 30, axis=0), "hand_absolute": np.repeat(hand, 30, axis=0), "plan_id": plan_id, "first_physical_frame": frame + 1, "prefix_length": 0, "total_inference_seconds": 0.0}
        else:
            ego = sim.ego_rgb()
            Image.fromarray(ego).save(args.output / f"observed_ego_{plan_id:03d}.png")
            plan = model.predict(ego, body, hand, frame, previous_plan=previous, seed=args.seed)
            observations.append({"plan_id": plan_id, "observation_timestamp": observation_time, "ego_sha256": hashlib.sha256(ego.tobytes()).hexdigest(), "source": "current_live_sim_ego", "future_frames_used": False})
        if not np.isfinite(plan["body_absolute"]).all() or not np.isfinite(plan["hand_absolute"]).all():
            failure = "non_finite_policy_output"
            break
        previous = plan
        np.savez_compressed(args.output / f"plan_{plan_id:03d}.npz", **{key: np.asarray(value) for key, value in plan.items() if value is not None})
        summary = {"plan_id": plan_id, "first_physical_frame": frame + 1, "simulation_time": observation_time, "horizon": 30, "execute": 20, "prefix_length": int(plan.get("prefix_length", 0)), "body_root_channels_executed": False}
        for key in ["context_seconds", "body_seconds", "hand_seconds", "total_inference_seconds", "prefix_body_max_abs_error", "prefix_hand_max_abs_error"]:
            summary[key] = float(plan.get(key, 0))
        summary["exceeds_200ms_remaining_buffer"] = bool(summary["prefix_length"] and summary["total_inference_seconds"] > .2)
        plans.append(summary)
        for offset in range(min(20, total_frames - frame)):
            target = np.r_[plan["body_absolute"][offset, :29], plan["hand_absolute"][offset]]
            sample = sim.step(target)
            sample["plan_id"], sample["plan_offset"] = plan_id, offset
            samples.append(sample)
            frame += 1
            tracked = np.r_[sample["body_abs"][:29], sample["hand_state"]]
            tracking += (tracked - sample["target"]) ** 2
            clipping += sample["clipped"]
            records.append(np.r_[sample["time"], plan_id, offset, tracked, sample["apple_xyz"], sample["plate_xy_distance"], [int(sample["contacts"][key]) for key in ["left_fingers", "left_wrist", "table", "plate"]]])
            failure = sample["failure"]
            if sample["apple_xyz"][2] < sim.scene_info["table_top_z"] - .15:
                failure = "apple_fell_below_table"
            if writer is not None and sample["time"] + 1e-9 >= next_render:
                image = Image.fromarray(np.concatenate([sim.render(), sim.ego_rgb()], axis=1))
                draw = ImageDraw.Draw(image)
                draw.rectangle([0, 0, 960, 42], fill="white")
                draw.text((8, 5), "INTERFACE SMOKE | NO POLICY" if args.interface_smoke else "LIVE EGO POLICY | FIXED BASE | PAUSED INFERENCE", fill="black")
                draw.text((8, 23), f"t={sample['time']:.2f}s plan={plan_id} seed={args.seed}", fill="black")
                writer.stdin.write(np.asarray(image).tobytes())
                if snapshots < 8 and sample["time"] >= snapshots * args.seconds / 8:
                    image.save(args.output / f"snapshot_{snapshots:02d}.png")
                    snapshots += 1
                next_render += 1 / args.render_fps
            if failure:
                break
        (args.output / "plans.json").write_text(json.dumps(plans, indent=2) + "\n")
        print(json.dumps({"plan_id": plan_id, "simulation_time": float(sim.data.time), "inference_seconds": summary["total_inference_seconds"], "failure": failure}), flush=True)
    if writer is not None:
        writer.stdin.close()
        error = writer.stderr.read().decode(errors="replace")
        if writer.wait() != 0:
            raise RuntimeError(error)
    result = support_summary(samples, sim.baseline_apple_z, sim.apple_weight_N)
    if not args.interface_smoke and failure is None and not result["support_verified_success"]:
        failure = "no_sustained_grasp_detected" if not result["grasped_and_lifted"] else "plate_support_not_verified"
    rmse = np.sqrt(tracking / max(len(samples), 1))
    report = {"evaluation_type": "interface_smoke_not_policy" if args.interface_smoke else "live_sim_ego_closed_loop_policy_diagnostic",
        "checkpoint": None if args.interface_smoke else args.checkpoint, "seed": args.seed, "sampling_steps": 10,
        "scene_setup": args.scene_setup, "robot_config": args.robot_config, "scene": sim.scene_info,
        "source_scene_sha256": sim.scene_info.get("source_scene_sha256"), "portable_scene_sha256": sim.scene_info["scene_sha256"],
        "reset_episode": args.reset_episode if args.reset_index else None, "reset_protocol": sim.reset_protocol,
        "initial_world": sim.initial_world, "initial_world_after_settle": sim.post_settle_world,
        "initial_apple_xyz_after_settle": sim.post_settle_world["apple_xyz"], "initial_plate_xyz_after_settle": sim.post_settle_world["plate_xyz"],
        "model_load_seconds": model_load_seconds, "simulation_seconds": float(sim.data.time), "executed_frames": frame,
        "wall_seconds_excluding_model_load": time.perf_counter() - wall_start,
        "body_tracking_rmse_rad": float(np.sqrt(np.mean(rmse[:29] ** 2))), "hand_tracking_rmse_rad": float(np.sqrt(np.mean(rmse[29:] ** 2))),
        "tracking_rmse_by_joint": dict(zip(sim.joint_names, rmse.tolist())), "joint_limit_clipping_count_by_joint": dict(zip(sim.joint_names, clipping.tolist())),
        "causal_observations": observations, "plan_count": len(plans), "plans": plans,
        "inference_seconds": [plan["total_inference_seconds"] for plan in plans],
        "would_exceed_remaining_200ms_buffer_count": sum(plan["exceeds_200ms_remaining_buffer"] for plan in plans),
        "predicted_horizon": 30, "execution_horizon": 20, "retained_prefix": 10, "fps": 50,
        "fixed_base_root_channels_consumed": False, "ground_truth_object_poses_used_as_policy_inputs": False,
        "inference_execution": "Simulation pauses during serial visual encoding and Body/Hand prediction; this is not real-time deployment.",
        "apple_free_body": True, "apple_attached_or_forced_after_reset": False, "failure": failure,
        "support": result, "support_verified_success": None if args.interface_smoke else result["support_verified_success"],
        "diagnostic_task_success": None if args.interface_smoke else result["support_verified_success"],
        "boundary": "A fixed-base position-servo simulation with an approximate camera. It does not validate SONIC whole-body tracking or real-world task success."}
    (args.output / "rollout_report.json").write_text(json.dumps(report, indent=2) + "\n")
    columns = ["sim_time", "plan_id", "plan_offset"] + sim.joint_names + ["apple_x", "apple_y", "apple_z", "plate_xy_distance", "left_fingers", "left_wrist", "table", "plate"]
    np.savez_compressed(args.output / "policy_measurements.npz", records=np.asarray(records).reshape((-1, len(columns))), targets=np.asarray([sample["target"] for sample in samples]), joint_names=np.asarray(sim.joint_names), record_columns=np.asarray(columns))
    np.savez_compressed(args.output / "physical_states.npz", timestamps=np.asarray([sample["time"] for sample in samples]), qpos=np.asarray([sample["qpos"] for sample in samples]), qvel=np.asarray([sample["qvel"] for sample in samples]), ctrl=np.asarray([sample["ctrl"] for sample in samples]), final_qpos=sim.data.qpos.copy(), final_qvel=sim.data.qvel.copy(), final_ctrl=sim.data.ctrl.copy())
    force_columns = ["simulation_time", "force_evaluation_time"] + [f"{kind}_f{axis}" for kind in ["plate", "table", "robot", "other"] for axis in "xyz"] + [f"{kind}_contact" for kind in ["plate", "table", "robot", "other"]] + ["constraint_fx", "constraint_fy", "constraint_fz"]
    force_rows = [np.concatenate([np.asarray([sample["time"], sample["force_evaluation_time"]]), *[sample["forces"][key] for key in ["plate", "table", "robot", "other"]], np.asarray([int(sample["contacts"][key]) for key in ["plate", "table", "robot", "other"]]), sample["constraint_force"]]) for sample in samples]
    np.savez_compressed(args.output / "support_force_measurements.npz", records=np.asarray(force_rows).reshape((-1, len(force_columns))), columns=np.asarray(force_columns), apple_weight_N=np.asarray(sim.apple_weight_N))
    sim.close()
    print(json.dumps({"support_verified_success": report["support_verified_success"], "simulation_seconds": report["simulation_seconds"], "failure": failure}))


if __name__ == "__main__":
    main()
