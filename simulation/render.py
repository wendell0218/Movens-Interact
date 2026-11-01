import argparse
import json
import shutil
import subprocess
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from simulation.common import AppleSimulation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("outputs/rollout"))
    parser.add_argument("--scene-setup")
    parser.add_argument("--view", choices=["overview", "ego", "close", "topdown"], default="overview")
    parser.add_argument("--frame", type=int)
    parser.add_argument("--fps", type=float, default=10)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.loads((args.run_dir / "rollout_report.json").read_text())
    states = np.load(args.run_dir / "physical_states.npz", allow_pickle=False)
    timestamps = states["timestamps"]
    if len(timestamps) == 0:
        raise ValueError("Rollout has no saved physical states")
    sim = AppleSimulation(args.scene_setup or report["scene_setup"], report.get("robot_config"))
    sim.model.vis.global_.offwidth = max(args.width, sim.model.vis.global_.offwidth)
    sim.model.vis.global_.offheight = max(args.height, sim.model.vis.global_.offheight)
    renderer = mujoco.Renderer(sim.model, height=args.height, width=args.width)
    if args.frame is not None:
        indices = [int(np.arange(len(timestamps))[args.frame])]
        output = args.output or args.run_dir / f"state_{indices[0]:04d}_{args.view}.png"
    else:
        if args.fps <= 0:
            parser.error("fps must be positive")
        indices = np.minimum(np.searchsorted(timestamps, np.arange(timestamps[0], timestamps[-1] + 1e-9, 1 / args.fps)), len(timestamps) - 1)
        output = args.output or args.run_dir / f"{args.view}.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    if args.frame is None:
        if not shutil.which("ffmpeg"):
            raise FileNotFoundError("FFmpeg is required for video rendering")
        writer = subprocess.Popen([shutil.which("ffmpeg"), "-y", "-loglevel", "error", "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{args.width}x{args.height}", "-framerate", str(args.fps), "-i", "-", "-an", "-c:v", "libx264", "-crf", "22", "-pix_fmt", "yuv420p", str(output)], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    for index in indices:
        sim.data.qpos[:] = states["qpos"][index]
        sim.data.qvel[:] = states["qvel"][index]
        sim.data.ctrl[:] = states["ctrl"][index]
        mujoco.mj_forward(sim.model, sim.data)
        camera = sim.scene_info["camera"]["name"]
        if args.view != "ego":
            camera = mujoco.MjvCamera()
            if args.view == "overview":
                camera.lookat[:] = [.35, 0, .95 + sim.scene_info["root_z"] - .793]
                camera.distance, camera.azimuth, camera.elevation = 2.15, 135, -22
            else:
                plate = sim.data.xpos[sim.plate_id]
                camera.lookat[:] = [plate[0], plate[1], plate[2] + .025]
                camera.distance = .34 if args.view == "topdown" else .37
                camera.azimuth = 90 if args.view == "topdown" else 135
                camera.elevation = -90 if args.view == "topdown" else -35
        renderer.update_scene(sim.data, camera=camera)
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
        image = Image.fromarray(renderer.render())
        draw = ImageDraw.Draw(image)
        draw.rectangle([0, 0, args.width, 27], fill="white")
        draw.text((8, 8), f"SAVED PHYSICS STATE | t={timestamps[index]:.3f}s | {args.view} | seed={report['seed']}", fill="black")
        if writer is None:
            image.save(output)
        else:
            writer.stdin.write(np.asarray(image).tobytes())
    if writer is not None:
        writer.stdin.close()
        error = writer.stderr.read().decode(errors="replace")
        if writer.wait() != 0:
            raise RuntimeError(error)
    renderer.close()
    sim.close()
    print(json.dumps({"output": str(output), "frames": len(indices), "source": "saved physical states; no policy execution or physics integration"}))


if __name__ == "__main__":
    main()
