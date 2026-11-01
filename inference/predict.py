import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from movens_interact.model import MovensInteractModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/Movens-Interact/adapter.pt")
    parser.add_argument("--base-checkpoint")
    parser.add_argument("--backbone")
    parser.add_argument("--image", type=Path, default=Path("examples/ego.png"))
    parser.add_argument("--observation", type=Path, default=Path("examples/observation.npz"))
    parser.add_argument("--previous-plan", type=Path)
    parser.add_argument("--frame", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=Path("outputs/prediction"))
    args = parser.parse_args()
    observation = np.load(args.observation, allow_pickle=False)
    frame = int(observation["frame"]) if args.frame is None and "frame" in observation else (0 if args.frame is None else args.frame)
    previous = dict(np.load(args.previous_plan, allow_pickle=False)) if args.previous_plan else None
    model = MovensInteractModel.from_pretrained(args.checkpoint, base_checkpoint=args.base_checkpoint, backbone=args.backbone, device=args.device)
    plan = model.predict(np.asarray(Image.open(args.image).convert("RGB")), observation["body_abs"], observation["hand_state"], frame, previous_plan=previous, seed=args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "plan.npz", **{key: np.asarray(value) for key, value in plan.items() if value is not None})
    summary = {"body_absolute_shape": list(plan["body_absolute"].shape), "hand_absolute_shape": list(plan["hand_absolute"].shape), "frame": frame, "seed": args.seed, "plan_id": int(plan["plan_id"]), "first_physical_frame": int(plan["first_physical_frame"]), "horizon": 30, "execute": 20, "retained_prefix": 10, "checkpoint": str(args.checkpoint)}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
