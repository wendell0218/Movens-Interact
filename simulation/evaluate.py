import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=Path("outputs/randomization/index.json"))
    parser.add_argument("--checkpoint", default="checkpoints/Movens-Interact/adapter.pt")
    parser.add_argument("--base-checkpoint")
    parser.add_argument("--backbone")
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=Path("outputs/robustness"))
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.index.exists() and args.index == Path("outputs/randomization/index.json"):
        subprocess.run([sys.executable, "-m", "simulation.randomize", "--output", str(args.index.parent)], check=True)
    index_path = args.index.resolve()
    index = json.loads(index_path.read_text())
    if not index["complete"]:
        raise ValueError("Variant generation is incomplete")
    cases = [case for case in index["cases"] if args.cases is None or case["case_id"] in args.cases]
    if args.cases and len(cases) != len(set(args.cases)):
        raise ValueError("Requested case does not exist in the index")
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.checkpoint).resolve()
    identity = None
    if not args.dry_run:
        digest = hashlib.sha256()
        with checkpoint_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
                digest.update(chunk)
        identity = {"path": str(checkpoint_path), "sha256": digest.hexdigest(), "bytes": checkpoint_path.stat().st_size, "mtime_ns": checkpoint_path.stat().st_mtime_ns}
    results = []
    for case in cases:
        setup = (index_path.parent / case["scene_json"]).resolve()
        output = args.output / case["case_id"]
        command = [sys.executable, "-m", "simulation.rollout", "--checkpoint", args.checkpoint, "--scene-setup", str(setup), "--seed", str(args.seed), "--seconds", str(args.seconds), "--device", args.device, "--output", str(output)]
        if args.base_checkpoint:
            command += ["--base-checkpoint", args.base_checkpoint]
        if args.backbone:
            command += ["--backbone", args.backbone]
        if args.no_video:
            command += ["--no-video"]
        if args.dry_run:
            print(json.dumps({"case_id": case["case_id"], "command": command}))
            continue
        if checkpoint_path.stat().st_size != identity["bytes"] or checkpoint_path.stat().st_mtime_ns != identity["mtime_ns"]:
            raise ValueError("Checkpoint changed during the controlled evaluation")
        output.mkdir(parents=True, exist_ok=True)
        print(json.dumps({"case_id": case["case_id"], "status": "running"}), flush=True)
        with (output / "run.log").open("w") as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        report_path = output / "rollout_report.json"
        report = json.loads(report_path.read_text()) if process.returncode == 0 and report_path.exists() else None
        valid = report is not None and report["portable_scene_sha256"] == case["scene_sha256"] and report["seed"] == args.seed and report["evaluation_type"] == "live_sim_ego_closed_loop_policy_diagnostic"
        results.append({"case_id": case["case_id"], "group": case["group"], "seed": args.seed, "returncode": process.returncode, "valid_result": valid, "support_verified_success": bool(report["support_verified_success"]) if valid else None, "failure": report["failure"] if valid else "missing_or_invalid_result", "report": str(report_path)})
        groups = {}
        for row in results:
            group = groups.setdefault(row["group"], {"cases": 0, "valid": 0, "successes": 0})
            group["cases"] += 1
            group["valid"] += int(row["valid_result"])
            group["successes"] += int(row["support_verified_success"] is True)
        summary = {"checkpoint": identity, "index": str(index_path), "seed": args.seed, "seconds": args.seconds, "results": results, "groups": groups, "requested_cases": len(cases), "completed_cases": len(results), "missing_cases": [case["case_id"] for case in cases if case["case_id"] not in {row["case_id"] for row in results}], "scope": "One fixed sampling seed per controlled case; not a success-rate estimate over independent trials."}
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        with (args.output / "summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
        print(json.dumps(results[-1]), flush=True)
    if identity and (checkpoint_path.stat().st_size != identity["bytes"] or checkpoint_path.stat().st_mtime_ns != identity["mtime_ns"]):
        raise ValueError("Checkpoint changed during the final evaluation case")


if __name__ == "__main__":
    main()
