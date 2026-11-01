import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from movens_interact.backbone import MovensZeroModel


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Cache frozen Qwen features of current simulator ego observations")
    parser.add_argument("--manifest", default="data/apple_teacher/index.json")
    parser.add_argument("--output", default="data/apple_context")
    parser.add_argument("--base-checkpoint", default="checkpoints/Movens-Zero-XL/model.safetensors")
    parser.add_argument("--base-sha256", default="d1e6a44092938ce641d3bec43138df03e2cdb2bb0ebf1fb64840f1635de1dd3a")
    parser.add_argument("--qwen-config", default="checkpoints/Qwen2.5-Omni-3B")
    parser.add_argument("--body-config", default="configs/body.json")
    parser.add_argument("--prompt", default="Move the apple to the plate.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("Batch size must be positive")
    index_path = Path(args.manifest)
    index = json.loads(index_path.read_text())
    if not index["collection_complete"]:
        raise ValueError("Finish the paired teacher collection before freezing its identity")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_hash = sha256_file(args.base_checkpoint)
    if checkpoint_hash != args.base_sha256:
        raise ValueError("Base checkpoint checksum differs from the selected release")
    config = json.loads(Path(args.body_config).read_text())
    processor_names = ["config.json", "preprocessor_config.json", "tokenizer_config.json", "tokenizer.json", "special_tokens_map.json", "added_tokens.json", "chat_template.json", "vocab.json", "vocab.txt", "merges.txt"]
    qwen_files = sorted(path for path in Path(args.qwen_config).iterdir() if path.is_file() and path.name in processor_names)
    if not any(path.name == "config.json" for path in qwen_files):
        raise FileNotFoundError("Download the local Qwen configuration and processor assets first")
    qwen_hash = hashlib.sha256(json.dumps({path.name: sha256_file(path) for path in qwen_files}, sort_keys=True).encode()).hexdigest()
    identity = {"checkpoint_sha256": checkpoint_hash, "body_config_sha256": sha256_file(args.body_config),
        "qwen_configuration_sha256": qwen_hash, "prompt": args.prompt,
        "teacher_index_sha256": sha256_file(index_path), "scene_sha256": index["scene_sha256"]}
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if any(previous[key] != value for key, value in identity.items()):
            raise ValueError("Existing Qwen cache belongs to a different dataset/model/configuration/prompt")
        if previous["complete"]:
            cached_episodes = {row["episode_index"]: row for row in previous["episodes"]}
            for row in index["episodes"]:
                if not row["teacher_success"] or not row["success"]:
                    continue
                cached = cached_episodes[row["episode_index"]]
                if sha256_file(index_path.parent / row["rgb"]) != cached["source_sha256"] or sha256_file(output / cached["path"]) != cached["sha256"]:
                    raise ValueError("Previously completed Qwen cache or source RGB changed")
            print(json.dumps({"complete": True, "reused_unchanged_manifest": True, "episodes": len(cached_episodes)}), flush=True)
            return
    report = dict(identity, checkpoint=str(Path(args.base_checkpoint)), dataset_index=str(index_path),
        dataset_index_sha256=identity["teacher_index_sha256"], context_domain="paired_sim_teacher_current_ego",
        data_revision=index["data_revision"], complete=False,
        causal_rule="Encode only the RGB at its recorded observation time, duplicate that frame for temporal patching, and select cache timestamp <= policy observation time.", episodes=[])
    torch.set_num_threads(8)
    model = None
    started = time.perf_counter()
    for row in index["episodes"]:
        if not row["teacher_success"] or not row["success"]:
            continue
        if not row["rgb_ready"]:
            raise ValueError("Successful teacher is missing its ego observations")
        episode = row["episode_index"]
        source = index_path.parent / row["rgb"]
        checksum = sha256_file(source)
        if row.get("rgb_sha256", checksum) != checksum:
            raise ValueError("Teacher RGB changed after collection")
        destination = output / f"episode_{episode:06d}.npz"
        with np.load(source, allow_pickle=False) as archive:
            frames, timestamps, frame_indices = archive["rgb"], archive["timestamps"].astype(np.float64), archive["frame_indices"].astype(np.int64)
        if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3 or len(frames) != len(timestamps) or len(timestamps) != len(frame_indices) or not np.all(np.diff(timestamps) > 0):
            raise ValueError("Invalid current-ego observation arrays")
        reused = destination.exists()
        if reused:
            with np.load(destination, allow_pickle=False) as cached:
                if str(cached["source_sha256"]) != checksum or not np.array_equal(cached["timestamps"], timestamps) or not np.array_equal(cached["frame_indices"], frame_indices):
                    raise ValueError("Existing context cache differs from the teacher RGB")
                shape = list(cached["context"].shape)
        else:
            if model is None:
                if not args.device.startswith("cuda") or not torch.cuda.is_available():
                    raise RuntimeError("Qwen caching requires CUDA")
                torch.cuda.set_per_process_memory_fraction(.30, torch.device(args.device))
                model = MovensZeroModel.from_pretrained(args.base_checkpoint, args.qwen_config, config, device=args.device, torch_dtype=torch.bfloat16)
                model.requires_grad_(False).eval()
            chunks = []
            with torch.inference_mode():
                for start in range(0, len(frames), args.batch_size):
                    videos = [np.stack([frame, frame]) for frame in frames[start:start + args.batch_size]]
                    inputs = model._prepare_inputs([args.prompt] * len(videos), "video", None, videos, video_sample_fps=30.0, video_size_pixels=50176)
                    with model._autocast():
                        context = model._extract_context(inputs)
                    chunks.append(context[:, 0].to(torch.float16).cpu().numpy())
            context = np.concatenate(chunks)
            if context.shape[0] != len(frames) or context.shape[-1] != 2048 or not np.isfinite(context).all():
                raise ValueError("Invalid Qwen context")
            shape = list(context.shape)
            temporary = destination.with_suffix(".partial.npz")
            np.savez_compressed(temporary, context=context, timestamps=timestamps, frame_indices=frame_indices, source_sha256=np.asarray(checksum))
            temporary.replace(destination)
        report["episodes"].append({"episode_index": episode, "path": destination.name, "frames": len(frames),
            "context_shape": shape, "source_sha256": checksum, "sha256": sha256_file(destination), "reused": reused})
        manifest_path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"episode_index": episode, "context_shape": shape, "reused": reused}), flush=True)
    report.update(complete=True, elapsed_seconds=time.perf_counter() - started,
        gpu_peak_gib=torch.cuda.max_memory_allocated() / 1024 ** 3 if model is not None else 0.)
    manifest_path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
