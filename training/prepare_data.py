import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from movens_interact.kinematics import ISAAC_JOINT_NAMES


DATASET = "nvidia/GR00T-N1.7-AppleToPlate"
REVISION = "d89c126a713c6632432a607c12661546ff4d6ea9"
FILES = {
    "meta/info.json": "0e3dd494bace81fcb172f77b5cfe089ed7f7d6babb4b008cb9ada425e9efaf36",
    "meta/modality.json": "84e5eb4f708a214923b387e2258336e137b6669bdeab1fdf01142aa967f0d62a",
    "meta/tasks.jsonl": "8b5e54d2509a433222a8a478079d6edba73d366072da32602d5d7fd496663335",
    "data/chunk-000/episode_000000.parquet": "6c47e7176707c3e92754fbed51ed0fc72ead9d3314b68cf9ede235c199837700",
}
VIDEO = "videos/chunk-000/observation.images.ego_view/episode_000000.mp4"
VIDEO_SHA256 = "7b5e7359b0b8c4a045203790aa120e9049a854c7df1673d3b903e9b6538de29e"


def main():
    parser = argparse.ArgumentParser(description="Prepare the pinned real-robot Apple episode as a fixed-scene reference")
    parser.add_argument("--raw-dir", default="data/apple_raw")
    parser.add_argument("--output", default="data/apple_prepared")
    parser.add_argument("--endpoint", default="https://huggingface.co")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--download-video", action="store_true")
    args = parser.parse_args()
    raw = Path(args.raw_dir) / REVISION
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    expected = dict(FILES)
    if args.download_video:
        expected[VIDEO] = VIDEO_SHA256
    receipts = []
    for filename, checksum in expected.items():
        destination = raw / filename
        url = f"{args.endpoint.rstrip('/')}/datasets/{DATASET}/resolve/{REVISION}/{filename}"
        reused = destination.exists()
        final_url = None
        if not reused:
            if args.offline:
                raise FileNotFoundError(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".partial")
            request = urllib.request.Request(url, headers={"User-Agent": "Movens-Interact"})
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
                final_url = response.geturl()
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    handle.write(chunk)
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != checksum:
                raise ValueError(f"Pinned source checksum mismatch: {filename}")
            temporary.replace(destination)
        actual = hashlib.sha256(destination.read_bytes()).hexdigest()
        if actual != checksum:
            raise ValueError(f"Existing raw file differs from the pinned source: {destination}")
        receipts.append({"path": filename, "sha256": actual, "bytes": destination.stat().st_size,
            "source_url": url, "final_url": final_url, "reused": reused})
    (raw / "download_manifest.json").write_text(json.dumps({"dataset": DATASET, "revision": REVISION,
        "selected_episodes": [0], "complete_lerobot_dataset": False, "files": receipts}, indent=2) + "\n")
    source = raw / "data/chunk-000/episode_000000.parquet"
    table = pq.read_table(source, columns=["observation.state", "action", "timestamp", "frame_index"])
    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)
    action = np.asarray(table["action"].to_pylist(), dtype=np.float64)
    source_times = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64)
    source_times -= source_times[0]
    if state.shape != action.shape or state.shape[1] != 43 or not np.isfinite(state).all() or not np.isfinite(action).all() or not np.all(np.diff(source_times) > 0):
        raise ValueError("Invalid raw Apple state/action/time interface")
    source_body_names = []
    for side in ["left", "right"]:
        source_body_names += [f"{side}_{joint}_joint" for joint in ["hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"]]
    source_body_names += [f"waist_{axis}_joint" for axis in ["yaw", "roll", "pitch"]]
    for side in ["left", "right"]:
        source_body_names += [f"{side}_{joint}_joint" for joint in ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"]]
    body_names = list(ISAAC_JOINT_NAMES)
    permutation = [source_body_names.index(name) for name in body_names]
    hand_names = [f"{side}_hand_{joint}_joint" for side in ["left", "right"] for joint in ["index_0", "index_1", "middle_0", "middle_1", "thumb_0", "thumb_1", "thumb_2"]]
    hand_permutation = [5, 6, 3, 4, 0, 1, 2, 10, 11, 12, 13, 7, 8, 9]
    canonical_state = np.c_[state[:, permutation], state[:, 29:43][:, hand_permutation]]
    reference_raw = np.c_[state[:, permutation], action[:, 29:43]]
    timestamps = np.arange(int(np.ceil(source_times[-1] * 50)) + 1, dtype=np.float64) / 50
    references = np.stack([np.interp(timestamps, source_times, reference_raw[:, j]) for j in range(43)], axis=1)
    prepared = output / "episode_000000.npz"
    np.savez_compressed(prepared, timestamps=timestamps, references43=references,
        initial_qpos43=canonical_state[0].astype(np.float32), source_timestamps=source_times,
        source_frame_indices=np.asarray(table["frame_index"].to_pylist(), dtype=np.int64),
        source_state43=canonical_state.astype(np.float32), source_action43=action.astype(np.float32),
        body_joint_names=np.asarray(body_names), hand_joint_names=np.asarray(hand_names))
    manifest = {"format": "movens_interact_prepared_reference_v1", "dataset": DATASET, "revision": REVISION,
        "source_episode_index": 0, "source_parquet": str(source), "source_sha256": FILES["data/chunk-000/episode_000000.parquet"],
        "prepared": prepared.name, "prepared_sha256": hashlib.sha256(prepared.read_bytes()).hexdigest(),
        "fps": 50, "frames": len(timestamps), "body_joint_names": body_names, "hand_joint_names": hand_names,
        "body_reference_source": "original measured observation.state body29, not original expert body commands",
        "hand_reference_source": "original desired hand action14, interpreted in canonical export order",
        "resampling": "linear interpolation of measured body joints and desired hand commands; hold the final sample through the final 50Hz tick",
        "real_root_available": False, "root_estimated": False,
        "scope": "one public real demonstration adapted as a reference for the supplied fixed-base scene; simulator collection must validate successful execution",
        "mapping": {"source_body_names": source_body_names, "body_to_isaac": permutation,
            "hand_state_to_canonical": hand_permutation, "hand_action_to_canonical": list(range(14)),
            "evidence_url": "https://huggingface.co/nvidia/GR00T-N1.7-ApplePnP-V1/blob/1c956f1f622cc6496f120efff43012dddeb2ff5e/exported_leapp.yaml",
            "certainty": "cross-source export mapping; left-hand temporal alignment supports it, within-finger equal command pairs and zero right-hand targets prevent complete dynamic identification"}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"prepared": str(prepared), "frames": len(timestamps), "real_root_available": False}), flush=True)


if __name__ == "__main__":
    main()
