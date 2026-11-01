import argparse
import hashlib
import json
import os
import shutil
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("assets/manifest.json"))
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    root = manifest_path.parent
    reused = downloaded = 0
    for entry in manifest["files"]:
        destination = root / entry["path"]
        if destination.is_file() and destination.stat().st_size == entry["bytes"] and hashlib.sha256(destination.read_bytes()).hexdigest() == entry["sha256"]:
            reused += 1
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".partial")
        cached = None
        if args.cache:
            candidates = [args.cache / entry["upstream_path"], args.cache / entry["path"], args.cache / Path(entry["path"]).relative_to("vendor")]
            for candidate in candidates:
                if candidate.is_file() and candidate.stat().st_size == entry["bytes"] and hashlib.sha256(candidate.read_bytes()).hexdigest() == entry["sha256"]:
                    cached = candidate
                    break
        if cached:
            temporary.unlink(missing_ok=True)
            try:
                os.link(cached, temporary)
            except OSError:
                shutil.copyfile(cached, temporary)
            reused += 1
        elif args.offline:
            raise FileNotFoundError(f"Missing verified asset: {entry['path']}. Supply --cache or allow downloads.")
        else:
            request = urllib.request.Request(entry["url"], headers={"User-Agent": "Movens-Interact"})
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
                shutil.copyfileobj(response, handle)
            downloaded += 1
        if temporary.stat().st_size != entry["bytes"] or hashlib.sha256(temporary.read_bytes()).hexdigest() != entry["sha256"]:
            temporary.unlink(missing_ok=True)
            raise ValueError(f"Asset checksum mismatch: {entry['path']}")
        temporary.replace(destination)
    print(json.dumps({"verified": len(manifest["files"]), "reused": reused, "downloaded": downloaded, "bytes": manifest["total_bytes"]}))


if __name__ == "__main__":
    main()
