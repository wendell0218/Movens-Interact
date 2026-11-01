import argparse
import hashlib
import json
from pathlib import Path
import ssl
import urllib.request


parser = argparse.ArgumentParser(description="Download the public Body base and Qwen processor/configuration files")
parser.add_argument("--models", nargs="+", choices=["base", "backbone"],
    default=["base", "backbone"])
parser.add_argument("--checkpoint-root", default="checkpoints")
parser.add_argument("--skip-verify", action="store_true",
    help="Skip SHA256 for existing files of the expected size; new downloads are always verified")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "configs/models.json").read_text())
checkpoint_root = (root / args.checkpoint_root).resolve()
context = ssl.create_default_context()
system_ca = Path("/etc/ssl/certs/ca-certificates.crt")
if system_ca.is_file():
    context.load_verify_locations(system_ca)


def fetch(url, target, size, expected_sha256):
    if target.is_file() and target.stat().st_size == size:
        if args.skip_verify:
            print(f"Using existing {target} (size checked)", flush=True)
            return
        checksum = hashlib.sha256()
        with target.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                checksum.update(chunk)
        if checksum.hexdigest() == expected_sha256:
            print(f"Verified existing {target}", flush=True)
            return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    print(f"Downloading {target} ({size / 1024 ** 3:.3f} GiB)", flush=True)
    request = urllib.request.Request(url, headers={"User-Agent": "Movens-Interact/0.1.0"})
    checksum = hashlib.sha256()
    count, displayed_gib = 0, 0
    with urllib.request.urlopen(request, timeout=60, context=context) as response, partial.open("wb") as output:
        while True:
            chunk = response.read(8 * 1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
            checksum.update(chunk)
            count += len(chunk)
            if count // 1024 ** 3 > displayed_gib:
                displayed_gib = count // 1024 ** 3
                print(f"  {displayed_gib} GiB / {size / 1024 ** 3:.2f} GiB", flush=True)
    if count != size or checksum.hexdigest() != expected_sha256:
        raise ValueError(f"Downloaded file failed size/SHA256 verification: {partial}")
    partial.replace(target)
    print(f"Verified {target}", flush=True)


for name in args.models:
    entry = manifest[name]
    relative = Path(entry["target"]).relative_to("checkpoints")
    destination = checkpoint_root / relative
    if name == "backbone":
        for file in entry["files"]:
            url = f"https://huggingface.co/{entry['repo_id']}/resolve/{entry['revision']}/{file['filename']}"
            fetch(url, destination / file["filename"], file["bytes"], file["sha256"])
    else:
        url = f"https://huggingface.co/{entry['repo_id']}/resolve/{entry['revision']}/{entry['filename']}"
        fetch(url, destination, entry["bytes"], entry["sha256"])
