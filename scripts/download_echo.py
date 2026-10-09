"""Download an audited ModelScope snapshot and verify publisher SHA256 values."""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / "configs/model_snapshots.json"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=["2B", "9B"], default="2B")
    parser.add_argument("--root", required=True, type=Path, help="Model storage directory")
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    snapshot = json.loads(args.manifest.read_text(encoding="utf-8"))["snapshots"][args.size]
    destination = args.root.resolve() / snapshot["model_id"].split("/")[-1]
    args.report_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.report_dir / f"echo_{args.size.lower()}_download.json"
    report = {
        "stage": "artifact_download",
        "model_inference_tested": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": snapshot["source"],
        "model_id": snapshot["model_id"],
        "revision": snapshot["revision"],
        "destination": str(destination),
        "status": "FAIL",
        "files": [],
    }
    try:
        from modelscope import snapshot_download

        print(f'Downloading {snapshot["model_id"]} at {snapshot["revision"]}', flush=True)
        downloaded = snapshot_download(
            model_id=snapshot["model_id"],
            revision=snapshot["revision"],
            local_dir=str(destination),
        )
        if Path(downloaded).resolve() != destination:
            raise RuntimeError(f"Unexpected download directory: {downloaded}")
        for expected in snapshot["files"]:
            relative = Path(expected["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Invalid manifest path: {relative}")
            path = destination / relative
            actual_bytes = path.stat().st_size
            print(f'Verifying {expected["path"]} ({actual_bytes:,} bytes)', flush=True)
            actual_sha256 = sha256_file(path)
            matches = actual_bytes == expected["bytes"] and actual_sha256 == expected["sha256"]
            report["files"].append({
                "path": expected["path"],
                "bytes": actual_bytes,
                "sha256": actual_sha256,
                "matches_publisher": matches,
            })
            if not matches:
                raise RuntimeError(f'Publisher checksum mismatch: {expected["path"]}')
        report["status"] = "PASS"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f'{report["status"]}: {report_path.resolve()}', flush=True)
    if report["status"] != "PASS":
        print(report["error"], file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
