#!/usr/bin/env python3
"""Audit private v0.7.5+ training ZIPs without extracting or uploading images.

The report contains only counts, timestamps, sizes and checksums. All actual
court photographs must remain outside public GitHub repositories.
"""
import argparse
import csv
import hashlib
import io
import json
import statistics
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

MAX_ARCHIVE_BYTES = 350 * 1024 * 1024
MAX_ARCHIVE_ENTRIES = 1500
MAX_FRAME_COUNT = 1200
REQUIRED_FIELDS = {"filename", "timestamp_utc_ms", "image_width", "image_height"}


def inspect_capture(path: Path, verify_jpegs: bool = False) -> dict:
    """Inspect one ZIP. ZIP CRC is always checked; optional JPEG validation uses PIL."""
    if verify_jpegs:
        from PIL import Image

    with zipfile.ZipFile(path, "r") as z:
        entries = z.infolist()
        names = [entry.filename for entry in entries]
        if len(entries) > MAX_ARCHIVE_ENTRIES:
            raise ValueError("Unexpected ZIP entry count")
        if len(set(names)) != len(names):
            raise ValueError("Duplicate ZIP member names")
        if sum(entry.file_size for entry in entries) > MAX_ARCHIVE_BYTES:
            raise ValueError("Archive exceeds private-capture safety bound")
        for name in names:
            p = PurePosixPath(name)
            if name.startswith("/") or ".." in p.parts or chr(92) in name:
                raise ValueError("Unsafe ZIP member path")
        if "frames.csv" not in names:
            raise ValueError("Missing frames.csv")

        raw = z.read("frames.csv")
        reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
        if not REQUIRED_FIELDS.issubset(reader.fieldnames or []):
            raise ValueError("Missing required frame metadata columns")
        rows = list(reader)
        if not 1 <= len(rows) <= MAX_FRAME_COUNT:
            raise ValueError("Invalid frame count")

        filenames = [row["filename"] for row in rows]
        if len(set(filenames)) != len(filenames):
            raise ValueError("Duplicate frame paths in manifest")
        jpeg_entries = {name for name in names if name.startswith("images/")}
        if set(filenames) != jpeg_entries:
            raise ValueError("Frame manifest and ZIP image list disagree")
        if any(not name.endswith(".jpg") or
               not name.startswith("images/frame_") for name in filenames):
            raise ValueError("Unexpected frame filename")

        times = []
        dimensions = set()
        for row in rows:
            ts = int(row["timestamp_utc_ms"])
            w, h = int(row["image_width"]), int(row["image_height"])
            if ts <= 0 or not (1 <= w <= 8192 and 1 <= h <= 8192):
                raise ValueError("Invalid timestamp or image size")
            times.append(ts)
            dimensions.add((w, h))
        intervals = [b - a for a, b in zip(times, times[1:])]
        if any(dt <= 0 for dt in intervals):
            raise ValueError("Non-increasing frame timestamps")

        bad_crc = z.testzip()
        if bad_crc is not None:
            raise ValueError("ZIP CRC mismatch in " + bad_crc)

        # Hash actual uncompressed JPEG bytes, not just the CSV manifest.
        # A re-export with altered frames must not be called a duplicate.
        content_hash = hashlib.sha256()
        for row in rows:
            name = row["filename"]
            name_bytes = name.encode("utf-8")
            content_hash.update(len(name_bytes).to_bytes(4, "big"))
            content_hash.update(name_bytes)
            image_hash = hashlib.sha256()
            with z.open(name) as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    image_hash.update(chunk)
            content_hash.update(image_hash.digest())

        if verify_jpegs:
            for row in rows:
                with z.open(row["filename"]) as stream:
                    with Image.open(stream) as image:
                        dims = image.size
                        image.verify()
                expected = (int(row["image_width"]), int(row["image_height"]))
                if dims != expected:
                    raise ValueError("JPEG dimensions differ from manifest: " + row["filename"])

        return {
            "zip": path.name,
            "zip_bytes": path.stat().st_size,
            "session_id": "capture_" + str(times[0]),
            "manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "image_content_sha256": content_hash.hexdigest(),
            "frames": len(rows),
            "frame_start_utc": datetime.fromtimestamp(times[0] / 1000, timezone.utc).isoformat(),
            "frame_end_utc": datetime.fromtimestamp(times[-1] / 1000, timezone.utc).isoformat(),
            "span_seconds": round((times[-1] - times[0]) / 1000, 1),
            "median_interval_ms": statistics.median(intervals) if intervals else None,
            "long_gaps_over_2000ms": sum(dt > 2000 for dt in intervals),
            "max_gap_ms": max(intervals, default=0),
            "dimensions": sorted([list(pair) for pair in dimensions]),
            "crc_ok": True,
            "jpeg_verified": verify_jpegs,
        }


def audit(paths, verify_jpegs=False):
    results, failures = [], []
    for path in paths:
        try:
            results.append(inspect_capture(path, verify_jpegs))
        except (OSError, ValueError, zipfile.BadZipFile, RuntimeError) as error:
            failures.append({"zip": path.name, "error": str(error)})
    groups = defaultdict(list)
    for result in results:
        groups[result["session_id"]].append(result)
    unique = []
    for session, records in groups.items():
        manifests = {r["manifest_sha256"] for r in records}
        image_contents = {r["image_content_sha256"] for r in records}
        if len(manifests) != 1 or len(image_contents) != 1:
            failures.append({
                "session_id": session,
                "error": "Conflicting manifests or image bytes for same session",
            })
            continue
        canonical = records[0]
        unique.append({
            "session_id": session,
            "frames": canonical["frames"],
            "duplicates": [r["zip"] for r in records[1:]],
            "canonical_zip": canonical["zip"],
        })
    return {
        "input_zip_count": len(paths),
        "readable_zip_count": len(results),
        "unique_session_count": len(unique),
        "unique_frame_count": sum(s["frames"] for s in unique),
        "sessions": unique,
        "zip_details": results,
        "failures": failures,
        "ready_for_annotation": not failures and len(unique) >= 3,
        "model_trained": False,
        "note": "This report does not create ball labels or prove detector accuracy.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zips", type=Path, nargs="+", help="Private original capture ZIP files")
    parser.add_argument("--verify-jpegs", action="store_true",
                        help="Use Pillow to verify every JPEG; ZIP CRC is always checked")
    parser.add_argument("--report", type=Path, help="Write metadata-only JSON")
    args = parser.parse_args()
    result = audit(args.zips, args.verify_jpegs)
    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(output + "\n", encoding="utf-8")
    print(output)
    if result["failures"] or not result["ready_for_annotation"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
