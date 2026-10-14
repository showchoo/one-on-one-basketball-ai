#!/usr/bin/env python3
"""Safely unpack a v0.7.5+ training ZIP, with explicit source-domain tagging.

No automatic labels are generated; screen-replay frames are kept separate from
real-court frames.  This script never uploads private photos anywhere.
"""
import argparse
import csv
import hashlib
import io
import json
import re
import shutil
import zipfile
from pathlib import Path, PurePosixPath
from PIL import Image, ImageOps

MAX_UNCOMPRESSED = 350 * 1024 * 1024
MAX_FRAMES = 20000


def process(source: Path, destination: Path, mode: str, crop=None):
    if crop and mode != "screen_replay":
        raise ValueError("--crop is limited to screen_replay diagnostics")
    if destination.resolve() == source.resolve():
        raise ValueError("Destination cannot be the input file")
    with zipfile.ZipFile(source) as archive:
        members = archive.infolist()
        if sum(x.file_size for x in members) > MAX_UNCOMPRESSED:
            raise ValueError("ZIP too large for safe local ingestion")
        for info in members:
            name = PurePosixPath(info.filename)
            if info.filename.startswith("/") or ".." in name.parts or "\\" in info.filename:
                raise ValueError(f"Unsafe ZIP member: {info.filename!r}")
        if "frames.csv" not in archive.namelist():
            raise ValueError("No frames.csv: expected v0.7.5+ capture ZIP")
        data = archive.read("frames.csv").decode("utf-8-sig")
        rows = list(csv.DictReader(io.StringIO(data)))
        if not rows or len(rows) > MAX_FRAMES:
            raise ValueError("Empty or excessively large frame manifest")
        required = {"filename", "timestamp_utc_ms", "image_width", "image_height"}
        if not required.issubset(rows[0]):
            raise ValueError("Missing required capture metadata")
        archive_names = set(archive.namelist())
        expected = {r["filename"] for r in rows}
        if len(expected) != len(rows):
            raise ValueError("Duplicate filenames in capture manifest")
        if not expected.issubset(archive_names):
            raise ValueError("Missing JPEGs referenced by frames.csv")
        if any(not x.startswith("images/") or not x.lower().endswith(".jpg") for x in expected):
            raise ValueError("Frame manifest contains unexpected paths")
        session_id = "capture_" + str(int(rows[0]["timestamp_utc_ms"]))
        session_dir = destination / "sessions" / session_id
        if session_dir.exists():
            raise FileExistsError(f"Refusing to overwrite existing session: {session_dir}")
        output_dir = session_dir / "images"
        output_dir.mkdir(parents=True, exist_ok=False)
        output_rows = []
        try:
            for row in rows:
                name = row["filename"]
                raw = archive.read(name)
                if not raw.startswith(b"\xff\xd8"):
                    raise ValueError(f"Not a JPEG: {name}")
                with Image.open(io.BytesIO(raw)) as image:
                    image.verify()
                with Image.open(io.BytesIO(raw)) as image:
                    image = ImageOps.exif_transpose(image).convert("RGB")
                    if (image.width, image.height) != (
                        int(row["image_width"]), int(row["image_height"])
                    ):
                        raise ValueError(f"Dimensions disagree with manifest: {name}")
                    if crop:
                        x0, y0, x1, y1 = crop
                        if not (0 <= x0 < x1 <= image.width and
                                0 <= y0 < y1 <= image.height):
                            raise ValueError(f"Crop outside image: {name}")
                        image = image.crop(crop)
                    out_name = Path(name).name
                    output = output_dir / out_name
                    if crop:
                        image.save(output, "JPEG", quality=90)
                    else:
                        output.write_bytes(raw)
                output_rows.append({
                    "filename": f"images/{out_name}",
                    "timestamp_utc_ms": row["timestamp_utc_ms"],
                    "image_width": image.width,
                    "image_height": image.height,
                    "source_mode": mode,
                    "original_sha256": hashlib.sha256(raw).hexdigest(),
                    "label_status": "unreviewed"
                })
            with (session_dir / "index.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
                writer.writeheader()
                writer.writerows(output_rows)
            summary = {
                "session_id": session_id,
                "source_mode": mode,
                "frames": len(output_rows),
                "crop_xyxy": list(crop) if crop else None,
                "labeled_positive": 0,
                "labeled_negative": 0,
                "trainable": False,
                "warning": (
                    "Monitor/video-replay images MUST NOT be mixed into direct-court "
                    "performance validation."
                    if mode == "screen_replay" else
                    "No labels available: annotate ball boxes and no-ball frames first."
                )
            }
            (session_dir / "session.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            return session_dir, summary
        except Exception:
            shutil.rmtree(session_dir, ignore_errors=True)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip_file", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-mode", required=True,
                        choices=["direct_court", "screen_replay"])
    parser.add_argument("--crop", help="x0,y0,x1,y1 for screen replay ONLY")
    args = parser.parse_args()
    crop = tuple(int(x.strip()) for x in args.crop.split(",")) if args.crop else None
    if crop is not None and len(crop) != 4:
        parser.error("--crop requires four comma-separated pixel coordinates")
    folder, summary = process(args.zip_file, args.output, args.source_mode, crop)
    print(json.dumps({"folder": str(folder), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
