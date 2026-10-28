#!/usr/bin/env python3
"""Build a small, representative HUMAN-review queue from private court frames.

Uses 3 kinds of time-distributed frames: high-motion, low-motion and
middle-motion. It makes no claim of finding the ball and creates NO labels.
All photos remain on the operator's private local machine.
"""
import argparse
import csv
import json
import math
from pathlib import Path, PurePosixPath

from PIL import Image, ImageChops, ImageStat


def _safe_frame(session: Path, name: str) -> Path:
    parts = PurePosixPath(name)
    if (parts.is_absolute() or len(parts.parts) != 2 or
            parts.parts[0] != "images" or
            not parts.parts[1].lower().endswith(".jpg")):
        raise ValueError("Invalid frame path: " + name)
    path = session / "images" / parts.parts[1]
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _small_gray(path: Path):
    with Image.open(path) as image:
        image.load()
        return image.convert("L").resize((64, 36), Image.Resampling.BILINEAR)


def build_queue(session: Path, per_session: int = 120) -> dict:
    """Create review_queue.csv: one candidate per consecutive temporal bucket."""
    if per_session < 3:
        raise ValueError("At least 3 review samples are needed")
    index_path = session / "index.csv"
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    with index_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Empty session manifest")

    # Scores measure global image change, NOT ball probability.
    scores = []
    previous = None
    seen = set()
    for row in rows:
        name = row["filename"]
        if name in seen:
            raise ValueError("Repeated frame filename in index")
        seen.add(name)
        frame = _small_gray(_safe_frame(session, name))
        if previous is None:
            score = 0.0
        else:
            difference = ImageChops.difference(frame, previous)
            score = float(ImageStat.Stat(difference).mean[0])
        if not math.isfinite(score):
            raise ValueError("Non-finite motion score")
        scores.append(score)
        previous = frame

    count = min(per_session, len(rows))
    selected = []
    policies = ("high-motion", "low-motion", "middle-motion")
    for bucket in range(count):
        start = bucket * len(rows) // count
        stop = (bucket + 1) * len(rows) // count
        options = list(range(start, stop))
        by_activity = sorted(options, key=lambda idx: (scores[idx], idx))
        policy = policies[bucket % 3]
        if policy == "high-motion":
            chosen = by_activity[-1]
        elif policy == "low-motion":
            chosen = by_activity[0]
        else:
            chosen = by_activity[len(by_activity) // 2]
        selected.append({
            "filename": rows[chosen]["filename"],
            "timestamp_utc_ms": rows[chosen]["timestamp_utc_ms"],
            "motion_score": f"{scores[chosen]:.3f}",
            "selection_policy": policy,
        })

    output = session / "review_queue.csv"
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    return {
        "session": session.name,
        "source_frames": len(rows),
        "selected": len(selected),
        "output": str(output),
        "labels_created": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path,
                        help="An ingested capture session or root with sessions/ directory")
    parser.add_argument("--per-session", type=int, default=120,
                        help="Number of images to review from each session")
    args = parser.parse_args()

    if (args.path / "index.csv").is_file():
        sessions = [args.path]
    else:
        sessions = sorted((args.path / "sessions").glob("capture_*"))
    if not sessions:
        parser.error("No ingested sessions with index.csv found")

    summaries = [build_queue(session, args.per_session) for session in sessions]
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
