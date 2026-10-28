#!/usr/bin/env python3
"""Report private ball-label review progress without revealing camera imagery."""
import argparse
import csv
import json
from pathlib import Path


def report(prepared_root: Path) -> dict:
    sessions = []
    for folder in sorted((prepared_root / "sessions").glob("capture_*")):
        index = folder / "index.csv"
        if not index.is_file():
            continue
        queue_path = folder / "review_queue.csv"
        with index.open(encoding="utf-8", newline="") as stream:
            all_frames = list(csv.DictReader(stream))
        if queue_path.is_file():
            with queue_path.open(encoding="utf-8", newline="") as stream:
                queue = list(csv.DictReader(stream))
        else:
            queue = all_frames
        statuses_path = folder / "review_status.json"
        statuses = (json.loads(statuses_path.read_text(encoding="utf-8"))
                    if statuses_path.is_file() else {})
        counts = {"positive": 0, "negative": 0, "unclear": 0, "unreviewed": 0}
        seen = set()
        for frame in queue:
            filename = frame["filename"]
            if filename in seen:
                raise ValueError("Duplicate review queue filename: " + filename)
            seen.add(filename)
            stem = Path(filename).stem
            state = statuses.get(stem, "unreviewed")
            if state not in counts:
                raise ValueError("Unexpected review state: " + str(state))
            counts[state] += 1
            if state in ("positive", "negative"):
                path = folder / "labels" / (stem + ".txt")
                if not path.is_file():
                    raise ValueError("Missing human-reviewed label: " + str(path))
        sessions.append({
            "session_id": folder.name,
            "total_frames": len(all_frames),
            "review_queue_frames": len(queue),
            **counts,
            "reviewed_percent": round(
                100.0 * (counts["positive"] + counts["negative"]) /
                len(queue), 1
            ) if queue else 0.0,
            "suitable_for_split": counts["positive"] > 0 and counts["negative"] > 0,
        })
    return {
        "sessions": sessions,
        "session_count": len(sessions),
        "ready_for_split_count": sum(x["suitable_for_split"] for x in sessions),
        "confirmed_positive_images": sum(x["positive"] for x in sessions),
        "confirmed_no_ball_images": sum(x["negative"] for x in sessions),
        "unreviewed_queue_images": sum(x["unreviewed"] for x in sessions),
        "note": "Progress report only. It does not train or evaluate a detector.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared_root", type=Path)
    parser.add_argument("--report", type=Path, help="Optional metadata-only JSON report")
    args = parser.parse_args()
    summary = report(args.prepared_root)
    result = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(result + "\n", encoding="utf-8")
    print(result)


if __name__ == "__main__":
    main()
