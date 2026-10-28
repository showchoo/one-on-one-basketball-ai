#!/usr/bin/env python3
"""Minimal local box reviewer for one-ball-per-frame 1on1 captures.

Uses OpenCV only for the local desktop UI. No image or label is uploaded.
No model prediction is ever silently treated as ground truth.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def verified_suggestion(entry, photo: Path):
    """Validate a model hint; never mark it reviewed until a human presses a."""
    if entry is None:
        return None
    box = entry.get("box")
    score = entry.get("confidence")
    digest = entry.get("image_sha256")
    if (not isinstance(box, list) or len(box) != 4 or
            not all(isinstance(v, (float, int)) and math.isfinite(v) for v in box) or
            not isinstance(score, (float, int)) or not math.isfinite(score) or
            not 0 <= score <= 1):
        raise ValueError("Invalid unverified model suggestion")
    cx, cy, w, h = box
    if not (0 < w <= 1 and 0 < h <= 1 and
            0 <= cx - w / 2 < cx + w / 2 <= 1 and
            0 <= cy - h / 2 < cy + h / 2 <= 1):
        raise ValueError("Invalid suggested ball rectangle")
    with photo.open("rb") as source:
        actual = hashlib.file_digest(source, "sha256").hexdigest()
    if actual != digest:
        raise ValueError("Model suggestion belongs to a different image: " + str(photo))
    return entry


def label_session(session: Path, revisit=False, queue=None, proposals=None):
    import cv2
    with (session / "index.csv").open(encoding="utf-8", newline="") as stream:
        original_rows = list(csv.DictReader(stream))
    if queue is None:
        rows = original_rows
    else:
        with queue.open(encoding="utf-8", newline="") as stream:
            requested = list(csv.DictReader(stream))
        indexed = {row["filename"]: row for row in original_rows}
        if len(indexed) != len(original_rows):
            raise ValueError("Duplicate paths in index.csv")
        filenames = [row["filename"] for row in requested]
        if not filenames or len(set(filenames)) != len(filenames):
            raise ValueError("Queue must include unique, nonempty frame names")
        if any(name not in indexed for name in filenames):
            raise ValueError("Queue contains frames not present in session index")
        rows = [indexed[name] for name in filenames]
    labels = session / "labels"
    labels.mkdir(exist_ok=True)
    status_path = session / "review_status.json"
    statuses = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
    proposal_entries = {}
    if proposals is not None:
        data = json.loads(proposals.read_text(encoding="utf-8"))
        if data.get("schema_version") != 1 or not isinstance(data.get("proposals"), dict):
            raise ValueError("Unsupported model proposal file")
        proposal_entries = data["proposals"]
    cv2.namedWindow("Ball labeler", cv2.WINDOW_NORMAL)

    for i, row in enumerate(rows):
        key_name = Path(row["filename"]).stem
        if key_name in statuses and not revisit:
            continue
        image = cv2.imread(str(session / row["filename"]))
        if image is None:
            raise ValueError(f"Cannot open {row['filename']}")
        proposed = verified_suggestion(
            proposal_entries.get(key_name), session / row["filename"]
        )
        height, width = image.shape[:2]
        scale = min(1.0, 1050.0 / max(width, height))
        shown = cv2.resize(image, (round(width * scale), round(height * scale)))
        label_file = labels / (key_name + ".txt")

        while True:
            view = shown.copy()
            if label_file.exists():
                for line in label_file.read_text(encoding="utf-8").splitlines():
                    parts = line.split()
                    if len(parts) != 5 or parts[0] != "0":
                        continue
                    cx, cy, bw, bh = map(float, parts[1:])
                    x0 = round((cx - bw / 2) * shown.shape[1])
                    y0 = round((cy - bh / 2) * shown.shape[0])
                    x1 = round((cx + bw / 2) * shown.shape[1])
                    y1 = round((cy + bh / 2) * shown.shape[0])
                    cv2.rectangle(view, (x0, y0), (x1, y1), (0, 255, 0), 2)
            if proposed is not None:
                cx, cy, bw, bh = proposed["box"]
                x0 = round((cx - bw / 2) * shown.shape[1])
                y0 = round((cy - bh / 2) * shown.shape[0])
                x1 = round((cx + bw / 2) * shown.shape[1])
                y1 = round((cy + bh / 2) * shown.shape[0])
                cv2.rectangle(view, (x0, y0), (x1, y1), (0, 215, 255), 2)
                cv2.putText(view, "AI HINT unverified %.2f" % proposed["confidence"],
                            (max(4, x0), max(50, y0 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 215, 255), 2)
            cv2.rectangle(view, (0, 0), (min(view.shape[1] - 1, 1050), 35), (0, 0, 0), -1)
            cv2.putText(
                view,
                f"{i + 1}/{len(rows)} | a:confirm AI  b:box  n:no-ball  u:unclear  s:skip  q:quit",
                (6, 24), cv2.FONT_HERSHEY_SIMPLEX, .58, (255, 255, 255), 1
            )
            cv2.imshow("Ball labeler", view)
            key = cv2.waitKey(0) & 0xFF
            if key == ord("a") and proposed is not None:
                cx, cy, bw, bh = proposed["box"]
                label_file.write_text(
                    f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\\n".replace("\\\\n", "\\n"),
                    encoding="utf-8"
                )
                statuses[key_name] = "positive"
                break
            if key == ord("b"):
                x, y, w, h = cv2.selectROI(
                    "Ball labeler", shown, showCrosshair=True, fromCenter=False
                )
                if w < 2 or h < 2:
                    continue
                cx = (x + w / 2) / shown.shape[1]
                cy = (y + h / 2) / shown.shape[0]
                bw = w / shown.shape[1]
                bh = h / shown.shape[0]
                # Never accept an invalid box.
                if not all(0 <= x <= 1 for x in (cx, cy, bw, bh)):
                    continue
                label_file.write_text(
                    f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n",
                    encoding="utf-8"
                )
                statuses[key_name] = "positive"
                break
            if key == ord("n"):
                label_file.write_text("", encoding="utf-8")
                statuses[key_name] = "negative"
                break
            if key == ord("u"):
                label_file.unlink(missing_ok=True)
                statuses[key_name] = "unclear"
                break
            if key == ord("s"):
                break
            if key == ord("q") or key == 27:
                cv2.destroyAllWindows()
                return statuses
        status_path.write_text(
            json.dumps(statuses, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    cv2.destroyAllWindows()
    return statuses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path,
                        help="output/sessions/capture_<timestamp>")
    parser.add_argument("--revisit", action="store_true",
                        help="also review already-labeled frames")
    parser.add_argument("--queue", type=Path,
                        help="CSV queue from select_review_frames.py (otherwise all frames)")
    parser.add_argument("--proposals", type=Path,
                        help="Optional unverified AI hints from propose_ball_boxes.py")
    args = parser.parse_args()
    results = label_session(args.session, args.revisit, args.queue, args.proposals)
    summary = {kind: list(results.values()).count(kind)
               for kind in ("positive", "negative", "unclear")}
    print("Review:", summary, "- session:", args.session)


if __name__ == "__main__":
    main()
