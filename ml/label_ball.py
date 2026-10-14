#!/usr/bin/env python3
"""Minimal local box reviewer for one-ball-per-frame 1on1 captures.

Uses OpenCV only for the local desktop UI. No image or label is uploaded.
No model prediction is ever silently treated as ground truth.
"""
import argparse
import csv
import json
from pathlib import Path

import cv2


def label_session(session: Path, revisit=False):
    rows = list(csv.DictReader((session / "index.csv").open(encoding="utf-8")))
    labels = session / "labels"
    labels.mkdir(exist_ok=True)
    status_path = session / "review_status.json"
    statuses = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
    cv2.namedWindow("Ball labeler", cv2.WINDOW_NORMAL)

    for i, row in enumerate(rows):
        key_name = Path(row["filename"]).stem
        if key_name in statuses and not revisit:
            continue
        image = cv2.imread(str(session / row["filename"]))
        if image is None:
            raise ValueError(f"Cannot open {row['filename']}")
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
            cv2.rectangle(view, (0, 0), (min(view.shape[1] - 1, 950), 35), (0, 0, 0), -1)
            cv2.putText(
                view,
                f"{i + 1}/{len(rows)} | b:box n:no-ball u:unclear s:skip q:quit",
                (6, 24), cv2.FONT_HERSHEY_SIMPLEX, .58, (255, 255, 255), 1
            )
            cv2.imshow("Ball labeler", view)
            key = cv2.waitKey(0) & 0xFF
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
    args = parser.parse_args()
    results = label_session(args.session, args.revisit)
    summary = {kind: list(results.values()).count(kind)
               for kind in ("positive", "negative", "unclear")}
    print("Review:", summary, "- session:", args.session)


if __name__ == "__main__":
    main()
