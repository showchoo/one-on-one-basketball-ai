#!/usr/bin/env python3
"""Export HUMAN-REVIEWED direct-court sessions to a session-disjoint COCO dataset.

For YOLOX Nano / other COCO loaders. Refuses screen replay and unlabeled images.
Never copies user images into the Git repository; output must be gitignored.
"""
import argparse
import csv
import json
import random
import shutil
from pathlib import Path


def export(source: Path, target: Path, seed=42, min_positives=20):
    sessions = []
    for folder in sorted((source / "sessions").glob("capture_*")):
        metadata_path = folder / "session.json"
        if not metadata_path.exists():
            continue
        info = json.loads(metadata_path.read_text(encoding="utf-8"))
        if info.get("source_mode") != "direct_court":
            continue
        statuses_path = folder / "review_status.json"
        if not statuses_path.exists():
            continue
        statuses = json.loads(statuses_path.read_text(encoding="utf-8"))
        frames = list(csv.DictReader((folder / "index.csv").open(encoding="utf-8")))
        reviewed = []
        for row in frames:
            stem = Path(row["filename"]).stem
            status = statuses.get(stem)
            if status not in ("positive", "negative"):
                continue
            label_file = folder / "labels" / (stem + ".txt")
            if not label_file.exists():
                raise ValueError(f"Missing reviewed label: {label_file}")
            values = []
            for line in label_file.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) != 5 or parts[0] != "0":
                    raise ValueError(f"Invalid class or annotation in {label_file}")
                cx, cy, bw, bh = (float(v) for v in parts[1:])
                if not (0 < bw <= 1 and 0 < bh <= 1 and
                        0 <= cx - bw / 2 < cx + bw / 2 <= 1 and
                        0 <= cy - bh / 2 < cy + bh / 2 <= 1):
                    raise ValueError(f"Invalid YOLO box in {label_file}")
                values.append((cx, cy, bw, bh))
            if status == "positive" and len(values) != 1:
                raise ValueError(f"Expected one ball in {label_file}")
            if status == "negative" and values:
                raise ValueError(f"No-ball label must be empty: {label_file}")
            reviewed.append((row, values))
        if reviewed:
            sessions.append((folder, reviewed))

    if len(sessions) < 3:
        raise ValueError(
            f"Need >=3 distinct, HUMAN-REVIEWED direct-court sessions "
            f"for independent train/val/test; have {len(sessions)}."
        )
    rng = random.Random(seed)
    rng.shuffle(sessions)
    session_groups = {
        "val2017": [sessions[0]],
        "test2017": [sessions[1]],
        "train2017": sessions[2:],
    }
    total_positive = sum(
        len(boxes) for session in sessions for _, boxes in session[1]
    )
    if total_positive < min_positives:
        raise ValueError(
            f"Need at least {min_positives} confirmed positive boxes; have {total_positive}."
        )
    coco_dir = target / "COCO"
    if coco_dir.exists():
        raise FileExistsError(f"Will not overwrite {coco_dir}")
    coco_dir.mkdir(parents=True)
    ann_dir = coco_dir / "annotations"
    ann_dir.mkdir()
    details = {}
    try:
        for split, group in session_groups.items():
            split_dir = coco_dir / split
            split_dir.mkdir()
            images, annotations, a_id, i_id = [], [], 1, 1
            for session, frames in group:
                for row, boxes in frames:
                    src_image = session / row["filename"]
                    if not src_image.is_file():
                        raise FileNotFoundError(src_image)
                    dest_name = session.name + "_" + Path(row["filename"]).name
                    shutil.copy2(src_image, split_dir / dest_name)
                    w, h = int(row["image_width"]), int(row["image_height"])
                    images.append({
                        "id": i_id, "file_name": dest_name,
                        "width": w, "height": h
                    })
                    for cx, cy, bw, bh in boxes:
                        x, y = (cx - bw / 2) * w, (cy - bh / 2) * h
                        pw, ph = bw * w, bh * h
                        annotations.append({
                            "id": a_id, "image_id": i_id, "category_id": 1,
                            "bbox": [x, y, pw, ph], "area": pw * ph,
                            "iscrowd": 0
                        })
                        a_id += 1
                    i_id += 1
            payload = {
                "info": {"description": "1on1 Basketball - verified, direct-court"},
                "licenses": [],
                "images": images,
                "annotations": annotations,
                "categories": [{"id": 1, "name": "basketball",
                                "supercategory": "sports"}]
            }
            (ann_dir / f"instances_{split}.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            details[split] = {
                "sessions": [folder.name for folder, _ in group],
                "images": len(images),
                "positive_boxes": len(annotations)
            }
        (coco_dir / "split_summary.json").write_text(
            json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        shutil.rmtree(coco_dir, ignore_errors=True)
        raise
    return details


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("prepared_root", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-positives", type=int, default=20)
    args = ap.parse_args()
    report = export(args.prepared_root, args.output, args.seed, args.min_positives)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
