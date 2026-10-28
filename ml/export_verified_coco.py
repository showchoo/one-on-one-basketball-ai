#!/usr/bin/env python3
"""Export human-reviewed basketball boxes without cross-session data leakage.

Fails closed if a split has no verified ball or no-ball images, if labels are
malformed, or if identical actual JPEG bytes occur in different sessions.
Never uploads any private training photo to GitHub.
"""
import argparse
import csv
import hashlib
import json
import math
import random
import shutil
from pathlib import Path


def load_session(folder: Path):
    info = json.loads((folder / "session.json").read_text(encoding="utf-8"))
    if info.get("source_mode") != "direct_court":
        return None
    statuses_file = folder / "review_status.json"
    if not statuses_file.is_file():
        return None
    statuses = json.loads(statuses_file.read_text(encoding="utf-8"))
    with (folder / "index.csv").open(encoding="utf-8", newline="") as stream:
        source = list(csv.DictReader(stream))
    if not source:
        raise ValueError(f"Empty frame index in {folder}")

    reviewed = []
    for row in source:
        relative = row["filename"]
        # Only process photos produced by the private ingest step.
        if (Path(relative).is_absolute() or Path(relative).parent != Path("images")
                or Path(relative).suffix.lower() != ".jpg"):
            raise ValueError(f"Unexpected frame path in {folder}: {relative}")
        stem = Path(relative).stem
        status = statuses.get(stem)
        if status not in ("positive", "negative"):
            continue
        label_file = folder / "labels" / (stem + ".txt")
        if not label_file.is_file():
            raise ValueError(f"Missing reviewed label: {label_file}")

        boxes = []
        for line in label_file.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5 or parts[0] != "0":
                raise ValueError(f"Invalid annotation class in {label_file}")
            try:
                cx, cy, bw, bh = [float(value) for value in parts[1:]]
            except ValueError as error:
                raise ValueError(f"Non-numeric annotation in {label_file}") from error
            if (not all(math.isfinite(x) for x in (cx, cy, bw, bh))
                    or not (0 < bw <= 1 and 0 < bh <= 1
                            and 0 <= cx - bw / 2 < cx + bw / 2 <= 1
                            and 0 <= cy - bh / 2 < cy + bh / 2 <= 1)):
                raise ValueError(f"Invalid YOLO box in {label_file}")
            boxes.append((cx, cy, bw, bh))

        if status == "positive" and len(boxes) != 1:
            raise ValueError(f"Expected one human-confirmed ball in {label_file}")
        if status == "negative" and boxes:
            raise ValueError(f"Verified no-ball label must be empty: {label_file}")

        # The checksum was measured during ingest; cross-check it against the
        # current actual image bytes so post-ingest changes cannot go unnoticed.
        source_image = folder / relative
        if not source_image.is_file():
            raise FileNotFoundError(source_image)
        digest = hashlib.file_digest(source_image.open("rb"), "sha256").hexdigest()
        expected = row.get("original_sha256", "")
        if expected and digest != expected:
            raise ValueError(f"Image changed since ingest: {source_image}")
        width, height = int(row["image_width"]), int(row["image_height"])
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image dimensions: {source_image}")
        reviewed.append({"row": row, "boxes": boxes, "hash": digest})

    if not reviewed:
        return None
    positive = sum(len(x["boxes"]) for x in reviewed)
    return {
        "folder": folder,
        "reviewed": reviewed,
        "positives": positive,
        "negatives": len(reviewed) - positive,
    }


def export(source: Path, target: Path, seed=42, min_positives=20,
           min_positives_per_split=1, min_negatives_per_split=1):
    sessions = []
    for folder in sorted((source / "sessions").glob("capture_*")):
        if not (folder / "session.json").is_file():
            continue
        session = load_session(folder)
        if session is not None:
            sessions.append(session)

    if len(sessions) < 3:
        raise ValueError(
            f"Need >=3 distinct, human-reviewed direct-court sessions; "
            f"have {len(sessions)}."
        )
    total_positive = sum(s["positives"] for s in sessions)
    if total_positive < min_positives:
        raise ValueError(
            f"Need at least {min_positives} confirmed positive boxes; have {total_positive}."
        )

    # Exact same JPEG across sessions is often a duplicate export (or
    # accidentally replayed video). Such leakage invalidates validation.
    observed = {}
    for session in sessions:
        for image in session["reviewed"]:
            digest = image["hash"]
            other = observed.setdefault(digest, session["folder"])
            if other != session["folder"]:
                raise ValueError(
                    f"Identical reviewed image in distinct sessions: "
                    f"{other.name} and {session['folder'].name}"
                )

    # Choose validation and test from qualifying independent sessions, not
    # from sessions with only ball-negative images.
    eligible = [s for s in sessions
                if s["positives"] >= min_positives_per_split
                and s["negatives"] >= min_negatives_per_split]
    if len(eligible) < 3:
        raise ValueError(
            "Need >=3 separate sessions each with at least "
            f"{min_positives_per_split} verified ball and "
            f"{min_negatives_per_split} verified no-ball examples; "
            f"have {len(eligible)} suitable sessions."
        )
    rng = random.Random(seed)
    rng.shuffle(eligible)
    val, test = eligible[:2]
    train = [s for s in sessions if s is not val and s is not test]
    split_groups = {
        "train2017": train,
        "val2017": [val],
        "test2017": [test],
    }
    summary = {}
    for split, group in split_groups.items():
        positives = sum(s["positives"] for s in group)
        negatives = sum(s["negatives"] for s in group)
        if positives < min_positives_per_split or negatives < min_negatives_per_split:
            raise ValueError(
                f"{split}: insufficient verified labels "
                f"(balls={positives}, no-ball={negatives})"
            )
        summary[split] = {
            "sessions": [s["folder"].name for s in group],
            "images": sum(len(s["reviewed"]) for s in group),
            "positive_boxes": positives,
            "negative_images": negatives,
        }

    coco_dir = target / "COCO"
    if coco_dir.exists():
        raise FileExistsError(f"Will not overwrite {coco_dir}")
    coco_dir.mkdir(parents=True)
    try:
        ann_dir = coco_dir / "annotations"
        ann_dir.mkdir()
        for split, group in split_groups.items():
            out = coco_dir / split
            out.mkdir()
            images, annotations = [], []
            for session in group:
                for item in session["reviewed"]:
                    row, boxes = item["row"], item["boxes"]
                    source_image = session["folder"] / row["filename"]
                    dest_name = session["folder"].name + "_" + Path(row["filename"]).name
                    shutil.copy2(source_image, out / dest_name)
                    # Refuse altered bytes even if the source changed during copy.
                    with (out / dest_name).open("rb") as stream:
                        copied_hash = hashlib.file_digest(stream, "sha256").hexdigest()
                    if copied_hash != item["hash"]:
                        raise ValueError(f"File changed during copy: {source_image}")
                    width, height = int(row["image_width"]), int(row["image_height"])
                    image_id = len(images) + 1
                    images.append({
                        "id": image_id, "file_name": dest_name,
                        "width": width, "height": height,
                    })
                    for cx, cy, bw, bh in boxes:
                        w, h = bw * width, bh * height
                        annotations.append({
                            "id": len(annotations) + 1,
                            "image_id": image_id, "category_id": 1,
                            "bbox": [(cx - bw / 2) * width,
                                     (cy - bh / 2) * height, w, h],
                            "area": w * h, "iscrowd": 0,
                        })
            dataset = {
                "info": {"description": "1on1 Basketball - human-reviewed direct court"},
                "licenses": [],
                "images": images,
                "annotations": annotations,
                "categories": [{"id": 1, "name": "basketball",
                                "supercategory": "sports"}],
            }
            (ann_dir / f"instances_{split}.json").write_text(
                json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        (coco_dir / "split_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        shutil.rmtree(coco_dir, ignore_errors=True)
        raise
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-positives", type=int, default=20)
    parser.add_argument("--min-positives-per-split", type=int, default=1)
    parser.add_argument("--min-negatives-per-split", type=int, default=1)
    args = parser.parse_args()
    result = export(args.prepared_root, args.output, args.seed, args.min_positives,
                    args.min_positives_per_split, args.min_negatives_per_split)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
