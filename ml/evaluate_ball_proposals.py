#!/usr/bin/env python3
"""Evaluate unverified offline E-BARD proposals against confirmed human labels.

A proposal alone is NOT ground truth. Only frames with human positive/negative
labels contribute; positives need IoU >= threshold to count as true positives.
Output has no raw frames or personally identifying image content.
"""
import argparse
import json
import math
from pathlib import Path

from ml.label_ball import verified_suggestion


def box_iou_xywh(a, b):
    if len(a) != 4 or len(b) != 4:
        raise ValueError("Box must be cx,cy,w,h")
    if not all(math.isfinite(float(x)) for x in [*a, *b]):
        raise ValueError("Non-finite rectangle")
    ax, ay, aw, ah = [float(v) for v in a]
    bx, by, bw, bh = [float(v) for v in b]
    a_left, a_top, a_right, a_bottom = (ax-aw/2, ay-ah/2, ax+aw/2, ay+ah/2)
    b_left, b_top, b_right, b_bottom = (bx-bw/2, by-bh/2, bx+bw/2, by+bh/2)
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0
    w = max(0.0, min(a_right, b_right) - max(a_left, b_left))
    h = max(0.0, min(a_bottom, b_bottom) - max(a_top, b_top))
    intersection = w * h
    return intersection / (aw*ah + bw*bh - intersection)


def confirmed_ball(session: Path, stem: str, status: str):
    path = session / "labels" / (stem + ".txt")
    if not path.is_file():
        raise FileNotFoundError(path)
    data = path.read_text(encoding="utf-8").splitlines()
    if status == "negative":
        if data:
            raise ValueError("No-ball label must be empty: " + str(path))
        return None
    if status != "positive" or len(data) != 1:
        raise ValueError("Positive frame requires exactly one ball box")
    parts = data[0].split()
    if len(parts) != 5 or parts[0] != "0":
        raise ValueError("Unexpected annotation class")
    try:
        box = [float(x) for x in parts[1:]]
    except ValueError as e:
        raise ValueError("Invalid numeric ball label") from e
    if not (all(math.isfinite(x) for x in box)
            and 0 < box[2] <= 1 and 0 < box[3] <= 1
            and 0 <= box[0] - box[2]/2 < box[0] + box[2]/2 <= 1
            and 0 <= box[1] - box[3]/2 < box[1] + box[3]/2 <= 1):
        raise ValueError("Invalid normalized human ball box")
    return box


def summarize(counts):
    tp, fp, fn, tn = (counts[k] for k in ("tp", "fp", "fn", "tn"))
    return {
        **counts,
        "precision": round(tp/(tp+fp), 4) if tp+fp else None,
        "recall": round(tp/(tp+fn), 4) if tp+fn else None,
        "no_ball_false_positive_rate": (
            round(counts["negative_fp"]/counts["negative_total"], 4)
            if counts["negative_total"] else None
        ),
        "reviewed": counts["positive_total"] + counts["negative_total"],
    }


def evaluate_session(session: Path, thresholds=(.04, .08, .15, .25, .5),
                     min_iou=.5):
    if not (0 < min_iou <= 1):
        raise ValueError("IoU must be between 0 and 1")
    proposals_path = session / "proposals.json"
    statuses_path = session / "review_status.json"
    if not proposals_path.is_file() or not statuses_path.is_file():
        raise FileNotFoundError("Need local proposals.json and human review_status.json")
    dataset = json.loads(proposals_path.read_text(encoding="utf-8"))
    if dataset.get("schema_version") != 1:
        raise ValueError("Unsupported unverified proposal format")
    statuses = json.loads(statuses_path.read_text(encoding="utf-8"))
    hints = dataset["proposals"]
    if not isinstance(hints, dict):
        raise ValueError("Expected proposals mapping")

    observations = []
    for stem, entry in sorted(statuses.items()):
        if entry not in ("positive", "negative"):
            continue
        photo = session / "images" / (stem + ".jpg")
        if not photo.is_file():
            raise FileNotFoundError(photo)
        actual = confirmed_ball(session, stem, entry)
        hint = verified_suggestion(hints.get(stem), photo)
        observations.append((actual, hint))
    if not observations:
        raise ValueError("No confirmed ball/no-ball reviews to evaluate")

    result = {}
    for floor in thresholds:
        if not (0 <= floor <= 1):
            raise ValueError("Invalid confidence threshold")
        counts = dict(tp=0, fp=0, fn=0, tn=0,
                      positive_total=0, negative_total=0, negative_fp=0)
        for actual, hint in observations:
            predicted = (hint is not None and
                         hint["confidence"] >= floor)
            if actual is None:
                counts["negative_total"] += 1
                if predicted:
                    counts["fp"] += 1
                    counts["negative_fp"] += 1
                else:
                    counts["tn"] += 1
            else:
                counts["positive_total"] += 1
                if not predicted:
                    counts["fn"] += 1
                elif box_iou_xywh(actual, hint["box"]) >= min_iou:
                    counts["tp"] += 1
                else:
                    # An incorrect location both misses the true ball and
                    # creates one spurious ball candidate.
                    counts["fp"] += 1
                    counts["fn"] += 1
        result[f"{floor:g}"] = summarize(counts)
    return {
        "session": session.name,
        "model": dataset.get("model"),
        "model_sha256": dataset.get("model_weights_sha256"),
        "iou_threshold": min_iou,
        "thresholds": result,
    }


def evaluate_root(root: Path, thresholds=(.04,.08,.15,.25,.5), min_iou=.5):
    sessions = []
    missing = []
    for folder in sorted((root / "sessions").glob("capture_*")):
        if (folder / "proposals.json").is_file() and (folder / "review_status.json").is_file():
            sessions.append(evaluate_session(folder, thresholds, min_iou))
        else:
            missing.append(folder.name)
    if not sessions:
        raise ValueError("No sessions with model suggestions and confirmed reviews")
    aggregate = {}
    for floor in [f"{v:g}" for v in thresholds]:
        totals = {k: 0 for k in ("tp","fp","fn","tn","positive_total",
                                "negative_total","negative_fp")}
        for session in sessions:
            for key in totals:
                totals[key] += session["thresholds"][floor][key]
        aggregate[floor] = summarize(totals)
    return {
        "model_comparison_only": True,
        "sessions": sessions,
        "missing_sessions": missing,
        "aggregate": aggregate,
        "note": "Human-labeled still frames at one venue cannot validate real-time scoring accuracy.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("private_root", type=Path)
    parser.add_argument("--thresholds", default=".04,.08,.15,.25,.5",
                        help="Comma-separated score thresholds")
    parser.add_argument("--iou", type=float, default=.5)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    threshold_list = tuple(float(x) for x in args.thresholds.split(","))
    result = evaluate_root(args.private_root, threshold_list, args.iou)
    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
