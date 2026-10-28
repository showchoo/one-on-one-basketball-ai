#!/usr/bin/env python3
"""Suggest, NEVER auto-label, ball boxes for private direct-court review.

Runs a local YOLO checkpoint on frames selected by review_queue.csv.
Writes metadata-only proposals.json, not any confirmed label or training set.
Use the reviewed-label UI to accept/reject each proposed ball independently.
"""
import argparse
import csv
import hashlib
import json
import math
import urllib.request
from pathlib import Path

DEFAULT_E_BARD_URL = (
    "https://huggingface.co/GabrieleGiudici/E-BARD-detection-models/"
    "resolve/main/BODD_yolov8n_0001.pt"
)


def safe_review_frames(session: Path) -> list:
    index_file = session / "index.csv"
    queue_file = session / "review_queue.csv"
    if not index_file.is_file() or not queue_file.is_file():
        raise FileNotFoundError("Run private ingest and review queue selection first")
    with index_file.open(newline="", encoding="utf-8") as stream:
        indexed = {row["filename"] for row in csv.DictReader(stream)}
    with queue_file.open(newline="", encoding="utf-8") as stream:
        selected = [row["filename"] for row in csv.DictReader(stream)]
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("Review queue must be nonempty with unique frame paths")
    frames = []
    for relative in selected:
        p = Path(relative)
        if (relative not in indexed or p.is_absolute() or
                p.parent != Path("images") or p.suffix.lower() != ".jpg"):
            raise ValueError("Invalid or unknown review frame: " + relative)
        absolute = session / relative
        if not absolute.is_file():
            raise FileNotFoundError(absolute)
        frames.append(absolute)
    return frames


def get_ball_class(model) -> int:
    names = model.names
    if isinstance(names, dict):
        matched = [int(key) for key, value in names.items()
                   if str(value).strip().lower() in ("basketball", "sports ball")]
    else:
        matched = [i for i, value in enumerate(names)
                   if str(value).strip().lower() in ("basketball", "sports ball")]
    if len(matched) != 1:
        raise ValueError("Expected exactly one basketball class in checkpoint")
    return matched[0]


def read_ball_candidates(prediction, class_id: int):
    """Convert unverified local YOLO boxes to normalized metadata."""
    proposals = []
    for candidate in prediction.boxes:
        cls = int(candidate.cls[0].item())
        if cls != class_id:
            continue
        conf = float(candidate.conf[0].item())
        box = [float(v) for v in candidate.xywhn[0].tolist()]
        if len(box) != 4 or not all(math.isfinite(v) for v in box + [conf]):
            continue
        cx, cy, w, h = box
        if (not 0.0 <= conf <= 1.0 or
                not (0.0 < w <= 1.0 and 0.0 < h <= 1.0 and
                     0.0 <= cx - w / 2 < cx + w / 2 <= 1.0 and
                     0.0 <= cy - h / 2 < cy + h / 2 <= 1.0)):
            continue
        proposals.append({"box": box, "confidence": round(conf, 6)})
    return sorted(proposals, key=lambda result: result["confidence"], reverse=True)


def propose(session: Path, weights: Path, min_conf=0.04, model_factory=None):
    if not (0.001 <= min_conf <= 0.95):
        raise ValueError("Confidence floor must be between 0.001 and 0.95")
    frames = safe_review_frames(session)
    if not weights.is_file():
        raise FileNotFoundError(weights)
    if model_factory is None:
        # Lazy dependency: not needed to run lightweight synthetic CI tests.
        from ultralytics import YOLO
        model_factory = YOLO
    model = model_factory(str(weights))
    class_id = get_ball_class(model)
    with weights.open("rb") as stream:
        weight_sha = hashlib.file_digest(stream, "sha256").hexdigest()

    proposals = {}
    for photo in frames:
        result = model.predict(source=str(photo), conf=min_conf,
                               imgsz=640, verbose=False)[0]
        candidates = read_ball_candidates(result, class_id)
        # Keep one strongest suggestion only. Do NOT save it as a label.
        if candidates:
            with photo.open("rb") as frame_stream:
                digest = hashlib.file_digest(frame_stream, "sha256").hexdigest()
            proposals[photo.stem] = {**candidates[0], "image_sha256": digest}
        else:
            proposals[photo.stem] = None
    output = {
        "schema_version": 1,
        "model_weights_sha256": weight_sha,
        "model": weights.name,
        "verified": False,
        "positive_labels_created": 0,
        "frames_inspected": len(frames),
        "proposals": proposals,
        "note": "Unverified hints; each ball must be confirmed using label_ball.py.",
    }
    path = session / "proposals.json"
    if path.exists():
        raise FileExistsError("Review existing proposals before overwriting: " + str(path))
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--weights", type=Path, required=True,
                        help="Path to a LOCAL YOLO model such as E-BARD BODD_yolov8n_0001.pt")
    parser.add_argument("--download-e-bard", action="store_true",
                        help="Download only the public model weights to --weights if missing")
    parser.add_argument("--min-confidence", type=float, default=0.04)
    args = parser.parse_args()
    if args.download_e_bard and not args.weights.exists():
        args.weights.parent.mkdir(parents=True, exist_ok=True)
        temp_path = args.weights.with_suffix(".download-part")
        try:
            urllib.request.urlretrieve(DEFAULT_E_BARD_URL, temp_path)
            if temp_path.stat().st_size < 100000:
                raise ValueError("Downloaded model weights are implausibly small")
            temp_path.rename(args.weights)
        finally:
            temp_path.unlink(missing_ok=True)
    summary = propose(args.session, args.weights, args.min_confidence)
    print(json.dumps({
        "session": args.session.name,
        "frames": summary["frames_inspected"],
        "has_proposal": sum(item is not None
                            for item in summary["proposals"].values()),
        "labels_created": summary["positive_labels_created"],
        "output": str(args.session / "proposals.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
