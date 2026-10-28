#!/usr/bin/env python3
"""Summarize cumulative street-court debug CSVs WITHOUT uploading log contents.

Repeated Drive shares often contain prefixes of the same continuously growing
on-device CSV. Merge only when that prefix relationship is proven; refuse logs
that diverge instead of multiplying frame/model counts.
"""
import argparse
import csv
import json
import re
import statistics
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

EXPECTED = {
    "timestamp", "event_type", "status", "inference_ms", "ball",
    "ball_source", "score_a", "score_b", "detail",
}
DATE_FORMAT = "%Y-%m-%d %H:%M:%S.%f"
TEMPORAL_FIELDS = {
    "inferenceMs": re.compile(r"(?:^|;)\s*inferenceMs=(\d+)(?:;|$)"),
    "rawPeak": re.compile(r"(?:^|;)\s*rawPeak=([0-9.eE+-]+)(?:;|$)"),
    "detected": re.compile(r"(?:^|;)\s*detected=(true|false)(?:;|$)"),
    "backend": re.compile(r"(?:^|;)\s*backend=([^;]+)(?:;|$)"),
}


def load_log(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not EXPECTED.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing debug columns in {path.name}")
        rows = []
        for index, row in enumerate(reader, 2):
            if None in row or not row["timestamp"] or not row["event_type"]:
                raise ValueError(f"Malformed row {index} in {path.name}")
            try:
                datetime.strptime(row["timestamp"], DATE_FORMAT)
            except ValueError as e:
                raise ValueError(f"Bad timestamp in {path.name} row {index}") from e
            rows.append(tuple(row[c] for c in reader.fieldnames))
        if not rows:
            raise ValueError(f"No debug events in {path.name}")
        return reader.fieldnames, rows


def resolve_cumulative(paths):
    """Only accept snapshots of the very same debug stream."""
    snapshots = [(*load_log(path), path.name) for path in paths]
    if not snapshots:
        raise ValueError("No CSV files supplied")
    snapshots.sort(key=lambda snap: len(snap[1]), reverse=True)
    fields, canonical, filename = snapshots[0]
    for columns, rows, name in snapshots[1:]:
        if columns != fields or canonical[:len(rows)] != rows:
            raise ValueError(
                f"Different or divergent CSV streams: {filename} and {name}. "
                "Audit these recordings separately; do not merge their counts."
            )
    return fields, canonical, [s[2] for s in snapshots]


def percentile(values, q):
    if not values:
        return None
    v = sorted(values)
    return v[max(0, min(len(v) - 1, int((len(v)-1)*q + .5)))]


def summarize(paths):
    columns, rows, filenames = resolve_cumulative(paths)
    events = [dict(zip(columns, row)) for row in rows]
    counts = Counter(row["event_type"] for row in events)
    latencies, peaks, backend_counts = [], [], Counter()
    detected = 0
    for row in events:
        if row["event_type"] != "TEMPORAL_BALL_RESULT":
            continue
        detail = row["detail"]
        def field(key):
            match = TEMPORAL_FIELDS[key].search(detail)
            return match.group(1) if match else None
        inference_ms = field("inferenceMs")
        raw_peak = field("rawPeak")
        backend = field("backend")
        status = field("detected")
        if inference_ms is not None:
            latencies.append(int(inference_ms))
        if raw_peak is not None:
            peaks.append(float(raw_peak))
        if backend:
            backend_counts[backend] += 1
        if status == "true":
            detected += 1

    # v0.7.10+: independent raw scene detector diagnostics.
    # Before this version, FRAME.ball_conf describes the fused track,
    # and cannot be used as the raw scene model's confidence.
    scene_scores, scene_times = [], []
    scenes_with_ball_candidate = 0
    scenes_forwarded_to_fusion = 0
    malformed_scene_events = 0
    for row in events:
        if row["event_type"] != "SCENE_YOLO_RESULT":
            continue
        detail = row["detail"]
        def extract(name):
            match = re.search(r"(?:^|;)\s*" + re.escape(name) +
                              r"=([^;]+)(?:;|$)", detail)
            return match.group(1).strip() if match else None
        try:
            best = float(extract("bestBallConfidence"))
            duration = int(extract("inferenceMs"))
            ball_count = int(extract("ballCandidates"))
            if not (0 <= best <= 1 and duration >= 0 and ball_count >= 0):
                raise ValueError("Invalid scene YOLO diagnostic")
        except (ValueError, TypeError):
            malformed_scene_events += 1
            continue
        scene_scores.append(best)
        scene_times.append(duration)
        if ball_count > 0:
            scenes_with_ball_candidate += 1
        if extract("acceptedForFusion") == "true":
            scenes_forwarded_to_fusion += 1

    capture_started = []
    archive_events = []
    for row in events:
        event = row["event_type"]
        if event == "TRAINING_CAPTURE_START":
            match = re.search(r"(?:^|;)\s*session=([^;]+)", row["detail"])
            capture_started.append(match.group(1) if match else "unknown")
        if event == "TRAINING_ZIP_READY":
            match = re.search(r"(?:^|;)\s*file=([^;]+)", row["detail"])
            archive_events.append(match.group(1) if match else "unknown")

    reset_times = [
        datetime.strptime(r["timestamp"], DATE_FORMAT)
        for r in events if r["event_type"] in ("RESET_PRESSED", "START_PRESSED")
    ]
    errors = [
        datetime.strptime(r["timestamp"], DATE_FORMAT)
        for r in events if r["event_type"] == "TRACKING_ERROR"
    ]
    adjacent = sum(
        any(timedelta(0) <= err - control <= timedelta(milliseconds=500)
            for control in reset_times)
        for err in errors
    )

    return {
        "source": "one-device cumulative debug logs",
        "csv_snapshots": filenames,
        "snapshots_are_nested_prefixes": True,
        "rows": len(events),
        "event_counts": dict(sorted(counts.items())),
        "ball_frame_entries": sum(
            1 for row in events if row["event_type"] == "FRAME" and row["ball"]
        ),
        "temporal_model": {
            "inferences": counts["TEMPORAL_BALL_RESULT"],
            "detected_true": detected,
            "inference_ms_p50": percentile(latencies, .5),
            "inference_ms_p95": percentile(latencies, .95),
            "raw_peak_p50": percentile(peaks, .5),
            "raw_peak_max": max(peaks) if peaks else None,
            "backends": dict(backend_counts),
            "warning": "Accepted detections are not ground-truth recall.",
        },
        "scene_yolo_model": {
            "diagnostic_available": counts["SCENE_YOLO_RESULT"] > 0,
            "reports": counts["SCENE_YOLO_RESULT"],
            "parsed_reports": len(scene_scores),
            "malformed_reports": malformed_scene_events,
            "with_ball_candidate": (scenes_with_ball_candidate if scene_scores else None),
            "best_ball_confidence_max": (max(scene_scores) if scene_scores else None),
            "inference_ms_p50": percentile(scene_times, .5),
            "already_locked_for_fusion": (
                scenes_forwarded_to_fusion if scene_scores else None
            ),
            "note": "Raw scene candidates are not confirmed ball tracks or scoring accuracy.",
        },
        "capture": {
            "start_events": len(capture_started),
            "started_session_ids": capture_started,
            "zip_ready_events": len(archive_events),
            "distinct_ready_archives": sorted(set(archive_events)),
        },
        "tracking_errors": {
            "count": len(errors),
            "within_500ms_of_start_or_reset": adjacent,
            "note": "Temporal association alone does not prove root cause.",
        },
        "user_images_examined": False,
        "model_retrained": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", type=Path, nargs="+",
                        help="Private CSV exports of ONE continuously growing app log")
    parser.add_argument("--report", type=Path, help="Optional metadata-only JSON")
    args = parser.parse_args()
    result = summarize(args.logs)
    body = json.dumps(result, ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(body + "\n", encoding="utf-8")
    print(body)


if __name__ == "__main__":
    main()
