#!/usr/bin/env python3
"""One-command PRIVATE real-court ZIP audit, deduplication, ingest and review queues.

Run on a local PC; it never contacts GitHub, Google, or any external service.
The original ZIPs are only read, never modified or deleted.
"""
import argparse
import json
import shutil
from pathlib import Path

from ml.audit_court_captures import audit
from ml.ingest_capture import process
from ml.select_review_frames import build_queue


def prepare(source_dir: Path, target_dir: Path, per_session=120, audit_only=False):
    source_dir = source_dir.resolve()
    target_dir = target_dir.resolve()
    archives = sorted(p for p in source_dir.glob("*.zip") if p.is_file())
    if not archives:
        raise ValueError("No ZIP files in " + str(source_dir))

    report = audit(archives, verify_jpegs=True)
    report["source_type"] = "direct_court"
    report["private_local_only"] = True
    if audit_only:
        return report
    if not report["ready_for_annotation"]:
        raise ValueError("Capture audit did not pass:\n" +
                         json.dumps(report["failures"], ensure_ascii=False, indent=2) +
                         f"\nUnique sessions: {report['unique_session_count']}")
    if target_dir.exists():
        raise FileExistsError("Refusing to overwrite prepared data: " + str(target_dir))
    staging = target_dir.with_name(target_dir.name + ".partial")
    if staging.exists():
        raise FileExistsError("Existing partial output must be reviewed: " + str(staging))
    staging.parent.mkdir(parents=True, exist_ok=True)
    try:
        ingest_summaries = []
        for item in report["sessions"]:
            original = source_dir / item["canonical_zip"]
            session, ingest_summary = process(
                original, staging, "direct_court"
            )
            queue_summary = build_queue(session, per_session)
            ingest_summaries.append({
                **ingest_summary,
                "review_queue": queue_summary,
                "original_zip": original.name,
            })
        report["prepared_sessions"] = ingest_summaries
        report["requires_human_ball_labels"] = True
        # A metadata-only report may include filenames and timestamps but NO images.
        (staging / "court_preparation_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        staging.rename(target_dir)
        return report
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path,
                        help="Local private folder containing the original Drive ZIP downloads")
    parser.add_argument("--output", type=Path,
                        default=Path("private_training_data/direct_court"))
    parser.add_argument("--per-session", type=int, default=120)
    parser.add_argument("--audit-only", action="store_true",
                        help="Validate ZIPs and report duplicate sessions, no image extraction")
    args = parser.parse_args()
    result = prepare(args.source_dir, args.output, args.per_session,
                     audit_only=args.audit_only)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["ready_for_annotation"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
