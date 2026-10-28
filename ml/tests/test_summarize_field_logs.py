"""Synthetic tests for cumulative Android CSV log analysis."""
import csv
import tempfile
import unittest
from pathlib import Path

from ml.summarize_field_logs import summarize


FIELDS = [
    "timestamp", "event_type", "status", "inference_ms", "detection_count",
    "player_a", "player_b", "ball", "ball_conf", "ball_source", "roi_passes",
    "last_possessor", "shot_player", "shot_value", "score_a", "score_b",
    "points", "detail",
]


def sample_events():
    return [
        ("2026-10-28 10:00:00.000", "APP_START", ""),
        ("2026-10-28 10:00:01.000", "TRAINING_CAPTURE_START",
         "session=session_20261028_100000; intervalMs=500"),
        ("2026-10-28 10:00:01.500", "FRAME", ""),
        ("2026-10-28 10:00:02.000", "TEMPORAL_BALL_RESULT",
         "detected=false; rawPeak=0.014; inferenceMs=440; backend=NNAPI_FP16"),
        ("2026-10-28 10:00:02.200", "TEMPORAL_BALL_RESULT",
         "detected=true; rawPeak=0.33; inferenceMs=480; backend=NNAPI_FP16"),
        ("2026-10-28 10:00:02.300", "TRAINING_ZIP_READY",
         "file=session_20261028_100000.zip; bytes=12345"),
        ("2026-10-28 10:00:02.400", "TRAINING_ZIP_READY",
         "file=session_20261028_100000.zip; bytes=12345"),
        ("2026-10-28 10:00:03.000", "RESET_PRESSED", ""),
        ("2026-10-28 10:00:03.150", "TRACKING_ERROR",
         "java.lang.ArithmeticException: divide by zero"),
    ]


def save_csv(path, events):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for timestamp, event_type, detail in events:
            row = {key: "" for key in FIELDS}
            row["timestamp"] = timestamp
            row["event_type"] = event_type
            row["detail"] = detail
            writer.writerow(row)


class FieldLogTests(unittest.TestCase):
    def test_prefix_snapshots_count_once(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            earlier, latest = root / "first.csv", root / "latest.csv"
            events = sample_events()
            save_csv(earlier, events[:4])
            save_csv(latest, events)
            result = summarize([earlier, latest])
            self.assertEqual(result["rows"], 9)
            self.assertEqual(result["event_counts"]["TEMPORAL_BALL_RESULT"], 2)
            self.assertEqual(result["temporal_model"]["detected_true"], 1)
            self.assertEqual(result["temporal_model"]["inference_ms_p50"], 480)
            self.assertEqual(result["capture"]["zip_ready_events"], 2)
            self.assertEqual(len(result["capture"]["distinct_ready_archives"]), 1)
            self.assertEqual(result["tracking_errors"]["within_500ms_of_start_or_reset"], 1)

    def test_divergent_csvs_refused_not_summed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            p, q = root / "a.csv", root / "b.csv"
            events = sample_events()
            save_csv(p, events[:4])
            changed = list(events[:4])
            changed[3] = ("2026-10-28 10:00:02.000", "TEMPORAL_BALL_RESULT",
                          "detected=true; rawPeak=0.3; inferenceMs=480")
            save_csv(q, changed)
            with self.assertRaisesRegex(ValueError, "Different or divergent"):
                summarize([p, q])

    def test_scene_candidates_are_not_confused_with_fused_track(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root / "scene.csv"
            original = sample_events()
            original.append((
                "2026-10-28 10:00:05.000", "SCENE_YOLO_RESULT",
                "ballCandidates=1; bestBallConfidence=0.42; "
                "playerCandidates=2; hoopCandidates=1; inferenceMs=121; "
                "acceptedForFusion=false"
            ))
            original.append((
                "2026-10-28 10:00:06.000", "SCENE_YOLO_RESULT",
                "ballCandidates=0; bestBallConfidence=0.0; "
                "playerCandidates=2; hoopCandidates=1; inferenceMs=103; "
                "acceptedForFusion=true"
            ))
            save_csv(path, original)
            r = summarize([path])["scene_yolo_model"]
            self.assertTrue(r["diagnostic_available"])
            self.assertEqual(r["reports"], 2)
            self.assertEqual(r["with_ball_candidate"], 1)
            self.assertEqual(r["best_ball_confidence_max"], 0.42)
            self.assertEqual(r["already_locked_for_fusion"], 1)
            self.assertEqual(r["inference_ms_p50"], 121)

    def test_old_logs_do_not_invent_scene_model_metrics(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "old.csv"
            save_csv(path, sample_events())
            r = summarize([path])["scene_yolo_model"]
            self.assertFalse(r["diagnostic_available"])
            self.assertIsNone(r["best_ball_confidence_max"])
            self.assertIsNone(r["with_ball_candidate"])

    def test_no_frames_in_short_log_is_not_false_recall(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            p = root / "only_app_start.csv"
            save_csv(p, sample_events()[:1])
            result = summarize([p])
            self.assertEqual(result["temporal_model"]["inferences"], 0)
            self.assertIsNone(result["temporal_model"]["raw_peak_max"])
            self.assertEqual(result["temporal_model"]["detected_true"], 0)
            self.assertFalse(result["model_retrained"])


if __name__ == "__main__":
    unittest.main()
