"""Checks private review progress reporting with synthetic frames."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

from ml.report_label_progress import report


class ProgressTests(unittest.TestCase):
    def test_report_distinguishes_positive_negative_unclear_unreviewed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / "sessions" / "capture_1"
            folder.mkdir(parents=True)
            rows = ["images/frame_1.jpg", "images/frame_2.jpg",
                    "images/frame_3.jpg", "images/frame_4.jpg"]
            for filename in ("index.csv", "review_queue.csv"):
                with (folder / filename).open("w", encoding="utf-8",
                                              newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["filename"])
                    for row in rows:
                        writer.writerow([row])
            (folder / "labels").mkdir()
            (folder / "labels" / "frame_1.txt").write_text(
                "0 0.5 0.5 0.1 0.1\n", encoding="utf-8"
            )
            (folder / "labels" / "frame_2.txt").write_text("", encoding="utf-8")
            (folder / "review_status.json").write_text(
                json.dumps({"frame_1": "positive", "frame_2": "negative",
                            "frame_3": "unclear"}), encoding="utf-8"
            )
            result = report(root)
            self.assertEqual(result["session_count"], 1)
            self.assertEqual(result["ready_for_split_count"], 1)
            self.assertEqual(result["confirmed_positive_images"], 1)
            self.assertEqual(result["confirmed_no_ball_images"], 1)
            self.assertEqual(result["unreviewed_queue_images"], 1)
            self.assertEqual(result["sessions"][0]["reviewed_percent"], 50.0)

    def test_missing_confirmed_label_is_error(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "sessions" / "capture_2"
            folder.mkdir(parents=True)
            (folder / "index.csv").write_text("filename\nimages/a.jpg\n",
                                             encoding="utf-8")
            (folder / "review_status.json").write_text(
                '{"a": "positive"}', encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "Missing human-reviewed label"):
                report(Path(temp))


if __name__ == "__main__":
    unittest.main()
