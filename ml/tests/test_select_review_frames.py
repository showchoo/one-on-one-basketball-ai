"""Tests for selecting private images without inventing ball labels."""
import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from ml.select_review_frames import build_queue, _safe_frame


class ReviewQueueTests(unittest.TestCase):
    def create_session(self, root):
        session = Path(root) / "capture_1800000000000"
        images = session / "images"
        images.mkdir(parents=True)
        with (session / "index.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "timestamp_utc_ms", "image_width", "image_height"])
            for i in range(12):
                name = f"images/frame_{i + 1:05d}_{1800000000000 + 500 * i}.jpg"
                Image.new("RGB", (48, 32),
                          (min(255, i * 20), 15, 45)).save(session / name)
                writer.writerow([name, 1800000000000 + 500 * i, 48, 32])
        return session

    def test_sampling_is_bounded_unique_ordered_and_reproducible(self):
        with tempfile.TemporaryDirectory() as d:
            session = self.create_session(d)
            first = build_queue(session, 6)
            with (session / "review_queue.csv").open(encoding="utf-8") as f:
                q1 = list(csv.DictReader(f))
            second = build_queue(session, 6)
            with (session / "review_queue.csv").open(encoding="utf-8") as f:
                q2 = list(csv.DictReader(f))
            self.assertEqual(first["source_frames"], 12)
            self.assertEqual(first["selected"], 6)
            self.assertEqual(first["labels_created"], 0)
            self.assertEqual(q1, q2)
            names = [x["filename"] for x in q1]
            self.assertEqual(len(set(names)), 6)
            self.assertEqual(len(q1), 6)
            self.assertTrue(all(float(x["motion_score"]) >= 0 for x in q1))
            self.assertEqual([x["selection_policy"] for x in q1],
                             ["high-motion", "low-motion", "middle-motion"] * 2)
            self.assertEqual(list(session.glob("labels/*.txt")), [])

    def test_request_more_than_present_selects_each_once(self):
        with tempfile.TemporaryDirectory() as d:
            session = self.create_session(d)
            self.assertEqual(build_queue(session, 100)["selected"], 12)

    def test_bad_private_file_path_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            session = self.create_session(d)
            with self.assertRaises(ValueError):
                _safe_frame(session, "../other.jpg")
            with self.assertRaises(ValueError):
                _safe_frame(session, "images/../../private.jpg")

    def test_missing_image_errors_without_fake_labels(self):
        with tempfile.TemporaryDirectory() as d:
            session = self.create_session(d)
            next((session / "images").glob("*.jpg")).unlink()
            with self.assertRaises(FileNotFoundError):
                build_queue(session, 5)


if __name__ == "__main__":
    unittest.main()
