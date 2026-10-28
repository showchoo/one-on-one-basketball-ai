"""Regression tests for strict human-reviewed COCO export and split safety."""
import csv
import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image
from ml.export_verified_coco import export
from ml.ingest_capture import process


def make_dataset(root, negative_only_session=None, nan_session=None):
    prepared = root / "private"
    sessions = []
    for index, stamp in enumerate((100000, 200000, 300000)):
        zip_path = root / f"court_{stamp}.zip"
        raw = io.BytesIO()
        Image.new("RGB", (64, 48),
                  (40 + index * 50, 40, 70)).save(raw, format="JPEG")
        csv_buffer = io.StringIO()
        writer = csv.writer(csv_buffer)
        writer.writerow([
            "filename", "timestamp_utc_ms", "capture_elapsed_ms",
            "image_width", "image_height", "rotation_applied_deg", "device",
        ])
        for j in range(3):
            writer.writerow([
                f"images/frame_{j:05d}_{stamp + j}.jpg",
                stamp + j, j * 500, 64, 48, 0, "test"
            ])
        with zipfile.ZipFile(zip_path, "w") as archive:
            archive.writestr("frames.csv", csv_buffer.getvalue())
            for j in range(3):
                archive.writestr(
                    f"images/frame_{j:05d}_{stamp + j}.jpg",
                    raw.getvalue()
                )
        folder, _ = process(zip_path, prepared, "direct_court")
        sessions.append(folder)
        statuses = {}
        (folder / "labels").mkdir()
        for j, image_path in enumerate(sorted((folder / "images").glob("*.jpg"))):
            positive = j == 0 and stamp != negative_only_session
            stem = image_path.stem
            statuses[stem] = "positive" if positive else "negative"
            label = ("0 nan 0.5 0.1 0.1\n" if stamp == nan_session else
                     "0 0.5 0.5 0.1 0.1\n") if positive else ""
            (folder / "labels" / (stem + ".txt")).write_text(
                label, encoding="utf-8"
            )
        (folder / "review_status.json").write_text(
            json.dumps(statuses), encoding="utf-8"
        )
    return prepared, sessions


class StrictSplitTests(unittest.TestCase):
    def test_no_ball_only_session_cannot_satisfy_three_split_gate(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            prepared, _ = make_dataset(root, negative_only_session=300000)
            with self.assertRaisesRegex(ValueError, "Need >=3 separate sessions"):
                export(prepared, root / "output", min_positives=2)

    def test_invalid_nan_label_blocks_export(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            prepared, _ = make_dataset(root, nan_session=200000)
            with self.assertRaisesRegex(ValueError, "Invalid YOLO box"):
                export(prepared, root / "output", min_positives=3)
            self.assertFalse((root / "output" / "COCO").exists())

    def test_duplicate_jpeg_across_sessions_blocks_export(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            prepared, sessions = make_dataset(root)
            original = sorted((sessions[0] / "images").glob("*.jpg"))[0]
            victim = sorted((sessions[1] / "images").glob("*.jpg"))[0]
            victim.write_bytes(original.read_bytes())
            index_path = sessions[1] / "index.csv"
            with index_path.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            rows[0]["original_sha256"] = hashlib.sha256(
                victim.read_bytes()
            ).hexdigest()
            with index_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "Identical reviewed image"):
                export(prepared, root / "output", min_positives=3)

    def test_unchanged_three_sessions_export_with_negatives(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            prepared, _ = make_dataset(root)
            result = export(prepared, root / "output", min_positives=3)
            self.assertEqual(len(result), 3)
            for details in result.values():
                self.assertEqual(details["positive_boxes"], 1)
                self.assertEqual(details["negative_images"], 2)


if __name__ == "__main__":
    unittest.main()
