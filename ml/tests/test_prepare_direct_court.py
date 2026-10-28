"""Tests for one-command PRIVATE preparation of synthetic direct-court images."""
import csv
import io
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image
from ml.prepare_direct_court import prepare


def capture(folder: Path, name: str, first_timestamp: int):
    raw_image = io.BytesIO()
    Image.new("RGB", (64, 48), (20, 140, 40)).save(raw_image, "JPEG")
    manifest = io.StringIO()
    writer = csv.writer(manifest)
    writer.writerow([
        "filename", "timestamp_utc_ms", "capture_elapsed_ms",
        "image_width", "image_height", "rotation_applied_deg", "device"
    ])
    for n in range(5):
        writer.writerow([
            f"images/frame_{n + 1:05d}_{first_timestamp + 500 * n}.jpg",
            first_timestamp + 500 * n, 500 * n, 64, 48, 0, "synthetic"
        ])
    archive = folder / name
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("frames.csv", manifest.getvalue())
        for n in range(5):
            z.writestr(
                f"images/frame_{n + 1:05d}_{first_timestamp + 500 * n}.jpg",
                raw_image.getvalue()
            )
    return archive


class PrivatePreparationTests(unittest.TestCase):
    def test_duplicate_shares_are_skipped_and_three_sessions_ingested(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "drive_downloads"
            source.mkdir()
            for n in range(3):
                capture(source, f"court_{n}.zip", 1800000000000 + n * 600000)
            capture(source, "duplicate_0.zip", 1800000000000)
            dest = root / "prepared"
            summary = prepare(source, dest, per_session=3)
            self.assertEqual(summary["input_zip_count"], 4)
            self.assertEqual(summary["unique_session_count"], 3)
            self.assertEqual(summary["unique_frame_count"], 15)
            self.assertTrue(summary["requires_human_ball_labels"])
            self.assertEqual(len(summary["prepared_sessions"]), 3)
            self.assertTrue((dest / "court_preparation_report.json").is_file())
            self.assertEqual(len(list(dest.glob("sessions/capture_*/review_queue.csv"))), 3)
            self.assertEqual(len(list(dest.glob("sessions/capture_*/images/*.jpg"))), 15)
            self.assertFalse(list(dest.rglob("labels/*.txt")))
            self.assertTrue((source / "duplicate_0.zip").is_file())

    def test_audit_only_does_not_extract(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "zips"
            source.mkdir()
            capture(source, "one.zip", 1800000000000)
            result = prepare(source, root / "never_created", audit_only=True)
            self.assertEqual(result["unique_session_count"], 1)
            self.assertFalse(result["ready_for_annotation"])
            self.assertFalse((root / "never_created").exists())

    def test_reject_fewer_than_three_distinct_sessions(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "zips"
            source.mkdir()
            capture(source, "one.zip", 1800000000000)
            with self.assertRaisesRegex(ValueError, "Unique sessions: 1"):
                prepare(source, root / "work", per_session=3)

    def test_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "zips"
            source.mkdir()
            for n in range(3):
                capture(source, f"session_{n}.zip", 1800000000000 + n * 600000)
            dest = root / "existing"
            dest.mkdir()
            (dest / "keep").write_text("do not erase", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                prepare(source, dest, per_session=3)
            self.assertEqual((dest / "keep").read_text(encoding="utf-8"),
                             "do not erase")


if __name__ == "__main__":
    unittest.main()
