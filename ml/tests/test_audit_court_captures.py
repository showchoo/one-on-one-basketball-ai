"""Tests for metadata-only private real-court capture audits."""
import csv
import io
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image
from ml.audit_court_captures import audit, inspect_capture


class AuditCapturesTests(unittest.TestCase):
    def make_capture(self, folder, name, first_ms=1800000000000, missing=False):
        folder = Path(folder)
        image = Image.new("RGB", (64, 48), (80, 70, 60))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        csv_file = io.StringIO()
        writer = csv.writer(csv_file)
        writer.writerow([
            "filename", "timestamp_utc_ms", "capture_elapsed_ms",
            "image_width", "image_height", "rotation_applied_deg", "device"
        ])
        for index in range(3):
            writer.writerow([
                f"images/frame_{index + 1:05d}_{first_ms + 500 * index}.jpg",
                first_ms + 500 * index, 1000 + 500 * index, 64, 48, 0, "test"
            ])
        dest = folder / name
        with zipfile.ZipFile(dest, "w") as z:
            z.writestr("frames.csv", csv_file.getvalue())
            for index in range(3 if not missing else 2):
                z.writestr(
                    f"images/frame_{index + 1:05d}_{first_ms + 500 * index}.jpg",
                    buffer.getvalue()
                )
        return dest

    def test_metadata_and_real_jpeg_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            z = self.make_capture(temp, "sample.zip")
            info = inspect_capture(z, verify_jpegs=True)
            self.assertEqual(info["frames"], 3)
            self.assertEqual(info["median_interval_ms"], 500)
            self.assertEqual(info["dimensions"], [[64, 48]])
            self.assertTrue(info["crc_ok"])
            self.assertTrue(info["jpeg_verified"])

    def test_duplicate_shares_do_not_count_as_distinct_sessions(self):
        with tempfile.TemporaryDirectory() as temp:
            first = self.make_capture(temp, "shared_twice_a.zip")
            second = self.make_capture(temp, "shared_twice_b.zip")
            report = audit([first, second])
            self.assertEqual(report["unique_session_count"], 1)
            self.assertEqual(report["unique_frame_count"], 3)
            self.assertFalse(report["ready_for_annotation"])
            self.assertEqual(report["sessions"][0]["duplicates"], ["shared_twice_b.zip"])

    def test_three_real_sessions_pass_metadata_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = [
                self.make_capture(temp, f"court_{n}.zip", first_ms=1800000000000 + n * 600000)
                for n in range(3)
            ]
            report = audit(paths)
            self.assertEqual(report["unique_session_count"], 3)
            self.assertEqual(report["unique_frame_count"], 9)
            self.assertTrue(report["ready_for_annotation"])
            self.assertFalse(report["model_trained"])

    def test_missing_referenced_jpeg_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            z = self.make_capture(temp, "broken.zip", missing=True)
            report = audit([z])
            self.assertEqual(report["readable_zip_count"], 0)
            self.assertEqual(report["unique_session_count"], 0)
            self.assertIn("manifest", report["failures"][0]["error"].lower())


if __name__ == "__main__":
    unittest.main()
