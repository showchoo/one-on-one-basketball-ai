"""Unit tests for private, session-safe training data tools."""
import csv
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image
from ml.ingest_capture import process
from ml.export_verified_coco import export


class DatasetTests(unittest.TestCase):
    def make_capture(self, where: Path, stamp: int):
        zpath = where / f"example_{stamp}.zip"
        image = Image.new("RGB", (64, 48), (15, 45, 90))
        buf = io.BytesIO()
        image.save(buf, "JPEG")
        header = [
            "filename", "timestamp_utc_ms", "capture_elapsed_ms",
            "image_width", "image_height", "rotation_applied_deg", "device"
        ]
        rows = [
            [f"images/frame_{i:05d}_{stamp + i}.jpg", str(stamp + i), str(500 + i),
             "64", "48", "0", "test"]
            for i in range(1, 4)
        ]
        strbuf = io.StringIO()
        writer = csv.writer(strbuf)
        writer.writerow(header)
        writer.writerows(rows)
        with zipfile.ZipFile(zpath, "w") as archive:
            archive.writestr("frames.csv", strbuf.getvalue())
            for row in rows:
                archive.writestr(row[0], buf.getvalue())
        return zpath

    def test_ingest_and_screen_separation(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = self.make_capture(root, 100000)
            session, summary = process(source, root / "work", "screen_replay")
            self.assertEqual(summary["frames"], 3)
            self.assertFalse(summary["trainable"])
            self.assertEqual(len(list((session / "images").glob("*.jpg"))), 3)
            with self.assertRaises(ValueError):
                process(self.make_capture(root, 200000), root / "invalid",
                        "direct_court", crop=(2, 2, 50, 40))

    def test_no_session_leakage(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            prepared = root / "prepared"
            for stamp in (100000, 200000, 300000):
                session, _ = process(
                    self.make_capture(root, stamp), prepared, "direct_court"
                )
                statuses = {}
                for idx, image_file in enumerate(sorted((session / "images").glob("*.jpg"))):
                    name = image_file.stem
                    statuses[name] = "positive" if idx == 0 else "negative"
                    (session / "labels").mkdir(exist_ok=True)
                    (session / "labels" / (name + ".txt")).write_text(
                        "0 0.5 0.5 0.1 0.1\n" if idx == 0 else "", encoding="utf-8"
                    )
                (session / "review_status.json").write_text(json.dumps(statuses))
            report = export(prepared, root / "final", min_positives=3)
            self.assertEqual(
                set(report), {"train2017", "val2017", "test2017"}
            )
            self.assertTrue(all(v["positive_boxes"] == 1 for v in report.values()))
            self.assertEqual(
                len(set(s for part in report.values() for s in part["sessions"])), 3
            )
            payload = json.loads(
                (root / "final" / "COCO" / "annotations" /
                 "instances_train2017.json").read_text()
            )
            self.assertEqual(payload["categories"][0]["name"], "basketball")
            self.assertEqual(len(payload["annotations"]), 1)

    def test_replay_cannot_enter_training_split(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            prepared = root / "prepared"
            session, _ = process(
                self.make_capture(root, 400000), prepared, "screen_replay"
            )
            (session / "review_status.json").write_text('{"frame_00001_400001": "positive"}')
            (session / "labels").mkdir()
            (session / "labels" / "frame_00001_400001.txt").write_text(
                "0 0.5 0.5 0.2 0.2\n"
            )
            with self.assertRaisesRegex(ValueError, "Need >=3"):
                export(prepared, root / "final", min_positives=1)


if __name__ == "__main__":
    unittest.main()
