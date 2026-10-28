"""Tests for offline, non-labeling private frame contact sheets."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image
from ml.make_review_sheets import build_sheets


class ContactSheetTests(unittest.TestCase):
    def create(self, root):
        session = Path(root) / "capture_1"
        (session / "images").mkdir(parents=True)
        files = []
        for i in range(7):
            name = f"images/frame_{i:04d}.jpg"
            Image.new("RGB", (64, 48), (20 * i, 50, 120)).save(session / name)
            files.append(name)
        for csv_name in ("index.csv", "review_queue.csv"):
            with (session / csv_name).open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["filename"])
                for name in files:
                    writer.writerow([name])
        return session

    def test_pages_are_created_without_labels(self):
        with tempfile.TemporaryDirectory() as td:
            session = self.create(td)
            output = Path(td) / "private_sheets"
            result = build_sheets(session, output, columns=2, rows=2)
            self.assertEqual(result["sheets"], 2)
            self.assertEqual(result["frames"], 7)
            self.assertEqual(result["labels_generated"], 0)
            self.assertEqual(len(list(output.glob("sheet_*.jpg"))), 2)
            summary = json.loads((output / "contact_sheet_index.json").read_text())
            self.assertEqual(len(summary["frames"]), 7)
            self.assertFalse((session / "labels").exists())
            with self.assertRaises(FileExistsError):
                build_sheets(session, output, columns=2, rows=2)

    def test_invalid_grid_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            session = self.create(td)
            with self.assertRaises(ValueError):
                build_sheets(session, Path(td) / "out", columns=0)

    def test_unknown_frame_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            session = self.create(td)
            with (session / "review_queue.csv").open("a", encoding="utf-8") as stream:
                stream.write("images/not_in_index.jpg\n")
            with self.assertRaises(ValueError):
                build_sheets(session, Path(td) / "out")


if __name__ == "__main__":
    unittest.main()
