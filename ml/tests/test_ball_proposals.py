"""Synthetic tests: suggestions stay private and never become labels automatically."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image
from ml.propose_ball_boxes import propose, get_ball_class, read_ball_candidates
from ml.label_ball import verified_suggestion


class Scalar:
    def __init__(self, value):
        self.value = value
    def item(self):
        return self.value


class Vector:
    def __init__(self, values):
        self.values = values
    def tolist(self):
        return self.values


class Box:
    def __init__(self, kind, confidence, xywh):
        self.cls = [Scalar(kind)]
        self.conf = [Scalar(confidence)]
        self.xywhn = [Vector(xywh)]


class Predicted:
    def __init__(self):
        self.boxes = [
            Box(0, 0.67, [.4, .55, .08, .08]),
            Box(1, 0.99, [.5, .5, .2, .2])
        ]


class Model:
    names = {0: "basketball", 1: "hoop"}
    def __init__(self, weights):
        self.weights = weights
    def predict(self, **kwargs):
        return [Predicted()]


class SuggestionsTests(unittest.TestCase):
    def make_session(self, temp):
        folder = Path(temp) / "capture_1"
        folder.mkdir()
        (folder / "images").mkdir()
        photo = folder / "images" / "frame_1.jpg"
        Image.new("RGB", (60, 44), (33, 77, 99)).save(photo)
        for filename in ("index.csv", "review_queue.csv"):
            with (folder / filename).open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["filename"])
                writer.writerow(["images/frame_1.jpg"])
        return folder, photo

    def test_proposals_are_unverified_and_checksum_bound(self):
        with tempfile.TemporaryDirectory() as td:
            session, photo = self.make_session(td)
            weight = Path(td) / "model.pt"
            weight.write_bytes(b"synthetic model")
            result = propose(session, weight, model_factory=Model)
            self.assertFalse(result["verified"])
            self.assertEqual(result["positive_labels_created"], 0)
            self.assertEqual(result["frames_inspected"], 1)
            self.assertFalse((session / "labels").exists())
            self.assertFalse((session / "review_status.json").exists())
            data = json.loads((session / "proposals.json").read_text())
            candidate = data["proposals"]["frame_1"]
            self.assertAlmostEqual(candidate["confidence"], .67)
            self.assertEqual(candidate["box"], [.4, .55, .08, .08])
            self.assertEqual(verified_suggestion(candidate, photo), candidate)
            with self.assertRaises(FileExistsError):
                propose(session, weight, model_factory=Model)
            photo.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "different image"):
                verified_suggestion(candidate, photo)

    def test_wrong_model_class_refused(self):
        class WrongModel:
            names = {0: "person"}
        with self.assertRaisesRegex(ValueError, "basketball class"):
            get_ball_class(WrongModel())

    def test_rejects_nonfinite_boxes(self):
        fake = Predicted()
        fake.boxes.insert(0, Box(0, .99, [float("nan"), .5, .2, .2]))
        self.assertEqual(len(read_ball_candidates(fake, 0)), 1)

    def test_rejects_invalid_review_hint(self):
        with tempfile.TemporaryDirectory() as td:
            _, photo = self.make_session(td)
            with self.assertRaisesRegex(ValueError, "Invalid unverified"):
                verified_suggestion({
                    "box": [.5, .5, float("nan"), .1],
                    "confidence": .9, "image_sha256": "anything"
                }, photo)


if __name__ == "__main__":
    unittest.main()
