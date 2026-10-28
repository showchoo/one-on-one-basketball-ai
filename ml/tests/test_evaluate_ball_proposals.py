"""Synthetic detector baseline metrics; no unverified model box becomes ground truth."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from ml.evaluate_ball_proposals import (
    box_iou_xywh, evaluate_session, evaluate_root, summarize
)


def make_review_session(root: Path, folder_name="capture_1"):
    session = root / "sessions" / folder_name
    (session / "images").mkdir(parents=True)
    (session / "labels").mkdir()
    proposal_items = {}
    statuses = {}
    for i in range(1, 5):
        stem = f"frame_{i}"
        path = session / "images" / (stem + ".jpg")
        Image.new("RGB", (60, 48), (20*i, 45, 67)).save(path)
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        statuses[stem] = "positive" if i <= 2 else "negative"
        (session / "labels" / (stem + ".txt")).write_text(
            "0 0.5 0.5 0.1 0.1\n" if i <= 2 else "",
            encoding="utf-8"
        )
        if i == 1:
            entry = {"box": [.5,.5,.1,.1], "confidence": .9}
        elif i == 2:
            entry = {"box": [.2,.2,.1,.1], "confidence": .6}
        elif i == 3:
            entry = {"box": [.8,.8,.1,.1], "confidence": .7}
        else:
            entry = None
        if entry is not None:
            entry["image_sha256"] = checksum
        proposal_items[stem] = entry
    (session / "review_status.json").write_text(
        json.dumps(statuses), encoding="utf-8"
    )
    (session / "proposals.json").write_text(
        json.dumps({"schema_version": 1,
                    "model": "fake-weights.pt",
                    "model_weights_sha256": "modelhash",
                    "proposals": proposal_items}), encoding="utf-8"
    )
    return session


class EvaluationTests(unittest.TestCase):
    def test_undefined_metrics_are_not_misreported_as_zero(self):
        counts = dict(tp=0, fp=0, fn=0, tn=3, positive_total=0,
                      negative_total=3, negative_fp=0)
        metrics = summarize(counts)
        self.assertIsNone(metrics["precision"])
        self.assertIsNone(metrics["recall"])
        self.assertEqual(metrics["no_ball_false_positive_rate"], 0.0)

    def test_iou_geometry(self):
        self.assertEqual(box_iou_xywh([.5,.5,.1,.1], [.5,.5,.1,.1]), 1.0)
        self.assertEqual(box_iou_xywh([.5,.5,.1,.1], [.2,.2,.1,.1]), 0.0)
        self.assertAlmostEqual(
            box_iou_xywh([.5,.5,.2,.2], [.55,.5,.2,.2]), .6
        )

    def test_wrong_positive_location_counts_as_false_positive_and_miss(self):
        with tempfile.TemporaryDirectory() as td:
            session = make_review_session(Path(td))
            result = evaluate_session(session, thresholds=(.5, .8))
            at_half = result["thresholds"]["0.5"]
            self.assertEqual([at_half[k] for k in ("tp","fp","fn","tn")],
                             [1, 2, 1, 1])
            self.assertEqual(at_half["precision"], .3333)
            self.assertEqual(at_half["recall"], .5)
            self.assertEqual(at_half["no_ball_false_positive_rate"], .5)
            at_high = result["thresholds"]["0.8"]
            self.assertEqual([at_high[k] for k in ("tp","fp","fn","tn")],
                             [1, 0, 1, 2])
            self.assertEqual(at_high["precision"], 1.0)

    def test_root_aggregate_and_missing_review(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            make_review_session(root, "capture_1")
            missing = root / "sessions" / "capture_2"
            missing.mkdir()
            result = evaluate_root(root, thresholds=(.5,))
            self.assertEqual(result["aggregate"]["0.5"]["reviewed"], 4)
            self.assertEqual(result["missing_sessions"], ["capture_2"])

    def test_stale_frame_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            session = make_review_session(Path(td))
            (session / "images" / "frame_1.jpg").write_bytes(b"stale")
            with self.assertRaisesRegex(ValueError, "different image"):
                evaluate_session(session)

    def test_unreviewed_proposals_are_not_truth(self):
        with tempfile.TemporaryDirectory() as td:
            session = make_review_session(Path(td))
            (session / "review_status.json").write_text(
                '{"frame_1":"unclear"}', encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "No confirmed"):
                evaluate_session(session)


if __name__ == "__main__":
    unittest.main()
