# Basketball-only detector development (ML stage 1)

This folder prepares **human-verified** basketball bounding boxes for a future
fast Android ball detector. It does **not** contain a trained model yet.

## Evidence from the first S20+ sample (2026-10-15)

- 459 valid 1280x720 JPEG frames in the collected ZIP (~240 seconds; ~523ms median spacing).
- The phone was pointed at a monitor replaying a 1on1 video, **not a real half-court**.
- The app's WASB detection count was 0/541 with a 365ms median inference time.
- The APK-bundled E-BARD YOLOv8n class-0 basketball detector was also tested
  offline on 92 sampled replay frames at its native 640x640 input. Its maximum
  basketball-class score was >=0.08 on 8/92 *raw* frames; 9/92 when cropping
  the visible video to x=150..1065, y=2..400 before inference.
- These are **unverified model proposals, not correct detections or recall**.
  Some boxes highlight unrelated objects. Crop does NOT reliably resolve the
  failure. Promoting such proposals automatically to true labels is unsafe.
- A direct court recording at the *final tripod location* is therefore
  required to assess product-level generalization.
- The source imagery **must remain private** and must never be committed to
  the public repository.

## Pipeline

### 1. Ingest safely, classifying source domain

From the project root:

```powershell
python -m pip install Pillow opencv-python
python ml/ingest_capture.py "D:\path\session.zip" --output ml_data --source-mode screen_replay
```

For footage captured **directly in the basketball court**, use
`--source-mode direct_court`. Do not mislabel monitor playback as direct-court.

The script validates JPEGs, timestamps, image dimensions, manifest references
and ZIP paths. It produces:

```text
ml_data/
  sessions/
    capture_<start_utc_ms>/
      images/
      index.csv
      session.json
```

A replay-only diagnostic crop is supported via
`--crop 150,2,1065,400`; those coordinates fit **only the first sample**.
Cropped monitor frames must remain in the `screen_replay` domain.

### 2. Confirm labels, including real negative examples

```powershell
python ml/label_ball.py ml_data/sessions/capture_<start_utc_ms>
```

OpenCV keyboard controls: `b` draw one box around the actual ball, `n` mark
**no visible ball**, `u` mark uncertain/occluded, `s` leave unreviewed,
`q` exit. Reopen to resume. Use `--revisit` to correct reviewed frames.

This creates YOLO-format one-class labels in `labels/*.txt`
plus `review_status.json`. An empty label file represents an explicit,
verified **no-ball** image; unknown/uncertain frames are excluded from training.

If using AI to suggest boxes in the future, have them reviewed before training.

### 3. Create leak-free training data

Only after **at least three separate real-court sessions** have been collected
and labeled (20 confirmed ball boxes is a minimal tool gate, NOT sufficient
for production quality):

```powershell
python ml/export_verified_coco.py ml_data --output ml_data/export --min-positives 20
```

Generates a one-class COCO layout in `ml_data/export/COCO` with a
**session-disjoint** train/val/test split and excludes replay sessions entirely.
This prevents nearly identical adjacent frames from leaking into validation.

For product evaluation, collect additional diverse days/venues/lighting,
more than 20 positives per split, near/far dribbling, fast shots, passes,
occlusions and negative frames. Detection must be evaluated with human labels,
precision/recall and on-device latency; candidate scores alone are inadequate.

### 4. Training architecture prepared (not yet trained)

We selected **YOLOX Nano** for an initial 1-class experiment. The official
YOLOX implementation is Apache-2.0; independently verify distribution rights
for each pretrained weight file and training dataset before commercial release.

Official project: https://github.com/Megvii-BaseDetection/YOLOX

After installing YOLOX's training dependencies in a GPU-capable PC or cloud
environment and preparing the verified COCO data:

```bash
export BALL_COCO_DIR="$PWD/ml_data/export/COCO"
python /path/to/YOLOX/tools/train.py \
  -f "$PWD/ml/ball_yolox_nano_exp.py" -d 1 -b 8
```

Adjust paths and batch size for the machine. CPU training is not intended for
use on an Android phone. A trained YOLOX model cannot simply replace the
existing YOLOv8 ONNX asset: **output decoding/NMS and model integration differ**.
The Android inference path and on-device metrics are a separate follow-up.

### 5. Verify the data pipeline

```bash
python -m unittest discover -s ml/tests -v
```

No real user JPEGs or full ZIP recordings should be uploaded to public GitHub.

## Next gate

Acquire direct-court S20+ footage with consent, with the camera mounted at the
intended in-game position (no monitor in view). After labeled image sets exist,
train/evaluate a small model. Compare it against WASB in *real court* video.
