# v0.7.5: Training-frame collection (Android)

This feature captures **unlabeled** full camera frames for a purpose-built basketball detector.
It does not train a model, and it does not claim the ball is already recognized.

## On-device instructions

v0.7.6: Press **START** and confirm **両方開始** to start BOTH game tracking
and training-frame capture at once. Set hoop / 3P calibration first for
scoring evaluation. The image capture is opt-in and confirmation is required.
To finish, open **操作を開く**, tap **■ 収集中 N枚**, then **学習ZIP共有**.
**RESET** also stops capture without deleting its images. Share the ZIP
before starting another session. **画像のみ収集** remains available if you
want pictures without starting a game.


1. Install v0.7.5 on the Galaxy S20+ and give camera permission.
2. Position the phone on the fixed tripod so the full half-court is visible.
3. Open **操作を開く** and tap **● 学習収集開始**. Read the privacy notice and tap **収集開始**.
4. Play normal 1on1, including shooting, passing, dribbling, and moments with no ball visible.
5. Open controls and tap **■ 収集中 N枚** to stop.
6. Tap **学習ZIP共有** and save the ZIP using Google Drive (or another Android share target).
7. Keep the debug log: the separate **ログ共有** button still exports its CSV.

## What gets saved

- One 1280x720 camera frame approximately every 500 ms (unless camera load skips frames).
- Original image resolution, JPEG quality 88, rotated into the upright orientation.
- \`images/frame_*.jpg\`, \`frames.csv\`, and \`README.txt\` packaged in the ZIP.
- \`frames.csv\`: filename, UTC epoch timestamp (milliseconds), elapsed uptime (milliseconds),
  saved width/height, applied rotation, device.
- At most 1,200 frames or ~250 MiB per session. The camera itself continues working after
  collection automatically stops.
- A new capture creates a new session. ZIP share exports **the most recent session**.
- Files stay on-device inside app-specific storage until the user explicitly shares them.
  The OS may delete app-specific files if the app is uninstalled or its data is cleared.
- No location or microphone data is collected by this feature.

## Privacy / quality

Obtain permission from anyone recognizable in the recordings. Avoid capturing bystanders.
Export only to a destination you trust. Full frames can contain faces, clothing, and
other personally identifying visual information. Do not publish raw frames openly.

Annotations such as basketball bounding boxes are **not present yet**. Those must be labeled
before the images can be used to train and evaluate a supervised object detector.
For evaluation, split by capture session rather than randomly splitting near-identical frames.

## Notes for follow-up training

Use session-separated train/validation/test sets and include small/distant balls,
motion blur, players occluding the ball, strong sunlight, shadows, and no-ball negatives.
The existing WASB model is unchanged in v0.7.5; only instrumentation is new.
