# Development Notes

## Goal

One Android phone + tripod + Bluetooth speaker. Half-court 1v1. First to 10. Inside arc = 1, outside arc = 2.

## v0.1 event pipeline

CameraX RGBA frame
→ EfficientDet-Lite0
→ filter `person` / `sports ball`
→ normalize coordinates
→ two-player nearest-neighbor tracking
→ possessor estimate (ball inside expanded player box)
→ release candidate (near → not-near transition)
→ shot value from calibrated 3P polyline
→ hoop crossing state machine
→ GameEngine score
→ TextToSpeech

## Performance knobs for arrows We2

- `MainActivity.lastInferenceAt`: 180ms interval. Raise to 250–300ms if hot/slow.
- `ObjectDetectorEngine`: 2 CPU threads.
- `setTargetResolution(1280x720)`: can be reduced to 960x540 if frame conversion is expensive.
- EfficientDet-Lite0 is only the bootstrap model. A dedicated basketball detector should be much smaller and more accurate for this task.
