package jp.showchoo.oneononeai

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.os.SystemClock
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.Rot90Op
import org.tensorflow.lite.task.core.BaseOptions
import org.tensorflow.lite.task.vision.detector.ObjectDetector
import kotlin.math.max
import kotlin.math.min

class ObjectDetectorEngine(context: Context) {
    private val detector: ObjectDetector

    init {
        val baseOptions = BaseOptions.builder()
            .setNumThreads(2)
            .build()
        val options = ObjectDetector.ObjectDetectorOptions.builder()
            .setBaseOptions(baseOptions)
            .setScoreThreshold(0.05f)
            .setMaxResults(20)
            .build()
        detector = ObjectDetector.createFromFileAndOptions(
            context,
            "efficientdet-lite0.tflite",
            options
        )
    }

    data class Result(
        val detections: List<AiDetection>,
        val inferenceMs: Long,
        val imageWidth: Int,
        val imageHeight: Int,
        val ballConfidence: Float,
        val ballSource: String,
        val roiPasses: Int
    )

    private data class PassResult(
        val detections: List<AiDetection>,
        val bestBallScore: Float,
        val bestBallSource: String
    )

    fun detect(
        bitmap: Bitmap,
        rotationDegrees: Int,
        hoopRect: RectF? = null,
        runFullFrame: Boolean = true,
        runPlayerFallback: Boolean = true
    ): Result {
        val start = SystemClock.uptimeMillis()

        val turns = -rotationDegrees / 90
        val processor = ImageProcessor.Builder()
            .add(Rot90Op(turns))
            .build()
        val fullTensor = processor.process(TensorImage.fromBitmap(bitmap))
        val rotatedBitmap = fullTensor.bitmap
        val imageWidth = rotatedBitmap.width
        val imageHeight = rotatedBitmap.height

        val full = if (runFullFrame) {
            runPass(
                bitmap = rotatedBitmap,
                offsetX = 0f,
                offsetY = 0f,
                source = "FULL",
                includePeople = true,
                ballThreshold = 0.10f
            )
        } else {
            PassResult(emptyList(), 0f, "NONE")
        }

        val merged = full.detections.toMutableList()
        var bestBallScore = full.bestBallScore
        var bestBallSource = full.bestBallSource
        var roiPasses = 0

        // Always run the hoop crop when calibration exists. A false positive
        // elsewhere in the full frame must not suppress the most important
        // scoring-region pass.
        var hoopBallFound = false
        if (hoopRect != null) {
            val hoopPx = RectF(
                hoopRect.left * imageWidth,
                hoopRect.top * imageHeight,
                hoopRect.right * imageWidth,
                hoopRect.bottom * imageHeight
            )
            val hoopCropRect = expandedHoopRoi(hoopPx, imageWidth, imageHeight)
            val crop = cropBitmap(rotatedBitmap, hoopCropRect)
            if (crop != null) {
                try {
                    roiPasses += 1
                    val pass = runPass(
                        bitmap = crop,
                        offsetX = hoopCropRect.left,
                        offsetY = hoopCropRect.top,
                        source = "HOOP_ROI",
                        includePeople = false,
                        ballThreshold = 0.05f
                    )
                    if (pass.bestBallScore > bestBallScore) {
                        bestBallScore = pass.bestBallScore
                        bestBallSource = pass.bestBallSource
                    }
                    val balls = pass.detections.filter { it.label == "sports ball" }
                    if (balls.isNotEmpty()) {
                        merged += balls
                        hoopBallFound = true
                    }
                } finally {
                    if (!crop.isRecycled) crop.recycle()
                }
            }
        }

        // Player crops are a fallback for dribbling / release frames.
        val fullBallFound = full.detections.any { it.label == "sports ball" }
        if (runFullFrame && runPlayerFallback && !hoopBallFound && !fullBallFound) {
            val people = full.detections
                .filter { it.label == "person" }
                .sortedByDescending { it.score }
                .take(2)

            for ((index, person) in people.withIndex()) {
                val roi = expandedPlayerRoi(person.box, imageWidth, imageHeight)
                val crop = cropBitmap(rotatedBitmap, roi) ?: continue
                try {
                    roiPasses += 1
                    val source = "PLAYER_ROI_" + (index + 1)
                    val pass = runPass(
                        bitmap = crop,
                        offsetX = roi.left,
                        offsetY = roi.top,
                        source = source,
                        includePeople = false,
                        ballThreshold = 0.05f
                    )
                    if (pass.bestBallScore > bestBallScore) {
                        bestBallScore = pass.bestBallScore
                        bestBallSource = pass.bestBallSource
                    }
                    val balls = pass.detections.filter { it.label == "sports ball" }
                    if (balls.isNotEmpty()) {
                        merged += balls
                        break
                    }
                } finally {
                    if (!crop.isRecycled) crop.recycle()
                }
            }
        }

        return Result(
            detections = merged,
            inferenceMs = SystemClock.uptimeMillis() - start,
            imageWidth = imageWidth,
            imageHeight = imageHeight,
            ballConfidence = bestBallScore,
            ballSource = if (bestBallScore > 0f) bestBallSource else "NONE",
            roiPasses = roiPasses
        )
    }

    private fun runPass(
        bitmap: Bitmap,
        offsetX: Float,
        offsetY: Float,
        source: String,
        includePeople: Boolean,
        ballThreshold: Float
    ): PassResult {
        val raw = detector.detect(TensorImage.fromBitmap(bitmap))
        val accepted = mutableListOf<AiDetection>()
        var bestBallScore = 0f

        raw.forEach { det ->
            val category = det.categories.maxByOrNull { it.score } ?: return@forEach
            val label = category.label.lowercase()
            val score = category.score

            if (label == "sports ball") {
                if (score > bestBallScore) bestBallScore = score
                if (score >= ballThreshold) {
                    accepted += AiDetection(
                        label = label,
                        score = score,
                        box = RectF(
                            det.boundingBox.left + offsetX,
                            det.boundingBox.top + offsetY,
                            det.boundingBox.right + offsetX,
                            det.boundingBox.bottom + offsetY
                        ),
                        source = source
                    )
                }
            } else if (includePeople && label == "person" && score >= 0.30f) {
                accepted += AiDetection(
                    label = label,
                    score = score,
                    box = RectF(
                        det.boundingBox.left + offsetX,
                        det.boundingBox.top + offsetY,
                        det.boundingBox.right + offsetX,
                        det.boundingBox.bottom + offsetY
                    ),
                    source = source
                )
            }
        }

        return PassResult(
            detections = accepted,
            bestBallScore = bestBallScore,
            bestBallSource = source
        )
    }

    private fun expandedHoopRoi(hoop: RectF, width: Int, height: Int): RectF {
        val cx = (hoop.left + hoop.right) / 2f
        val cy = (hoop.top + hoop.bottom) / 2f
        val roiW = max(hoop.width() * 5.0f, width * 0.24f)
        val roiH = max(hoop.height() * 7.0f, height * 0.30f)
        val shiftedCy = cy - roiH * 0.12f

        return clampRoi(
            RectF(
                cx - roiW / 2f,
                shiftedCy - roiH / 2f,
                cx + roiW / 2f,
                shiftedCy + roiH / 2f
            ),
            width,
            height
        )
    }

    private fun expandedPlayerRoi(person: RectF, width: Int, height: Int): RectF {
        val cx = (person.left + person.right) / 2f
        val cy = (person.top + person.bottom) / 2f
        val roiW = max(person.width() * 2.6f, width * 0.20f)
        val roiH = max(person.height() * 1.35f, height * 0.42f)

        return clampRoi(
            RectF(
                cx - roiW / 2f,
                cy - roiH * 0.52f,
                cx + roiW / 2f,
                cy + roiH * 0.48f
            ),
            width,
            height
        )
    }

    private fun clampRoi(rect: RectF, width: Int, height: Int): RectF {
        val left = rect.left.coerceIn(0f, width - 2f)
        val top = rect.top.coerceIn(0f, height - 2f)
        val right = rect.right.coerceIn(left + 2f, width.toFloat())
        val bottom = rect.bottom.coerceIn(top + 2f, height.toFloat())
        return RectF(left, top, right, bottom)
    }

    private fun cropBitmap(bitmap: Bitmap, roi: RectF): Bitmap? {
        val left = max(0, roi.left.toInt())
        val top = max(0, roi.top.toInt())
        val right = min(bitmap.width, roi.right.toInt())
        val bottom = min(bitmap.height, roi.bottom.toInt())
        val w = right - left
        val h = bottom - top
        if (w < 8 || h < 8) return null
        return Bitmap.createBitmap(bitmap, left, top, w, h)
    }
}
