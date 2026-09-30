package jp.showchoo.oneononeai

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.Rot90Op
import org.tensorflow.lite.task.core.BaseOptions
import org.tensorflow.lite.task.vision.detector.ObjectDetector

class ObjectDetectorEngine(context: Context) {
    private val detector: ObjectDetector

    init {
        val baseOptions = BaseOptions.builder()
            .setNumThreads(2)
            .build()
        val options = ObjectDetector.ObjectDetectorOptions.builder()
            .setBaseOptions(baseOptions)
            .setScoreThreshold(0.15f)
            .setMaxResults(12)
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
        val imageHeight: Int
    )

    fun detect(bitmap: Bitmap, rotationDegrees: Int): Result {
        val start = SystemClock.uptimeMillis()
        val turns = -rotationDegrees / 90
        val processor = ImageProcessor.Builder()
            .add(Rot90Op(turns))
            .build()
        val tensorImage = processor.process(TensorImage.fromBitmap(bitmap))
        val raw = detector.detect(tensorImage)
        val detections = raw.mapNotNull { det ->
            val category = det.categories.maxByOrNull { it.score } ?: return@mapNotNull null
            val label = category.label.lowercase()
            val minScore = when (label) {
                "person" -> 0.30f
                "sports ball" -> 0.15f
                else -> return@mapNotNull null
            }
            if (category.score < minScore) return@mapNotNull null
            AiDetection(label, category.score, det.boundingBox)
        }
        return Result(
            detections = detections,
            inferenceMs = SystemClock.uptimeMillis() - start,
            imageWidth = tensorImage.width,
            imageHeight = tensorImage.height
        )
    }
}
