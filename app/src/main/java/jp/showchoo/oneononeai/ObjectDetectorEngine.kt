package jp.showchoo.oneononeai

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.RectF
import android.os.SystemClock
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.max
import kotlin.math.min

/**
 * v0.3 detector.
 *
 * Uses a basketball-specific YOLOv8n model trained for:
 * basketball / hoop / player / referee.
 *
 * Model: Gabriele Giudici, E-BARD detection models (CC BY 4.0).
 */
class ObjectDetectorEngine(context: Context) : AutoCloseable {
    companion object {
        private const val MODEL_FILE = "basketball_yolov8n.tflite"
        private const val BALL = 0
        private const val HOOP = 1
        private const val PLAYER = 2
        private const val REFEREE = 3

        private const val BALL_THRESHOLD = 0.12f
        private const val HOOP_THRESHOLD = 0.30f
        private const val PLAYER_THRESHOLD = 0.25f
        private const val NMS_IOU = 0.45f
    }

    private val interpreter: Interpreter
    private val inputWidth: Int
    private val inputHeight: Int
    private val channels: Int
    private val anchors: Int
    private val channelFirstOutput: Boolean

    private val inputBuffer: ByteBuffer
    private val inputPixels: IntArray
    private val modelBitmap: Bitmap
    private val modelCanvas: Canvas
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)

    private val outputCF: Array<Array<FloatArray>>?
    private val outputCL: Array<Array<FloatArray>>?

    init {
        val options = Interpreter.Options()
            .setNumThreads(2)
            .setUseXNNPACK(true)

        interpreter = Interpreter(loadModelFile(context, MODEL_FILE), options)

        val inputShape = interpreter.getInputTensor(0).shape()
        require(inputShape.size == 4) { "Unexpected YOLO input shape" }
        inputHeight = inputShape[1]
        inputWidth = inputShape[2]
        require(inputShape[3] == 3) { "YOLO input must be NHWC RGB" }
        require(interpreter.getInputTensor(0).dataType() == DataType.FLOAT32) {
            "YOLO input must be FLOAT32"
        }

        val outputShape = interpreter.getOutputTensor(0).shape()
        require(outputShape.size == 3 && outputShape[0] == 1) {
            "Unexpected YOLO output shape: " + outputShape.joinToString("x")
        }

        channelFirstOutput = outputShape[1] <= 32
        if (channelFirstOutput) {
            channels = outputShape[1]
            anchors = outputShape[2]
            outputCF = Array(1) { Array(channels) { FloatArray(anchors) } }
            outputCL = null
        } else {
            anchors = outputShape[1]
            channels = outputShape[2]
            outputCF = null
            outputCL = Array(1) { Array(anchors) { FloatArray(channels) } }
        }

        require(channels >= 8) {
            "YOLO output has too few channels: $channels"
        }

        inputBuffer = ByteBuffer
            .allocateDirect(inputWidth * inputHeight * 3 * 4)
            .order(ByteOrder.nativeOrder())

        inputPixels = IntArray(inputWidth * inputHeight)
        modelBitmap = Bitmap.createBitmap(inputWidth, inputHeight, Bitmap.Config.ARGB_8888)
        modelCanvas = Canvas(modelBitmap)
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

    fun detect(
        bitmap: Bitmap,
        rotationDegrees: Int,
        hoopRect: RectF? = null,
        runFullFrame: Boolean = true,
        runPlayerFallback: Boolean = true
    ): Result {
        val start = SystemClock.uptimeMillis()

        val rotated = rotate(bitmap, rotationDegrees)
        try {
            val sourceWidth = rotated.width
            val sourceHeight = rotated.height

            val scale = min(
                inputWidth.toFloat() / sourceWidth,
                inputHeight.toFloat() / sourceHeight
            )
            val drawW = sourceWidth * scale
            val drawH = sourceHeight * scale
            val padX = (inputWidth - drawW) / 2f
            val padY = (inputHeight - drawH) / 2f

            modelCanvas.drawColor(Color.rgb(114, 114, 114))
            val matrix = Matrix().apply {
                postScale(scale, scale)
                postTranslate(padX, padY)
            }
            modelCanvas.drawBitmap(rotated, matrix, paint)

            modelBitmap.getPixels(
                inputPixels,
                0,
                inputWidth,
                0,
                0,
                inputWidth,
                inputHeight
            )

            inputBuffer.rewind()
            for (pixel in inputPixels) {
                inputBuffer.putFloat(((pixel shr 16) and 0xFF) / 255f)
                inputBuffer.putFloat(((pixel shr 8) and 0xFF) / 255f)
                inputBuffer.putFloat((pixel and 0xFF) / 255f)
            }
            inputBuffer.rewind()

            if (channelFirstOutput) {
                interpreter.run(inputBuffer, outputCF)
            } else {
                interpreter.run(inputBuffer, outputCL)
            }

            val raw = ArrayList<Candidate>(96)
            for (i in 0 until anchors) {
                val cx = value(i, 0)
                val cy = value(i, 1)
                val w = value(i, 2)
                val h = value(i, 3)

                var bestClass = -1
                var bestScore = 0f
                val classCount = min(4, channels - 4)
                for (classIndex in 0 until classCount) {
                    val score = value(i, 4 + classIndex)
                    if (score > bestScore) {
                        bestScore = score
                        bestClass = classIndex
                    }
                }

                val threshold = when (bestClass) {
                    BALL -> BALL_THRESHOLD
                    HOOP -> HOOP_THRESHOLD
                    PLAYER -> PLAYER_THRESHOLD
                    else -> 1f
                }
                if (bestScore < threshold) continue

                val left = ((cx - w / 2f - padX) / scale).coerceIn(0f, sourceWidth.toFloat())
                val top = ((cy - h / 2f - padY) / scale).coerceIn(0f, sourceHeight.toFloat())
                val right = ((cx + w / 2f - padX) / scale).coerceIn(0f, sourceWidth.toFloat())
                val bottom = ((cy + h / 2f - padY) / scale).coerceIn(0f, sourceHeight.toFloat())
                if (right - left < 2f || bottom - top < 2f) continue

                raw += Candidate(
                    classId = bestClass,
                    score = bestScore,
                    box = RectF(left, top, right, bottom)
                )
            }

            val selected = mutableListOf<Candidate>()
            for (classId in listOf(BALL, HOOP, PLAYER)) {
                val maxResults = when (classId) {
                    BALL -> 6
                    HOOP -> 3
                    else -> 6
                }
                selected += nms(
                    raw.filter { it.classId == classId },
                    NMS_IOU,
                    maxResults
                )
            }

            val detections = selected.mapNotNull { candidate ->
                val label = when (candidate.classId) {
                    BALL -> "sports ball"
                    HOOP -> "hoop"
                    PLAYER -> "person"
                    else -> return@mapNotNull null
                }
                AiDetection(
                    label = label,
                    score = candidate.score,
                    box = candidate.box,
                    source = "BASKET_YOLO"
                )
            }

            val bestBall = selected
                .filter { it.classId == BALL }
                .maxByOrNull { it.score }

            return Result(
                detections = detections,
                inferenceMs = SystemClock.uptimeMillis() - start,
                imageWidth = sourceWidth,
                imageHeight = sourceHeight,
                ballConfidence = bestBall?.score ?: 0f,
                ballSource = if (bestBall != null) "BASKET_YOLO" else "NONE",
                roiPasses = 0
            )
        } finally {
            if (rotated !== bitmap && !rotated.isRecycled) rotated.recycle()
        }
    }

    private data class Candidate(
        val classId: Int,
        val score: Float,
        val box: RectF
    )

    private fun value(anchor: Int, channel: Int): Float =
        if (channelFirstOutput) {
            outputCF!![0][channel][anchor]
        } else {
            outputCL!![0][anchor][channel]
        }

    private fun nms(
        candidates: List<Candidate>,
        iouThreshold: Float,
        maxResults: Int
    ): List<Candidate> {
        if (candidates.isEmpty()) return emptyList()

        val sorted = candidates.sortedByDescending { it.score }.toMutableList()
        val kept = mutableListOf<Candidate>()

        while (sorted.isNotEmpty() && kept.size < maxResults) {
            val best = sorted.removeAt(0)
            kept += best
            sorted.removeAll { other -> iou(best.box, other.box) > iouThreshold }
        }
        return kept
    }

    private fun iou(a: RectF, b: RectF): Float {
        val left = max(a.left, b.left)
        val top = max(a.top, b.top)
        val right = min(a.right, b.right)
        val bottom = min(a.bottom, b.bottom)
        val intersection = max(0f, right - left) * max(0f, bottom - top)
        val union = a.width() * a.height() + b.width() * b.height() - intersection
        return if (union <= 0f) 0f else intersection / union
    }

    private fun rotate(bitmap: Bitmap, degrees: Int): Bitmap {
        val normalized = ((degrees % 360) + 360) % 360
        if (normalized == 0) return bitmap
        val matrix = Matrix().apply { postRotate(normalized.toFloat()) }
        return Bitmap.createBitmap(
            bitmap,
            0,
            0,
            bitmap.width,
            bitmap.height,
            matrix,
            true
        )
    }

    private fun loadModelFile(context: Context, fileName: String): MappedByteBuffer {
        val descriptor = context.assets.openFd(fileName)
        FileInputStream(descriptor.fileDescriptor).channel.use { channel ->
            return channel.map(
                FileChannel.MapMode.READ_ONLY,
                descriptor.startOffset,
                descriptor.declaredLength
            )
        }
    }

    override fun close() {
        interpreter.close()
        if (!modelBitmap.isRecycled) modelBitmap.recycle()
    }
}
