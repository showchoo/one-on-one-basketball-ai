package jp.showchoo.oneononeai

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.RectF
import android.os.SystemClock
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.max
import kotlin.math.min

/**
 * v0.3 basketball-specific detector.
 *
 * Model: E-BARD BODD_yolov8n_0001
 * Classes: basketball / hoop / player / referee
 * License stated by the model repository: CC BY 4.0.
 */
class ObjectDetectorEngine(context: Context) : AutoCloseable {
    companion object {
        private const val MODEL_FILE = "basketball_yolov8n.onnx"
        private const val INPUT_SIZE = 640

        private const val BALL = 0
        private const val HOOP = 1
        private const val PLAYER = 2

        private const val BALL_THRESHOLD = 0.12f
        private const val HOOP_THRESHOLD = 0.30f
        private const val PLAYER_THRESHOLD = 0.25f
        private const val NMS_IOU = 0.45f
    }

    private val environment: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val sessionOptions = OrtSession.SessionOptions().apply {
        setIntraOpNumThreads(2)
        setInterOpNumThreads(1)
        setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
    }
    private val session: OrtSession
    private val inputName: String

    private val inputByteBuffer =
        ByteBuffer.allocateDirect(1 * 3 * INPUT_SIZE * INPUT_SIZE * 4)
            .order(ByteOrder.nativeOrder())
    private val inputBuffer: FloatBuffer = inputByteBuffer.asFloatBuffer()
    private val inputPixels = IntArray(INPUT_SIZE * INPUT_SIZE)

    private val modelBitmap =
        Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888)
    private val modelCanvas = Canvas(modelBitmap)
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)

    init {
        val modelBytes = context.assets.open(MODEL_FILE).use { it.readBytes() }
        session = environment.createSession(modelBytes, sessionOptions)
        inputName = session.inputNames.first()
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

    private data class Candidate(
        val classId: Int,
        val score: Float,
        val box: RectF
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
                INPUT_SIZE.toFloat() / sourceWidth,
                INPUT_SIZE.toFloat() / sourceHeight
            )
            val drawW = sourceWidth * scale
            val drawH = sourceHeight * scale
            val padX = (INPUT_SIZE - drawW) / 2f
            val padY = (INPUT_SIZE - drawH) / 2f

            modelCanvas.drawColor(Color.rgb(114, 114, 114))
            val matrix = Matrix().apply {
                postScale(scale, scale)
                postTranslate(padX, padY)
            }
            modelCanvas.drawBitmap(rotated, matrix, paint)

            modelBitmap.getPixels(
                inputPixels,
                0,
                INPUT_SIZE,
                0,
                0,
                INPUT_SIZE,
                INPUT_SIZE
            )

            inputBuffer.clear()

            // ONNX YOLO input is NCHW RGB float32, normalized to 0..1.
            for (pixel in inputPixels) {
                inputBuffer.put(((pixel shr 16) and 0xFF) / 255f)
            }
            for (pixel in inputPixels) {
                inputBuffer.put(((pixel shr 8) and 0xFF) / 255f)
            }
            for (pixel in inputPixels) {
                inputBuffer.put((pixel and 0xFF) / 255f)
            }
            inputBuffer.flip()

            val output = OnnxTensor.createTensor(
                environment,
                inputBuffer,
                longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong())
            ).use { inputTensor ->
                session.run(mapOf(inputName to inputTensor)).use { result ->
                    @Suppress("UNCHECKED_CAST")
                    (result[0].value as Array<Array<FloatArray>>)
                }
            }

            // YOLOv8 detection output: [1, 4 + numClasses, 8400].
            val rows = output[0]
            require(rows.size >= 7) {
                "Unexpected YOLO output channels: " + rows.size
            }
            val anchors = rows[0].size
            val classCount = min(4, rows.size - 4)

            val raw = ArrayList<Candidate>(128)

            for (i in 0 until anchors) {
                val cx = rows[0][i]
                val cy = rows[1][i]
                val w = rows[2][i]
                val h = rows[3][i]

                var bestClass = -1
                var bestScore = 0f
                for (classIndex in 0 until classCount) {
                    val score = rows[4 + classIndex][i]
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

                val left =
                    ((cx - w / 2f - padX) / scale).coerceIn(0f, sourceWidth.toFloat())
                val top =
                    ((cy - h / 2f - padY) / scale).coerceIn(0f, sourceHeight.toFloat())
                val right =
                    ((cx + w / 2f - padX) / scale).coerceIn(0f, sourceWidth.toFloat())
                val bottom =
                    ((cy + h / 2f - padY) / scale).coerceIn(0f, sourceHeight.toFloat())

                if (right - left < 2f || bottom - top < 2f) continue

                raw += Candidate(
                    classId = bestClass,
                    score = bestScore,
                    box = RectF(left, top, right, bottom)
                )
            }

            val selected = mutableListOf<Candidate>()
            selected += nms(
                raw.filter { it.classId == BALL },
                NMS_IOU,
                6
            )
            selected += nms(
                raw.filter { it.classId == HOOP },
                NMS_IOU,
                3
            )
            selected += nms(
                raw.filter { it.classId == PLAYER },
                NMS_IOU,
                6
            )

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
            if (rotated !== bitmap && !rotated.isRecycled) {
                rotated.recycle()
            }
        }
    }

    private fun nms(
        candidates: List<Candidate>,
        iouThreshold: Float,
        maxResults: Int
    ): List<Candidate> {
        if (candidates.isEmpty()) return emptyList()

        val sorted = candidates
            .sortedByDescending { it.score }
            .toMutableList()
        val kept = mutableListOf<Candidate>()

        while (sorted.isNotEmpty() && kept.size < maxResults) {
            val best = sorted.removeAt(0)
            kept += best
            sorted.removeAll { other ->
                iou(best.box, other.box) > iouThreshold
            }
        }

        return kept
    }

    private fun iou(a: RectF, b: RectF): Float {
        val left = max(a.left, b.left)
        val top = max(a.top, b.top)
        val right = min(a.right, b.right)
        val bottom = min(a.bottom, b.bottom)

        val intersection =
            max(0f, right - left) * max(0f, bottom - top)
        val union =
            a.width() * a.height() +
                b.width() * b.height() -
                intersection

        return if (union <= 0f) 0f else intersection / union
    }

    private fun rotate(bitmap: Bitmap, degrees: Int): Bitmap {
        val normalized = ((degrees % 360) + 360) % 360
        if (normalized == 0) return bitmap

        val matrix = Matrix().apply {
            postRotate(normalized.toFloat())
        }

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

    override fun close() {
        session.close()
        sessionOptions.close()
        if (!modelBitmap.isRecycled) modelBitmap.recycle()
    }
}
