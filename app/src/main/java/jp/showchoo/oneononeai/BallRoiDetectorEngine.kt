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
import android.graphics.Rect
import android.graphics.RectF
import android.os.SystemClock
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.max
import kotlin.math.min

/**
 * Fast basketball-only detector for a cropped region.\n * v0.6.2 uses the MIT-licensed Stardust87 basketball-specific YOLOv5s weights.
 *
 * The 320x320 model sees only a small part of the camera image, so a ball that
 * is only a few pixels wide in the full frame becomes much larger in model
 * space. This is the high-frequency detector used by v0.5.
 */
class BallRoiDetectorEngine(context: Context) : AutoCloseable {
    companion object {
        private const val MODEL_FILE = "basketball_ball_yolov5s_320.onnx"
        private const val INPUT_SIZE = 320
        private const val BALL_CLASS = 0
        private const val BALL_THRESHOLD = 0.08f
        private const val NMS_IOU = 0.40f
    }

    data class Result(
        val detections: List<AiDetection>,
        val inferenceMs: Long,
        val roi: RectF,
        val imageWidth: Int,
        val imageHeight: Int
    )

    private data class Candidate(
        val score: Float,
        val box: RectF
    )

    private val environment = OrtEnvironment.getEnvironment()
    private val sessionOptions = OrtSession.SessionOptions().apply {
        setIntraOpNumThreads(1)
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
    private val canvas = Canvas(modelBitmap)
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)

    init {
        val modelBytes = context.assets.open(MODEL_FILE).use { it.readBytes() }
        session = environment.createSession(modelBytes, sessionOptions)
        inputName = session.inputNames.first()
    }

    fun detect(
        bitmap: Bitmap,
        rotationDegrees: Int,
        roiNorm: RectF
    ): Result {
        val start = SystemClock.uptimeMillis()
        val rotated = rotate(bitmap, rotationDegrees)

        try {
            val imageWidth = rotated.width
            val imageHeight = rotated.height
            val roi = clampRoi(roiNorm)

            val src = Rect(
                (roi.left * imageWidth).toInt().coerceIn(0, imageWidth - 2),
                (roi.top * imageHeight).toInt().coerceIn(0, imageHeight - 2),
                (roi.right * imageWidth).toInt().coerceIn(2, imageWidth),
                (roi.bottom * imageHeight).toInt().coerceIn(2, imageHeight)
            )

            val cropW = (src.right - src.left).coerceAtLeast(2)
            val cropH = (src.bottom - src.top).coerceAtLeast(2)
            val scale = min(
                INPUT_SIZE.toFloat() / cropW,
                INPUT_SIZE.toFloat() / cropH
            )
            val drawW = cropW * scale
            val drawH = cropH * scale
            val padX = (INPUT_SIZE - drawW) / 2f
            val padY = (INPUT_SIZE - drawH) / 2f

            canvas.drawColor(Color.rgb(114, 114, 114))
            val dst = RectF(
                padX,
                padY,
                padX + drawW,
                padY + drawH
            )
            canvas.drawBitmap(rotated, src, dst, paint)

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
            ).use { tensor ->
                session.run(mapOf(inputName to tensor)).use { result ->
                    @Suppress("UNCHECKED_CAST")
                    (result[0].value as Array<Array<FloatArray>>)
                }
            }

            val rows = output[0]
            if (rows.isEmpty()) {
                return Result(
                    emptyList(),
                    SystemClock.uptimeMillis() - start,
                    roi,
                    imageWidth,
                    imageHeight
                )
            }

            // Dedicated Stardust87 YOLOv5 export:
            // [1, num_predictions, 5 + num_classes].
            // Each prediction is cx, cy, w, h, objectness, class probability.
            val raw = ArrayList<Candidate>(48)

            for (row in rows) {
                if (row.size < 6) continue
                val objectness = row[4]
                val classProbability = row[5]
                val score = objectness * classProbability
                if (score < BALL_THRESHOLD) continue

                val cx = row[0]
                val cy = row[1]
                val w = row[2]
                val h = row[3]

                val cropLeft = (cx - w / 2f - padX) / scale
                val cropTop = (cy - h / 2f - padY) / scale
                val cropRight = (cx + w / 2f - padX) / scale
                val cropBottom = (cy + h / 2f - padY) / scale

                val leftPx = (src.left + cropLeft).coerceIn(0f, imageWidth.toFloat())
                val topPx = (src.top + cropTop).coerceIn(0f, imageHeight.toFloat())
                val rightPx = (src.left + cropRight).coerceIn(0f, imageWidth.toFloat())
                val bottomPx = (src.top + cropBottom).coerceIn(0f, imageHeight.toFloat())

                if (rightPx - leftPx < 2f || bottomPx - topPx < 2f) continue

                raw += Candidate(
                    score = score,
                    box = RectF(
                        leftPx / imageWidth,
                        topPx / imageHeight,
                        rightPx / imageWidth,
                        bottomPx / imageHeight
                    )
                )
            }

            val selected = nms(raw, NMS_IOU, 5)
            val detections = selected.map {
                AiDetection(
                    label = "sports ball",
                    score = it.score,
                    box = RectF(it.box),
                    source = "BALL_DEDICATED_YOLOV5"
                )
            }

            return Result(
                detections = detections,
                inferenceMs = SystemClock.uptimeMillis() - start,
                roi = roi,
                imageWidth = imageWidth,
                imageHeight = imageHeight
            )
        } finally {
            if (rotated !== bitmap && !rotated.isRecycled) rotated.recycle()
        }
    }

    private fun nms(
        candidates: List<Candidate>,
        threshold: Float,
        maxResults: Int
    ): List<Candidate> {
        val sorted = candidates.sortedByDescending { it.score }.toMutableList()
        val kept = mutableListOf<Candidate>()
        while (sorted.isNotEmpty() && kept.size < maxResults) {
            val best = sorted.removeAt(0)
            kept += best
            sorted.removeAll { iou(best.box, it.box) > threshold }
        }
        return kept
    }

    private fun iou(a: RectF, b: RectF): Float {
        val left = max(a.left, b.left)
        val top = max(a.top, b.top)
        val right = min(a.right, b.right)
        val bottom = min(a.bottom, b.bottom)
        val inter = max(0f, right - left) * max(0f, bottom - top)
        val union = a.width() * a.height() + b.width() * b.height() - inter
        return if (union <= 0f) 0f else inter / union
    }

    private fun clampRoi(input: RectF): RectF {
        val minSize = 0.12f
        var left = input.left.coerceIn(0f, 1f)
        var top = input.top.coerceIn(0f, 1f)
        var right = input.right.coerceIn(0f, 1f)
        var bottom = input.bottom.coerceIn(0f, 1f)

        if (right - left < minSize) {
            val cx = (left + right) / 2f
            left = (cx - minSize / 2f).coerceIn(0f, 1f - minSize)
            right = left + minSize
        }
        if (bottom - top < minSize) {
            val cy = (top + bottom) / 2f
            top = (cy - minSize / 2f).coerceIn(0f, 1f - minSize)
            bottom = top + minSize
        }

        return RectF(left, top, right, bottom)
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

    override fun close() {
        session.close()
        sessionOptions.close()
        if (!modelBitmap.isRecycled) modelBitmap.recycle()
    }
}
