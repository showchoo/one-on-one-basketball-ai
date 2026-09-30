package jp.showchoo.oneononeai

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.providers.NNAPIFlags
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
import java.util.ArrayDeque
import java.util.EnumSet
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/**
 * Three-frame basketball tracker based on the lightweight WASB HRNet weights
 * published with NTT Communications' WASB-SBDT repository.
 *
 * Input: 3 RGB frames concatenated channel-first => [1, 9, 288, 512]
 * Output: 3 heatmap logits => [1, 3, 288, 512]
 *
 * The latest output heatmap is converted directly to a ball center. Unlike the
 * legacy YOLO path, temporal evidence is learned inside the network itself.
 */
class TemporalBallDetectorEngine(context: Context) : AutoCloseable {
    companion object {
        private const val MODEL_FILE = "wasb_basketball_3f.onnx"
        private const val INPUT_W = 512
        private const val INPUT_H = 288
        private const val FRAMES = 3
        private const val CHANNELS_PER_FRAME = 3
        private const val OUTPUT_FRAME_INDEX = 2
        private const val LOGIT_THRESHOLD = 0.0f // sigmoid(0) = 0.5
        private const val MIN_FRAME_INTERVAL_MS = 90L

        private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)
    }

    data class Result(
        val detection: AiDetection?,
        val inferenceMs: Long,
        val maxHeat: Float,
        val rawPeakProbability: Float,
        val rawPeakX: Float,
        val rawPeakY: Float,
        val blobPixels: Int,
        val backend: String,
        val captureTimeMs: Long,
        val imageWidth: Int,
        val imageHeight: Int
    )

    private data class TemporalFrame(
        val chw: FloatArray,
        val captureTimeMs: Long,
        val imageWidth: Int,
        val imageHeight: Int,
        val modelScale: Float,
        val modelOffsetX: Float,
        val modelOffsetY: Float
    )

    private data class PreprocessedFrame(
        val chw: FloatArray,
        val scale: Float,
        val offsetX: Float,
        val offsetY: Float
    )

    private val environment = OrtEnvironment.getEnvironment()
    private val sessionOptions: OrtSession.SessionOptions
    private val session: OrtSession
    private val inputName: String
    private val backendName: String

    private val frameBuffer = ArrayDeque<TemporalFrame>(FRAMES)
    private var lastOfferedAt = 0L

    private val modelBitmap = Bitmap.createBitmap(INPUT_W, INPUT_H, Bitmap.Config.ARGB_8888)
    private val canvas = Canvas(modelBitmap)
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)
    private val pixels = IntArray(INPUT_W * INPUT_H)

    private val inputByteBuffer =
        ByteBuffer.allocateDirect(1 * FRAMES * CHANNELS_PER_FRAME * INPUT_W * INPUT_H * 4)
            .order(ByteOrder.nativeOrder())
    private val inputBuffer: FloatBuffer = inputByteBuffer.asFloatBuffer()

    init {
        val modelBytes = context.assets.open(MODEL_FILE).use { it.readBytes() }

        var chosenOptions = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(4)
            setInterOpNumThreads(1)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        var chosenBackend = "CPU"

        val createdSession = try {
            chosenOptions.addNnapi(EnumSet.of(NNAPIFlags.USE_FP16))
            chosenBackend = "NNAPI_FP16"
            environment.createSession(modelBytes, chosenOptions)
        } catch (_: Exception) {
            chosenOptions.close()
            chosenOptions = OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(4)
                setInterOpNumThreads(1)
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            }
            chosenBackend = "CPU"
            environment.createSession(modelBytes, chosenOptions)
        }

        sessionOptions = chosenOptions
        session = createdSession
        backendName = chosenBackend
        inputName = session.inputNames.first()
    }

    /**
     * Preprocess and store a frame. This is intentionally separated from
     * inference so the 3-frame window remains temporally dense even on a slow
     * phone whose previous inference is still running.
     */
    @Synchronized
    fun offerFrame(
        bitmap: Bitmap,
        rotationDegrees: Int,
        captureTimeMs: Long
    ): Int {
        if (lastOfferedAt > 0L && captureTimeMs - lastOfferedAt < MIN_FRAME_INTERVAL_MS) {
            return frameBuffer.size
        }

        val rotated = rotate(bitmap, rotationDegrees)
        try {
            val preprocessed = preprocess(rotated)
            frameBuffer.addLast(
                TemporalFrame(
                    chw = preprocessed.chw,
                    captureTimeMs = captureTimeMs,
                    imageWidth = rotated.width,
                    imageHeight = rotated.height,
                    modelScale = preprocessed.scale,
                    modelOffsetX = preprocessed.offsetX,
                    modelOffsetY = preprocessed.offsetY
                )
            )
            while (frameBuffer.size > FRAMES) frameBuffer.removeFirst()
            lastOfferedAt = captureTimeMs
            return frameBuffer.size
        } finally {
            if (rotated !== bitmap && !rotated.isRecycled) rotated.recycle()
        }
    }

    @Synchronized
    fun resetFrames() {
        frameBuffer.clear()
        lastOfferedAt = 0L
    }

    fun detectLatest(): Result? {
        val frames = synchronized(this) {
            if (frameBuffer.size < FRAMES) return null
            frameBuffer.toList()
        }

        val start = SystemClock.uptimeMillis()
        inputBuffer.clear()
        frames.forEach { frame ->
            inputBuffer.put(frame.chw)
        }
        inputBuffer.flip()

        val output = OnnxTensor.createTensor(
            environment,
            inputBuffer,
            longArrayOf(1, 9, INPUT_H.toLong(), INPUT_W.toLong())
        ).use { tensor ->
            session.run(mapOf(inputName to tensor)).use { result ->
                @Suppress("UNCHECKED_CAST")
                (result[0].value as Array<Array<Array<FloatArray>>>)
            }
        }

        val heatmap = output[0][OUTPUT_FRAME_INDEX]
        val rawPeak = rawPeak(heatmap)
        val component = strongestComponent(heatmap)
        val latest = frames.last()

        fun modelToNormalized(modelX: Float, modelY: Float): Pair<Float, Float> {
            val sourceX =
                (modelX - latest.modelOffsetX) / latest.modelScale
            val sourceY =
                (modelY - latest.modelOffsetY) / latest.modelScale
            return Pair(
                (sourceX / latest.imageWidth).coerceIn(0f, 1f),
                (sourceY / latest.imageHeight).coerceIn(0f, 1f)
            )
        }

        val rawPeakNorm = modelToNormalized(rawPeak.x, rawPeak.y)

        val detection = component?.let { blob ->
            val center = modelToNormalized(blob.cx, blob.cy)
            val cx = center.first
            val cy = center.second

            // Approximate an 8 model-pixel radius in original-frame
            // normalized coordinates after undoing the official affine scale.
            val halfSourcePx = 8f / latest.modelScale
            val halfW = halfSourcePx / latest.imageWidth
            val halfH = halfSourcePx / latest.imageHeight

            AiDetection(
                label = "sports ball",
                score = blob.maxProbability,
                box = RectF(
                    (cx - halfW).coerceIn(0f, 1f),
                    (cy - halfH).coerceIn(0f, 1f),
                    (cx + halfW).coerceIn(0f, 1f),
                    (cy + halfH).coerceIn(0f, 1f)
                ),
                source = "WASB_HRNET"
            )
        }

        return Result(
            detection = detection,
            inferenceMs = SystemClock.uptimeMillis() - start,
            maxHeat = component?.maxProbability ?: 0f,
            rawPeakProbability = rawPeak.probability,
            rawPeakX = rawPeakNorm.first,
            rawPeakY = rawPeakNorm.second,
            blobPixels = component?.pixels ?: 0,
            backend = backendName,
            captureTimeMs = latest.captureTimeMs,
            imageWidth = latest.imageWidth,
            imageHeight = latest.imageHeight
        )
    }

    private data class Blob(
        val cx: Float,
        val cy: Float,
        val pixels: Int,
        val maxProbability: Float,
        val weightSum: Float
    )

    private data class RawPeak(
        val x: Float,
        val y: Float,
        val probability: Float
    )

    private fun rawPeak(heatmap: Array<FloatArray>): RawPeak {
        var maxLogit = Float.NEGATIVE_INFINITY
        var bestX = 0
        var bestY = 0
        for (y in 0 until INPUT_H) {
            val row = heatmap[y]
            for (x in 0 until INPUT_W) {
                if (row[x] > maxLogit) {
                    maxLogit = row[x]
                    bestX = x
                    bestY = y
                }
            }
        }
        return RawPeak(
            x = bestX.toFloat(),
            y = bestY.toFloat(),
            probability = sigmoid(maxLogit)
        )
    }

    private fun strongestComponent(heatmap: Array<FloatArray>): Blob? {
        var maxLogit = Float.NEGATIVE_INFINITY
        for (y in 0 until INPUT_H) {
            val row = heatmap[y]
            for (x in 0 until INPUT_W) {
                if (row[x] > maxLogit) maxLogit = row[x]
            }
        }
        if (maxLogit <= LOGIT_THRESHOLD) return null

        val visited = BooleanArray(INPUT_W * INPUT_H)
        val queue = IntArray(INPUT_W * INPUT_H)
        var best: Blob? = null

        for (sy in 0 until INPUT_H) {
            for (sx in 0 until INPUT_W) {
                val startIdx = sy * INPUT_W + sx
                if (visited[startIdx] || heatmap[sy][sx] <= LOGIT_THRESHOLD) continue

                var head = 0
                var tail = 0
                queue[tail++] = startIdx
                visited[startIdx] = true

                var count = 0
                var weightedX = 0f
                var weightedY = 0f
                var weightSum = 0f
                var componentMax = Float.NEGATIVE_INFINITY

                while (head < tail) {
                    val idx = queue[head++]
                    val x = idx % INPUT_W
                    val y = idx / INPUT_W
                    val logit = heatmap[y][x]
                    val prob = sigmoid(logit)

                    count += 1
                    weightedX += x * prob
                    weightedY += y * prob
                    weightSum += prob
                    if (logit > componentMax) componentMax = logit

                    val x0 = max(0, x - 1)
                    val x1 = min(INPUT_W - 1, x + 1)
                    val y0 = max(0, y - 1)
                    val y1 = min(INPUT_H - 1, y + 1)
                    for (ny in y0..y1) {
                        for (nx in x0..x1) {
                            val nidx = ny * INPUT_W + nx
                            if (!visited[nidx] && heatmap[ny][nx] > LOGIT_THRESHOLD) {
                                visited[nidx] = true
                                queue[tail++] = nidx
                            }
                        }
                    }
                }

                if (count <= 0 || weightSum <= 0f) continue
                val candidate = Blob(
                    cx = weightedX / weightSum,
                    cy = weightedY / weightSum,
                    pixels = count,
                    maxProbability = sigmoid(componentMax),
                    weightSum = weightSum
                )
                if (best == null || candidate.weightSum > best!!.weightSum) {
                    best = candidate
                }
            }
        }
        return best
    }

    private fun preprocess(bitmap: Bitmap): PreprocessedFrame {
        // Match WASB-SBDT's official get_transform(): use
        // s=max(width,height), map that square with scale INPUT_W/s, center it
        // at (INPUT_W/2, INPUT_H/2), and let the 512x288 output canvas crop
        // the excess. This is NOT a direct 4:3 -> 16:9 stretch.
        val sourceW = bitmap.width.toFloat()
        val sourceH = bitmap.height.toFloat()
        val sourceScaleBasis = max(sourceW, sourceH)
        val scale = INPUT_W.toFloat() / sourceScaleBasis
        val scaledW = sourceW * scale
        val scaledH = sourceH * scale
        val offsetX = (INPUT_W - scaledW) / 2f
        val offsetY = (INPUT_H - scaledH) / 2f

        canvas.drawColor(Color.BLACK)
        canvas.drawBitmap(
            bitmap,
            null,
            RectF(
                offsetX,
                offsetY,
                offsetX + scaledW,
                offsetY + scaledH
            ),
            paint
        )
        modelBitmap.getPixels(pixels, 0, INPUT_W, 0, 0, INPUT_W, INPUT_H)

        val planeSize = INPUT_W * INPUT_H
        val out = FloatArray(CHANNELS_PER_FRAME * planeSize)
        for (i in pixels.indices) {
            val p = pixels[i]
            val r = ((p shr 16) and 0xFF) / 255f
            val g = ((p shr 8) and 0xFF) / 255f
            val b = (p and 0xFF) / 255f
            out[i] = (r - MEAN[0]) / STD[0]
            out[planeSize + i] = (g - MEAN[1]) / STD[1]
            out[2 * planeSize + i] = (b - MEAN[2]) / STD[2]
        }

        return PreprocessedFrame(
            chw = out,
            scale = scale,
            offsetX = offsetX,
            offsetY = offsetY
        )
    }

    private fun sigmoid(v: Float): Float =
        (1.0 / (1.0 + exp(-v.toDouble()))).toFloat()

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
        frameBuffer.clear()
    }
}
