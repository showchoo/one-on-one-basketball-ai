package jp.showchoo.oneononeai

import android.graphics.Bitmap
import android.graphics.RectF
import android.os.SystemClock
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * v0.4.1 high-frequency ball tracker.
 *
 * Key design:
 * - YOLO is an occasional detector/re-acquisition source.
 * - Camera frames are retained in a small grayscale ring buffer.
 * - When a delayed YOLO result arrives, tracking is replayed from the
 *   historical capture frame to the newest buffered frame.
 * - Live frames then use local normalized-template matching + motion prediction.
 *
 * This avoids starting the tracker from a 250-400 ms stale ball position.
 */
class FastBallTracker(
    private val onDebugEvent: (String) -> Unit = {}
) {
    data class Result(
        val box: RectF,
        val confidence: Float,
        val source: String,
        val trackingMs: Long,
        val ageSinceYoloMs: Long
    )

    private data class GrayFrame(
        val timeMs: Long,
        val width: Int,
        val height: Int,
        val pixels: ByteArray
    )

    private data class Match(
        val x: Float,
        val y: Float,
        val score: Float,
        val secondScore: Float
    )

    private val frames = ArrayDeque<GrayFrame>()

    private var boxNorm: RectF? = null
    private var template: FloatArray? = null
    private var vx = 0f
    private var vy = 0f
    private var lastUpdateMs = 0L
    private var lastYoloReceivedMs = 0L
    private var misses = 0

    private var rawPixels = IntArray(0)
    private var rawWidth = 0
    private var rawHeight = 0

    companion object {
        private const val MAX_BUFFER_MS = 1050L
        private const val MAX_YOLO_AGE_MS = 1500L
        private const val TEMPLATE_GRID = 9
        private const val MIN_MATCH = 0.43f
        private const val MIN_MARGIN = 0.035f
    }

    @Synchronized
    fun reset() {
        frames.clear()
        boxNorm = null
        template = null
        vx = 0f
        vy = 0f
        lastUpdateMs = 0L
        lastYoloReceivedMs = 0L
        misses = 0
    }

    /**
     * Called for every camera frame, even before the first YOLO ball detection.
     * This is what makes historical replay possible.
     */
    @Synchronized
    fun track(
        bitmap: Bitmap,
        rotationDegrees: Int,
        nowMs: Long
    ): Result? {
        val start = SystemClock.uptimeMillis()
        val frame = makeGrayFrame(bitmap, rotationDegrees, nowMs)
        appendFrame(frame)

        val current = boxNorm ?: return null
        val tpl = template ?: return null

        if (lastYoloReceivedMs > 0L && nowMs - lastYoloReceivedMs > MAX_YOLO_AGE_MS) {
            clearActiveTrack("FAST_TRACK_EXPIRED")
            return null
        }

        val dt = (nowMs - lastUpdateMs).coerceIn(1L, 140L) / 1000f
        val predictedX = (centerX(current) + vx * dt).coerceIn(0.01f, 0.99f)
        val predictedY = (
            centerY(current) + vy * dt + 0.5f * 0.72f * dt * dt
        ).coerceIn(0.01f, 0.99f)

        val speedPx = hypot(
            (vx * frame.width).toDouble(),
            (vy * frame.height).toDouble()
        ).toFloat()

        val radiusPx = (
            max(12f, max(current.width() * frame.width, current.height() * frame.height) * 2.2f) +
                min(speedPx * dt * 1.6f, 28f) +
                misses * 4f
        ).coerceIn(12f, 42f)

        val match = search(
            frame = frame,
            tpl = tpl,
            predictedX = predictedX,
            predictedY = predictedY,
            wNorm = current.width(),
            hNorm = current.height(),
            radiusPx = radiusPx
        )

        val accepted =
            match != null &&
                match.score >= MIN_MATCH &&
                match.score - match.secondScore >= MIN_MARGIN

        if (accepted && match != null) {
            val next = boxAt(
                match.x,
                match.y,
                current.width(),
                current.height()
            )

            val measuredVx = ((match.x - centerX(current)) / dt).coerceIn(-3f, 3f)
            val measuredVy = ((match.y - centerY(current)) / dt).coerceIn(-3f, 3f)
            vx = vx * 0.58f + measuredVx * 0.42f
            vy = vy * 0.58f + measuredVy * 0.42f

            boxNorm = next
            lastUpdateMs = nowMs
            misses = 0

            return Result(
                box = RectF(next),
                confidence = ((match.score + 1f) / 2f).coerceIn(0f, 1f),
                source = "FAST_TRACK",
                trackingMs = SystemClock.uptimeMillis() - start,
                ageSinceYoloMs =
                    if (lastYoloReceivedMs > 0L) nowMs - lastYoloReceivedMs else Long.MAX_VALUE
            )
        }

        misses += 1
        if (misses > 3) {
            clearActiveTrack("FAST_TRACK_LOST")
            return null
        }

        val predicted = boxAt(
            predictedX,
            predictedY,
            current.width(),
            current.height()
        )
        boxNorm = predicted
        lastUpdateMs = nowMs

        return Result(
            box = RectF(predicted),
            confidence = (0.34f - misses * 0.06f).coerceAtLeast(0.12f),
            source = "FAST_PREDICT",
            trackingMs = SystemClock.uptimeMillis() - start,
            ageSinceYoloMs =
                if (lastYoloReceivedMs > 0L) nowMs - lastYoloReceivedMs else Long.MAX_VALUE
        )
    }

    /**
     * Called when YOLO finishes. The box belongs to captureTimeMs, not to now.
     * Replay the retained frames from capture time forward so the live track is
     * brought to the current moment before it is exposed to the UI/scorer.
     */
    @Synchronized
    fun anchor(
        normalizedBox: RectF,
        bitmap: Bitmap,
        rotationDegrees: Int,
        captureTimeMs: Long,
        receivedTimeMs: Long,
        detectorConfidence: Float
    ) {
        val captureFrame = makeGrayFrame(bitmap, rotationDegrees, captureTimeMs)
        val initialTemplate = extractPatch(
            frame = captureFrame,
            cxNorm = centerX(normalizedBox),
            cyNorm = centerY(normalizedBox),
            wNorm = normalizedBox.width(),
            hNorm = normalizedBox.height()
        ) ?: run {
            onDebugEvent("FAST_ANCHOR_REJECT no_template")
            return
        }

        var currentBox = RectF(normalizedBox)
        var currentTemplate = initialTemplate
        var currentTime = captureTimeMs

        // Estimate velocity from the previous active track only when YOLO agrees
        // reasonably well. Otherwise start fresh from this detector anchor.
        val existing = boxNorm
        var localVx = 0f
        var localVy = 0f
        if (existing != null && lastUpdateMs > 0L) {
            val distance = hypot(
                (centerX(existing) - centerX(normalizedBox)).toDouble(),
                (centerY(existing) - centerY(normalizedBox)).toDouble()
            ).toFloat()
            if (distance < 0.16f) {
                localVx = vx
                localVy = vy
            }
        }

        val replayFrames = frames
            .filter { it.timeMs > captureTimeMs && it.timeMs <= receivedTimeMs + 80L }
            .sortedBy { it.timeMs }

        var replayed = 0
        var failed = 0

        for (frame in replayFrames) {
            val dt = (frame.timeMs - currentTime).coerceIn(1L, 160L) / 1000f
            val predictedX = (
                centerX(currentBox) + localVx * dt
            ).coerceIn(0.01f, 0.99f)
            val predictedY = (
                centerY(currentBox) + localVy * dt + 0.5f * 0.72f * dt * dt
            ).coerceIn(0.01f, 0.99f)

            val radiusPx = (
                max(14f, max(currentBox.width() * frame.width, currentBox.height() * frame.height) * 2.4f) +
                    min(
                        hypot(
                            (localVx * frame.width).toDouble(),
                            (localVy * frame.height).toDouble()
                        ).toFloat() * dt * 1.7f,
                        34f
                    ) +
                    failed * 6f
            ).coerceIn(14f, 54f)

            val match = search(
                frame = frame,
                tpl = currentTemplate,
                predictedX = predictedX,
                predictedY = predictedY,
                wNorm = currentBox.width(),
                hNorm = currentBox.height(),
                radiusPx = radiusPx
            )

            val accepted =
                match != null &&
                    match.score >= 0.40f &&
                    match.score - match.secondScore >= 0.025f

            if (accepted && match != null) {
                val oldX = centerX(currentBox)
                val oldY = centerY(currentBox)

                val measuredVx = ((match.x - oldX) / dt).coerceIn(-3f, 3f)
                val measuredVy = ((match.y - oldY) / dt).coerceIn(-3f, 3f)
                localVx = localVx * 0.52f + measuredVx * 0.48f
                localVy = localVy * 0.52f + measuredVy * 0.48f

                currentBox = boxAt(
                    match.x,
                    match.y,
                    currentBox.width(),
                    currentBox.height()
                )
                currentTime = frame.timeMs
                replayed += 1
                failed = 0

                // Very slow template adaptation; keeps YOLO identity while
                // allowing modest lighting/rotation change.
                extractPatch(
                    frame,
                    match.x,
                    match.y,
                    currentBox.width(),
                    currentBox.height()
                )?.let { fresh ->
                    currentTemplate = blendTemplate(
                        currentTemplate,
                        fresh,
                        0.06f
                    )
                }
            } else {
                failed += 1
                if (failed >= 2) break

                currentBox = boxAt(
                    predictedX,
                    predictedY,
                    currentBox.width(),
                    currentBox.height()
                )
                currentTime = frame.timeMs
            }
        }

        boxNorm = currentBox
        template = currentTemplate
        vx = localVx
        vy = localVy
        lastUpdateMs = max(currentTime, receivedTimeMs)
        lastYoloReceivedMs = receivedTimeMs
        misses = 0

        onDebugEvent(
            "FAST_ANCHOR_REPLAY conf=$detectorConfidence latency=" +
                (receivedTimeMs - captureTimeMs) +
                " replayed=$replayed failed=$failed x=" +
                centerX(currentBox) + " y=" + centerY(currentBox)
        )
    }

    private fun search(
        frame: GrayFrame,
        tpl: FloatArray,
        predictedX: Float,
        predictedY: Float,
        wNorm: Float,
        hNorm: Float,
        radiusPx: Float
    ): Match? {
        val px = predictedX * frame.width
        val py = predictedY * frame.height

        val coarseStep = when {
            radiusPx >= 40f -> 4
            radiusPx >= 26f -> 3
            else -> 2
        }

        var bestScore = -2f
        var secondScore = -2f
        var bestX = predictedX
        var bestY = predictedY

        fun consider(xPx: Int, yPx: Int) {
            val xNorm = xPx.toFloat() / frame.width
            val yNorm = yPx.toFloat() / frame.height
            val candidate = extractPatch(
                frame,
                xNorm,
                yNorm,
                wNorm,
                hNorm
            ) ?: return

            val score = correlation(tpl, candidate)
            if (score > bestScore) {
                secondScore = bestScore
                bestScore = score
                bestX = xNorm
                bestY = yNorm
            } else if (score > secondScore) {
                secondScore = score
            }
        }

        val left = max(2, (px - radiusPx).toInt())
        val right = min(frame.width - 3, (px + radiusPx).toInt())
        val top = max(2, (py - radiusPx).toInt())
        val bottom = min(frame.height - 3, (py + radiusPx).toInt())

        var y = top
        while (y <= bottom) {
            var x = left
            while (x <= right) {
                val d = hypot((x - px).toDouble(), (y - py).toDouble()).toFloat()
                if (d <= radiusPx) consider(x, y)
                x += coarseStep
            }
            y += coarseStep
        }

        if (bestScore <= -1.5f) return null

        // Sub-pixel-ish local refinement at one-pixel resolution.
        val bx = (bestX * frame.width).toInt()
        val by = (bestY * frame.height).toInt()
        for (yy in max(2, by - coarseStep)..min(frame.height - 3, by + coarseStep)) {
            for (xx in max(2, bx - coarseStep)..min(frame.width - 3, bx + coarseStep)) {
                consider(xx, yy)
            }
        }

        return Match(
            x = bestX,
            y = bestY,
            score = bestScore,
            secondScore = secondScore
        )
    }

    private fun extractPatch(
        frame: GrayFrame,
        cxNorm: Float,
        cyNorm: Float,
        wNorm: Float,
        hNorm: Float
    ): FloatArray? {
        val cx = cxNorm * frame.width
        val cy = cyNorm * frame.height

        val rx = max(3f, wNorm * frame.width * 0.62f)
        val ry = max(3f, hNorm * frame.height * 0.62f)

        val values = FloatArray(TEMPLATE_GRID * TEMPLATE_GRID)
        var index = 0
        var sum = 0f

        for (gy in 0 until TEMPLATE_GRID) {
            val fy = (gy + 0.5f) / TEMPLATE_GRID
            val y = (cy + (fy - 0.5f) * 2f * ry).toInt()
            if (y !in 1 until frame.height - 1) return null

            for (gx in 0 until TEMPLATE_GRID) {
                val fx = (gx + 0.5f) / TEMPLATE_GRID
                val x = (cx + (fx - 0.5f) * 2f * rx).toInt()
                if (x !in 1 until frame.width - 1) return null

                val value = (frame.pixels[y * frame.width + x].toInt() and 0xFF).toFloat()
                values[index++] = value
                sum += value
            }
        }

        val mean = sum / values.size
        var variance = 0f
        for (i in values.indices) {
            values[i] -= mean
            variance += values[i] * values[i]
        }

        val std = sqrt(variance / values.size)
        if (std < 5.0f) return null

        val inv = 1f / std
        for (i in values.indices) values[i] *= inv
        return values
    }

    private fun correlation(a: FloatArray, b: FloatArray): Float {
        if (a.size != b.size || a.isEmpty()) return -1f
        var dot = 0f
        for (i in a.indices) dot += a[i] * b[i]
        return (dot / a.size).coerceIn(-1f, 1f)
    }

    private fun blendTemplate(
        old: FloatArray,
        fresh: FloatArray,
        alpha: Float
    ): FloatArray {
        if (old.size != fresh.size) return old
        val mixed = FloatArray(old.size)
        var mean = 0f
        for (i in old.indices) {
            mixed[i] = old[i] * (1f - alpha) + fresh[i] * alpha
            mean += mixed[i]
        }
        mean /= mixed.size

        var variance = 0f
        for (i in mixed.indices) {
            mixed[i] -= mean
            variance += mixed[i] * mixed[i]
        }

        val std = sqrt(variance / mixed.size).coerceAtLeast(0.001f)
        for (i in mixed.indices) mixed[i] /= std
        return mixed
    }

    private fun makeGrayFrame(
        bitmap: Bitmap,
        rotationDegrees: Int,
        timeMs: Long
    ): GrayFrame {
        val rotation = normalizedRotation(rotationDegrees)
        val rawW = bitmap.width
        val rawH = bitmap.height
        val orientedW = if (rotation == 90 || rotation == 270) rawH else rawW
        val orientedH = if (rotation == 90 || rotation == 270) rawW else rawH

        val targetW = min(320, orientedW)
        val targetH = max(1, (orientedH * targetW.toFloat() / orientedW).toInt())

        if (rawPixels.size != rawW * rawH || rawWidth != rawW || rawHeight != rawH) {
            rawPixels = IntArray(rawW * rawH)
            rawWidth = rawW
            rawHeight = rawH
        }
        bitmap.getPixels(rawPixels, 0, rawW, 0, 0, rawW, rawH)

        val gray = ByteArray(targetW * targetH)

        for (ty in 0 until targetH) {
            val oy = ((ty + 0.5f) * orientedH / targetH).toInt()
                .coerceIn(0, orientedH - 1)

            for (tx in 0 until targetW) {
                val ox = ((tx + 0.5f) * orientedW / targetW).toInt()
                    .coerceIn(0, orientedW - 1)

                val rawIndex = orientedToRawIndex(
                    ox,
                    oy,
                    rawW,
                    rawH,
                    rotation
                )
                val pixel = rawPixels[rawIndex]
                val r = (pixel shr 16) and 0xFF
                val g = (pixel shr 8) and 0xFF
                val b = pixel and 0xFF
                val y = (77 * r + 150 * g + 29 * b) shr 8
                gray[ty * targetW + tx] = y.toByte()
            }
        }

        return GrayFrame(
            timeMs = timeMs,
            width = targetW,
            height = targetH,
            pixels = gray
        )
    }

    private fun appendFrame(frame: GrayFrame) {
        frames.addLast(frame)
        while (frames.size > 36) frames.removeFirst()
        while (frames.isNotEmpty() && frame.timeMs - frames.first().timeMs > MAX_BUFFER_MS) {
            frames.removeFirst()
        }
    }

    private fun orientedToRawIndex(
        x: Int,
        y: Int,
        rawW: Int,
        rawH: Int,
        rotation: Int
    ): Int {
        val rawX: Int
        val rawY: Int

        when (rotation) {
            90 -> {
                rawX = y
                rawY = rawH - 1 - x
            }
            180 -> {
                rawX = rawW - 1 - x
                rawY = rawH - 1 - y
            }
            270 -> {
                rawX = rawW - 1 - y
                rawY = x
            }
            else -> {
                rawX = x
                rawY = y
            }
        }

        val safeX = rawX.coerceIn(0, rawW - 1)
        val safeY = rawY.coerceIn(0, rawH - 1)
        return safeY * rawW + safeX
    }

    private fun normalizedRotation(degrees: Int): Int {
        val d = ((degrees % 360) + 360) % 360
        return when {
            d < 45 -> 0
            d < 135 -> 90
            d < 225 -> 180
            d < 315 -> 270
            else -> 0
        }
    }

    private fun boxAt(
        cx: Float,
        cy: Float,
        width: Float,
        height: Float
    ): RectF =
        RectF(
            (cx - width / 2f).coerceIn(0f, 1f),
            (cy - height / 2f).coerceIn(0f, 1f),
            (cx + width / 2f).coerceIn(0f, 1f),
            (cy + height / 2f).coerceIn(0f, 1f)
        )

    private fun clearActiveTrack(reason: String) {
        boxNorm = null
        template = null
        vx = 0f
        vy = 0f
        lastUpdateMs = 0L
        misses = 0
        onDebugEvent(reason)
    }

    private fun centerX(r: RectF) = (r.left + r.right) / 2f
    private fun centerY(r: RectF) = (r.top + r.bottom) / 2f
}
