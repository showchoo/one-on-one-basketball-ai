package jp.showchoo.oneononeai

import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.RectF
import android.os.SystemClock
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

/**
 * High-frequency lightweight ball tracker.
 *
 * YOLO is only used as an anchor/re-detection source. Between YOLO passes this
 * tracker follows the ball using a constant-velocity model plus a tiny
 * appearance matcher around the predicted location.
 *
 * All returned boxes use the rotated/display coordinate system, normalized 0..1.
 */
class FastBallTracker {
    data class Result(
        val box: RectF,
        val confidence: Float,
        val source: String,
        val trackingMs: Long,
        val ageSinceYoloMs: Long
    )

    private data class Appearance(
        val cr: Float,
        val cg: Float,
        val cb: Float,
        val luminance: Float,
        val saturation: Float
    )

    private var boxNorm: RectF? = null
    private var vx = 0f
    private var vy = 0f
    private var lastUpdateMs = 0L
    private var lastYoloMs = 0L
    private var appearance: Appearance? = null
    private var misses = 0

    @Synchronized
    fun reset() {
        boxNorm = null
        vx = 0f
        vy = 0f
        lastUpdateMs = 0L
        lastYoloMs = 0L
        appearance = null
        misses = 0
    }

    @Synchronized
    fun anchor(
        normalizedBox: RectF,
        bitmap: Bitmap,
        rotationDegrees: Int,
        captureTimeMs: Long,
        receivedTimeMs: Long,
        detectorConfidence: Float
    ) {
        val sampled = sampleAppearance(
            bitmap = bitmap,
            rotationDegrees = rotationDegrees,
            box = normalizedBox,
            centerOverride = null
        )
        if (sampled != null) {
            appearance = if (appearance == null || detectorConfidence >= 0.22f) {
                sampled
            } else {
                blend(appearance!!, sampled, 0.22f)
            }
        }

        val old = boxNorm
        if (old == null || lastUpdateMs <= 0L) {
            boxNorm = RectF(normalizedBox)
            lastUpdateMs = receivedTimeMs
            lastYoloMs = receivedTimeMs
            misses = 0
            return
        }

        val ageSec = (receivedTimeMs - captureTimeMs).coerceIn(0L, 700L) / 1000f
        val projectedX = (
            centerX(normalizedBox) + vx * ageSec
        ).coerceIn(0.01f, 0.99f)
        val projectedY = (
            centerY(normalizedBox) + vy * ageSec + 0.5f * 0.70f * ageSec * ageSec
        ).coerceIn(0.01f, 0.99f)

        val currentX = centerX(old)
        val currentY = centerY(old)
        val correction = if (detectorConfidence >= 0.30f) 0.52f else 0.34f
        val correctedX = currentX + (projectedX - currentX) * correction
        val correctedY = currentY + (projectedY - currentY) * correction

        val corrected = RectF(
            (correctedX - normalizedBox.width() / 2f).coerceIn(0f, 1f),
            (correctedY - normalizedBox.height() / 2f).coerceIn(0f, 1f),
            (correctedX + normalizedBox.width() / 2f).coerceIn(0f, 1f),
            (correctedY + normalizedBox.height() / 2f).coerceIn(0f, 1f)
        )

        val dt = (receivedTimeMs - lastUpdateMs).coerceIn(1L, 500L) / 1000f
        val measuredVx = ((correctedX - currentX) / dt).coerceIn(-3f, 3f)
        val measuredVy = ((correctedY - currentY) / dt).coerceIn(-3f, 3f)
        vx = vx * 0.78f + measuredVx * 0.22f
        vy = vy * 0.78f + measuredVy * 0.22f

        boxNorm = corrected
        lastUpdateMs = receivedTimeMs
        lastYoloMs = receivedTimeMs
        misses = 0
    }

    @Synchronized
    fun track(
        bitmap: Bitmap,
        rotationDegrees: Int,
        nowMs: Long
    ): Result? {
        val start = SystemClock.uptimeMillis()
        val current = boxNorm ?: return null
        if (lastUpdateMs <= 0L) return null

        val gapSinceYolo = nowMs - lastYoloMs
        if (gapSinceYolo > 1400L) {
            reset()
            return null
        }

        val (ow, oh) = orientedSize(bitmap.width, bitmap.height, rotationDegrees)
        val dt = (nowMs - lastUpdateMs).coerceIn(1L, 160L) / 1000f

        val predictedX = (centerX(current) + vx * dt).coerceIn(0.01f, 0.99f)
        val predictedY = (
            centerY(current) + vy * dt + 0.5f * 0.70f * dt * dt
        ).coerceIn(0.01f, 0.99f)

        val wNorm = current.width().coerceIn(0.008f, 0.10f)
        val hNorm = current.height().coerceIn(0.008f, 0.10f)
        val wPx = max(5f, wNorm * ow)
        val hPx = max(5f, hNorm * oh)

        val speedPx = hypot((vx * ow).toDouble(), (vy * oh).toDouble()).toFloat()
        val searchRadius = (
            max(max(wPx, hPx) * 2.2f, 16f) +
                min(speedPx * dt * 1.8f, 34f) +
                misses * 5f
        ).coerceIn(16f, 58f)

        val px = predictedX * ow
        val py = predictedY * oh
        val targetAppearance = appearance

        var bestX = px
        var bestY = py
        var bestScore = 0f

        if (targetAppearance != null) {
            val step = when {
                searchRadius > 44f -> 4
                searchRadius > 28f -> 3
                else -> 2
            }

            var y = (py - searchRadius).toInt()
            val yEnd = (py + searchRadius).toInt()
            while (y <= yEnd) {
                if (y >= 2 && y < oh - 2) {
                    var x = (px - searchRadius).toInt()
                    val xEnd = (px + searchRadius).toInt()
                    while (x <= xEnd) {
                        if (x >= 2 && x < ow - 2) {
                            val d = hypot((x - px).toDouble(), (y - py).toDouble()).toFloat()
                            if (d <= searchRadius) {
                                val candidate = sampleAppearance(
                                    bitmap = bitmap,
                                    rotationDegrees = rotationDegrees,
                                    box = current,
                                    centerOverride = Pair(x.toFloat() / ow, y.toFloat() / oh)
                                )
                                if (candidate != null) {
                                    val appearanceScore = similarity(targetAppearance, candidate)
                                    val motionScore = exp(
                                        -(d * d) /
                                            (2f * max(10f, searchRadius * 0.58f) *
                                                max(10f, searchRadius * 0.58f))
                                    )
                                    val score =
                                        appearanceScore * 0.78f + motionScore.toFloat() * 0.22f
                                    if (score > bestScore) {
                                        bestScore = score
                                        bestX = x.toFloat()
                                        bestY = y.toFloat()
                                    }
                                }
                            }
                        }
                        x += step
                    }
                }
                y += step
            }
        }

        val accepted = bestScore >= if (misses == 0) 0.60f else 0.64f
        val nextX: Float
        val nextY: Float
        val source: String
        val confidence: Float

        if (accepted) {
            nextX = bestX / ow
            nextY = bestY / oh
            source = "FAST_TRACK"
            confidence = bestScore.coerceIn(0f, 1f)
            misses = 0

            val measuredVx = ((nextX - centerX(current)) / dt).coerceIn(-3f, 3f)
            val measuredVy = ((nextY - centerY(current)) / dt).coerceIn(-3f, 3f)
            vx = vx * 0.62f + measuredVx * 0.38f
            vy = vy * 0.62f + measuredVy * 0.38f

            sampleAppearance(
                bitmap,
                rotationDegrees,
                current,
                Pair(nextX, nextY)
            )?.let { fresh ->
                appearance = appearance?.let { blend(it, fresh, 0.08f) } ?: fresh
            }
        } else {
            misses += 1
            if (misses > 7 || gapSinceYolo > 700L) {
                return null
            }
            nextX = predictedX
            nextY = predictedY
            source = "FAST_PREDICT"
            confidence = (0.48f - misses * 0.045f).coerceAtLeast(0.16f)
        }

        val next = RectF(
            (nextX - wNorm / 2f).coerceIn(0f, 1f),
            (nextY - hNorm / 2f).coerceIn(0f, 1f),
            (nextX + wNorm / 2f).coerceIn(0f, 1f),
            (nextY + hNorm / 2f).coerceIn(0f, 1f)
        )

        boxNorm = next
        lastUpdateMs = nowMs

        return Result(
            box = RectF(next),
            confidence = confidence,
            source = source,
            trackingMs = SystemClock.uptimeMillis() - start,
            ageSinceYoloMs = gapSinceYolo
        )
    }

    private fun sampleAppearance(
        bitmap: Bitmap,
        rotationDegrees: Int,
        box: RectF,
        centerOverride: Pair<Float, Float>?
    ): Appearance? {
        val (ow, oh) = orientedSize(bitmap.width, bitmap.height, rotationDegrees)
        val cx = (centerOverride?.first ?: centerX(box)) * ow
        val cy = (centerOverride?.second ?: centerY(box)) * oh
        val rx = max(2.2f, box.width() * ow * 0.34f)
        val ry = max(2.2f, box.height() * oh * 0.34f)

        val offsets = arrayOf(
            0f to 0f,
            -0.55f to 0f,
            0.55f to 0f,
            0f to -0.55f,
            0f to 0.55f,
            -0.38f to -0.38f,
            0.38f to -0.38f,
            -0.38f to 0.38f,
            0.38f to 0.38f
        )

        var cr = 0f
        var cg = 0f
        var cb = 0f
        var lum = 0f
        var sat = 0f
        var count = 0

        for ((ox, oy) in offsets) {
            val x = (cx + ox * rx).toInt()
            val y = (cy + oy * ry).toInt()
            val pixel = orientedPixel(bitmap, rotationDegrees, x, y) ?: continue
            val r = Color.red(pixel) / 255f
            val g = Color.green(pixel) / 255f
            val b = Color.blue(pixel) / 255f
            val sum = r + g + b
            if (sum < 0.06f) continue

            val maxC = max(r, max(g, b))
            val minC = min(r, min(g, b))
            cr += r / sum
            cg += g / sum
            cb += b / sum
            lum += 0.299f * r + 0.587f * g + 0.114f * b
            sat += if (maxC <= 0.001f) 0f else (maxC - minC) / maxC
            count++
        }

        if (count < 5) return null
        return Appearance(
            cr / count,
            cg / count,
            cb / count,
            lum / count,
            sat / count
        )
    }

    private fun similarity(a: Appearance, b: Appearance): Float {
        val chroma = kotlin.math.sqrt(
            (a.cr - b.cr) * (a.cr - b.cr) +
                (a.cg - b.cg) * (a.cg - b.cg) +
                (a.cb - b.cb) * (a.cb - b.cb)
        )
        val luminance = abs(a.luminance - b.luminance)
        val saturation = abs(a.saturation - b.saturation)
        val distance = chroma * 2.8f + luminance * 0.55f + saturation * 0.45f
        return exp(-distance * 2.1f).toFloat().coerceIn(0f, 1f)
    }

    private fun blend(a: Appearance, b: Appearance, t: Float): Appearance =
        Appearance(
            a.cr + (b.cr - a.cr) * t,
            a.cg + (b.cg - a.cg) * t,
            a.cb + (b.cb - a.cb) * t,
            a.luminance + (b.luminance - a.luminance) * t,
            a.saturation + (b.saturation - a.saturation) * t
        )

    private fun orientedPixel(
        bitmap: Bitmap,
        rotationDegrees: Int,
        x: Int,
        y: Int
    ): Int? {
        val rotation = normalizedRotation(rotationDegrees)
        val (ow, oh) = orientedSize(bitmap.width, bitmap.height, rotation)
        if (x !in 0 until ow || y !in 0 until oh) return null

        val rw = bitmap.width
        val rh = bitmap.height
        val rawX: Int
        val rawY: Int

        when (rotation) {
            90 -> {
                rawX = y
                rawY = rh - 1 - x
            }
            180 -> {
                rawX = rw - 1 - x
                rawY = rh - 1 - y
            }
            270 -> {
                rawX = rw - 1 - y
                rawY = x
            }
            else -> {
                rawX = x
                rawY = y
            }
        }

        if (rawX !in 0 until rw || rawY !in 0 until rh) return null
        return bitmap.getPixel(rawX, rawY)
    }

    private fun orientedSize(rawW: Int, rawH: Int, rotationDegrees: Int): Pair<Int, Int> =
        when (normalizedRotation(rotationDegrees)) {
            90, 270 -> rawH to rawW
            else -> rawW to rawH
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

    private fun centerX(r: RectF) = (r.left + r.right) / 2f
    private fun centerY(r: RectF) = (r.top + r.bottom) / 2f
}
