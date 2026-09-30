package jp.showchoo.oneononeai

import android.graphics.RectF
import androidx.camera.core.ImageProxy
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/**
 * Cheap fixed-camera motion proposer.
 *
 * It does NOT decide "this is the basketball". It only finds compact moving
 * regions every camera frame, so the expensive basketball YOLO can inspect the
 * right small ROI instead of blindly cycling over the whole court.
 */
class MotionBallProposer {
    data class Proposal(
        val roi: RectF,
        val score: Float,
        val centerX: Float,
        val centerY: Float,
        val area: Int
    )

    private var previous: ByteArray? = null
    private var gridW = 0
    private var gridH = 0
    private var queue = IntArray(0)
    private var visited = BooleanArray(0)
    private var mask = BooleanArray(0)
    private var diffValues = IntArray(0)

    companion object {
        private const val TARGET_W = 240
        private const val DIFF_THRESHOLD = 24
        private const val MIN_AREA = 2
        private const val MAX_AREA = 95
        private const val MAX_BOX = 20
    }

    fun reset() {
        previous = null
        gridW = 0
        gridH = 0
    }

    fun update(image: ImageProxy): List<Proposal> {
        val plane = image.planes.firstOrNull() ?: return emptyList()
        val rawW = image.width
        val rawH = image.height
        if (rawW <= 0 || rawH <= 0) return emptyList()

        val newGridW = TARGET_W.coerceAtMost(rawW)
        val newGridH = max(1, (rawH * newGridW.toFloat() / rawW).toInt())
        val total = newGridW * newGridH

        if (newGridW != gridW || newGridH != gridH || mask.size != total) {
            gridW = newGridW
            gridH = newGridH
            queue = IntArray(total)
            visited = BooleanArray(total)
            mask = BooleanArray(total)
            diffValues = IntArray(total)
            previous = null
        }

        val buffer = plane.buffer.duplicate()
        val rowStride = plane.rowStride
        val pixelStride = plane.pixelStride
        val current = ByteArray(total)

        for (gy in 0 until gridH) {
            val rawY = ((gy + 0.5f) * rawH / gridH).toInt().coerceIn(0, rawH - 1)
            for (gx in 0 until gridW) {
                val rawX = ((gx + 0.5f) * rawW / gridW).toInt().coerceIn(0, rawW - 1)
                val base = rawY * rowStride + rawX * pixelStride

                val c0 = buffer.get(base).toInt() and 0xFF
                val c1 = if (base + 1 < buffer.limit()) buffer.get(base + 1).toInt() and 0xFF else c0
                val c2 = if (base + 2 < buffer.limit()) buffer.get(base + 2).toInt() and 0xFF else c0
                val lum = (c0 + c1 * 2 + c2) / 4
                current[gy * gridW + gx] = lum.toByte()
            }
        }

        val prev = previous
        previous = current
        if (prev == null || prev.size != current.size) return emptyList()

        java.util.Arrays.fill(visited, false)
        var activeCount = 0
        for (i in current.indices) {
            val d = abs((current[i].toInt() and 0xFF) - (prev[i].toInt() and 0xFF))
            diffValues[i] = d
            val active = d >= DIFF_THRESHOLD
            mask[i] = active
            if (active) activeCount++
        }

        // A camera bump or exposure jump changes a huge fraction of the image;
        // ignore that frame rather than flooding the ROI detector.
        if (activeCount > total / 5) return emptyList()

        val rawProposals = mutableListOf<Proposal>()
        val neighbors = intArrayOf(
            -1, 1, -gridW, gridW,
            -gridW - 1, -gridW + 1,
            gridW - 1, gridW + 1
        )

        for (seed in 0 until total) {
            if (!mask[seed] || visited[seed]) continue

            var head = 0
            var tail = 0
            queue[tail++] = seed
            visited[seed] = true

            var area = 0
            var sumDiff = 0
            var minX = gridW
            var maxX = 0
            var minY = gridH
            var maxY = 0
            var sumX = 0f
            var sumY = 0f

            while (head < tail) {
                val idx = queue[head++]
                val x = idx % gridW
                val y = idx / gridW

                area++
                sumDiff += diffValues[idx]
                sumX += x
                sumY += y
                if (x < minX) minX = x
                if (x > maxX) maxX = x
                if (y < minY) minY = y
                if (y > maxY) maxY = y

                for (delta in neighbors) {
                    val n = idx + delta
                    if (n < 0 || n >= total || visited[n] || !mask[n]) continue
                    val nx = n % gridW
                    val ny = n / gridW
                    if (abs(nx - x) > 1 || abs(ny - y) > 1) continue
                    visited[n] = true
                    queue[tail++] = n
                }
            }

            val bw = maxX - minX + 1
            val bh = maxY - minY + 1
            if (area !in MIN_AREA..MAX_AREA) continue
            if (bw > MAX_BOX || bh > MAX_BOX) continue

            val aspect = bw.toFloat() / bh.coerceAtLeast(1)
            if (aspect < 0.22f || aspect > 4.5f) continue

            val compactness = area.toFloat() / (bw * bh).coerceAtLeast(1)
            if (compactness < 0.12f) continue

            val rawCxNorm = ((sumX / area) + 0.5f) / gridW
            val rawCyNorm = ((sumY / area) + 0.5f) / gridH
            val oriented = rawToOriented(
                rawCxNorm,
                rawCyNorm,
                image.imageInfo.rotationDegrees
            )

            val meanDiff = sumDiff.toFloat() / area
            val score =
                (meanDiff / 255f) * 0.62f +
                    compactness.coerceAtMost(1f) * 0.23f +
                    (1f - (area - 14).let { abs(it) } / 90f).coerceIn(0f, 1f) * 0.15f

            val roi = centeredRoi(
                oriented.first,
                oriented.second,
                0.18f,
                0.24f
            )

            rawProposals += Proposal(
                roi = roi,
                score = score,
                centerX = oriented.first,
                centerY = oriented.second,
                area = area
            )
        }

        return rawProposals
            .sortedByDescending { it.score }
            .take(8)
    }

    private fun rawToOriented(
        x: Float,
        y: Float,
        rotationDegrees: Int
    ): Pair<Float, Float> {
        val rotation = normalizedRotation(rotationDegrees)
        return when (rotation) {
            90 -> Pair(1f - y, x)
            180 -> Pair(1f - x, 1f - y)
            270 -> Pair(y, 1f - x)
            else -> Pair(x, y)
        }
    }

    private fun centeredRoi(
        cx: Float,
        cy: Float,
        width: Float,
        height: Float
    ): RectF {
        val w = width.coerceIn(0.08f, 1f)
        val h = height.coerceIn(0.08f, 1f)
        var left = cx - w / 2f
        var top = cy - h / 2f
        left = left.coerceIn(0f, 1f - w)
        top = top.coerceIn(0f, 1f - h)
        return RectF(left, top, left + w, top + h)
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
}
