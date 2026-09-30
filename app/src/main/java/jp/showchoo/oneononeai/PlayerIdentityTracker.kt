package jp.showchoo.oneononeai

import android.graphics.RectF
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Persistent A/B identities driven by low-frequency YOLO detections.
 *
 * Identity is maintained with motion prediction + body appearance, rather than
 * re-labeling players left/right every detection frame.
 */
class PlayerIdentityTracker(
    private val onDebugEvent: (String) -> Unit = {}
) {
    data class Snapshot(
        val playerA: RectF?,
        val playerB: RectF?
    )

    private data class Observation(
        val box: RectF,
        val score: Float,
        val color: FloatArray?
    )

    private data class Track(
        val id: Char,
        var box: RectF,
        var lastSeenMs: Long,
        var vx: Float = 0f,
        var vy: Float = 0f,
        var color: FloatArray? = null
    )

    private var a: Track? = null
    private var b: Track? = null

    fun reset() {
        a = null
        b = null
        onDebugEvent("PLAYER_TRACKER_RESET")
    }

    fun update(
        detections: List<AiDetection>,
        imageWidth: Int,
        imageHeight: Int,
        nowMs: Long
    ) {
        val observations = detections
            .filter { it.label == "person" }
            .map {
                Observation(
                    box = normalize(it.box, imageWidth, imageHeight),
                    score = it.score,
                    color = if (
                        it.appearanceR >= 0f &&
                        it.appearanceG >= 0f &&
                        it.appearanceB >= 0f
                    ) {
                        floatArrayOf(it.appearanceR, it.appearanceG, it.appearanceB)
                    } else {
                        null
                    }
                )
            }
            .sortedByDescending { it.score }
            .take(5)

        if (observations.isEmpty()) return

        if (a == null || b == null) {
            if (observations.size < 2) return
            val initial = observations
                .sortedByDescending { it.score + area(it.box) }
                .take(2)
                .sortedBy { centerX(it.box) }

            a = Track('A', RectF(initial[0].box), nowMs, color = initial[0].color)
            b = Track('B', RectF(initial[1].box), nowMs, color = initial[1].color)
            onDebugEvent("PLAYER_IDS_LOCKED initial=left-right")
            return
        }

        val ta = a!!
        val tb = b!!

        var bestA: Observation? = null
        var bestB: Observation? = null
        var bestCost = Float.MAX_VALUE
        var secondBestCost = Float.MAX_VALUE

        for (i in observations.indices) {
            for (j in observations.indices) {
                if (i == j) continue
                val ca = cost(ta, observations[i], nowMs)
                val cb = cost(tb, observations[j], nowMs)
                if (!ca.isFinite() || !cb.isFinite()) continue
                val total = ca + cb

                if (total < bestCost) {
                    secondBestCost = bestCost
                    bestCost = total
                    bestA = observations[i]
                    bestB = observations[j]
                } else if (total < secondBestCost) {
                    secondBestCost = total
                }
            }
        }

        val assignmentClear =
            bestA != null &&
                bestB != null &&
                bestCost < 0.88f &&
                (secondBestCost == Float.MAX_VALUE || secondBestCost - bestCost > 0.035f)

        if (assignmentClear) {
            updateTrack(ta, bestA!!, nowMs)
            updateTrack(tb, bestB!!, nowMs)
            return
        }

        // Crossing/occlusion: update only an identity with an unambiguous match.
        // If ambiguous, freezing the ID is preferable to swapping A and B.
        val aCandidate = observations
            .map { it to cost(ta, it, nowMs) }
            .filter { it.second.isFinite() }
            .sortedBy { it.second }

        val bCandidate = observations
            .map { it to cost(tb, it, nowMs) }
            .filter { it.second.isFinite() }
            .sortedBy { it.second }

        val aBest = aCandidate.firstOrNull()
        val bBest = bCandidate.firstOrNull()

        var updatedA = false
        var updatedB = false

        if (
            aBest != null &&
            aBest.second < 0.34f &&
            (aCandidate.size == 1 || aCandidate[1].second - aBest.second > 0.06f) &&
            (bBest == null || aBest.first !== bBest.first || aBest.second + 0.08f < bBest.second)
        ) {
            updateTrack(ta, aBest.first, nowMs)
            updatedA = true
        }

        if (
            bBest != null &&
            bBest.second < 0.34f &&
            (bCandidate.size == 1 || bCandidate[1].second - bBest.second > 0.06f) &&
            (aBest == null || bBest.first !== aBest.first || bBest.second + 0.08f < aBest.second)
        ) {
            updateTrack(tb, bBest.first, nowMs)
            updatedB = true
        }

        if (!updatedA && !updatedB) {
            onDebugEvent("PLAYER_IDS_HELD ambiguous=true")
        }
    }

    fun snapshot(nowMs: Long): Snapshot {
        return Snapshot(
            playerA = predictedBox(a, nowMs),
            playerB = predictedBox(b, nowMs)
        )
    }

    private fun cost(track: Track, obs: Observation, nowMs: Long): Float {
        val dt = (nowMs - track.lastSeenMs).coerceIn(0L, 1400L) / 1000f
        val px = (centerX(track.box) + track.vx * dt).coerceIn(0f, 1f)
        val py = (centerY(track.box) + track.vy * dt).coerceIn(0f, 1f)
        val spatial = hypot(
            (centerX(obs.box) - px).toDouble(),
            (centerY(obs.box) - py).toDouble()
        ).toFloat()

        val age = nowMs - track.lastSeenMs
        val gate = when {
            age <= 500L -> 0.23f
            age <= 1100L -> 0.34f
            else -> 0.46f
        }
        if (spatial > gate) return Float.POSITIVE_INFINITY

        val oldArea = max(area(track.box), 0.0005f)
        val newArea = max(area(obs.box), 0.0005f)
        val scalePenalty = abs(ln((newArea / oldArea).toDouble())).toFloat() * 0.055f

        val colorPenalty = if (track.color != null && obs.color != null) {
            colorDistance(track.color!!, obs.color!!) * 0.82f
        } else {
            0f
        }

        val overlapBonus = iou(track.box, obs.box) * 0.08f
        return spatial + scalePenalty + colorPenalty - overlapBonus
    }

    private fun updateTrack(track: Track, obs: Observation, nowMs: Long) {
        val dt = (nowMs - track.lastSeenMs).coerceAtLeast(1L) / 1000f
        val measuredVx =
            ((centerX(obs.box) - centerX(track.box)) / dt).coerceIn(-1.4f, 1.4f)
        val measuredVy =
            ((centerY(obs.box) - centerY(track.box)) / dt).coerceIn(-1.4f, 1.4f)

        track.vx = track.vx * 0.68f + measuredVx * 0.32f
        track.vy = track.vy * 0.68f + measuredVy * 0.32f
        track.box = RectF(obs.box)
        track.lastSeenMs = nowMs

        if (obs.color != null) {
            val old = track.color
            track.color = if (old == null) {
                obs.color.copyOf()
            } else {
                floatArrayOf(
                    old[0] * 0.90f + obs.color[0] * 0.10f,
                    old[1] * 0.90f + obs.color[1] * 0.10f,
                    old[2] * 0.90f + obs.color[2] * 0.10f
                )
            }
        }
    }

    private fun predictedBox(track: Track?, nowMs: Long): RectF? {
        track ?: return null
        val age = nowMs - track.lastSeenMs
        if (age > 2200L) return null

        val dt = age.coerceIn(0L, 900L) / 1000f
        val dx = track.vx * dt
        val dy = track.vy * dt

        return RectF(
            (track.box.left + dx).coerceIn(0f, 1f),
            (track.box.top + dy).coerceIn(0f, 1f),
            (track.box.right + dx).coerceIn(0f, 1f),
            (track.box.bottom + dy).coerceIn(0f, 1f)
        )
    }

    private fun colorDistance(a: FloatArray, b: FloatArray): Float {
        val dr = a[0] - b[0]
        val dg = a[1] - b[1]
        val db = a[2] - b[2]
        return sqrt(dr * dr + dg * dg + db * db)
    }

    private fun iou(a: RectF, b: RectF): Float {
        val left = max(a.left, b.left)
        val top = max(a.top, b.top)
        val right = min(a.right, b.right)
        val bottom = min(a.bottom, b.bottom)
        val inter = max(0f, right - left) * max(0f, bottom - top)
        val union = area(a) + area(b) - inter
        return if (union <= 0f) 0f else inter / union
    }

    private fun normalize(box: RectF, width: Int, height: Int): RectF =
        RectF(
            (box.left / width).coerceIn(0f, 1f),
            (box.top / height).coerceIn(0f, 1f),
            (box.right / width).coerceIn(0f, 1f),
            (box.bottom / height).coerceIn(0f, 1f)
        )

    private fun area(r: RectF) = r.width() * r.height()
    private fun centerX(r: RectF) = (r.left + r.right) / 2f
    private fun centerY(r: RectF) = (r.top + r.bottom) / 2f
}
