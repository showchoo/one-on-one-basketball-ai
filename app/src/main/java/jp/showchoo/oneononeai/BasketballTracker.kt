package jp.showchoo.oneononeai

import android.graphics.PointF
import android.graphics.RectF
import kotlin.math.hypot

class BasketballTracker(
    private val onAutomaticScore: (Char, Int) -> Unit,
    private val onDebugEvent: (String) -> Unit = {}
) {
    var hoopRect: RectF? = null
    var threePointLine: List<PointF> = emptyList()

    private var playerA: PlayerTrack? = null
    private var playerB: PlayerTrack? = null
    private var lastPossessor: Char? = null
    private var wasBallNearPlayer = false
    private var currentShotPlayer: Char? = null
    private var currentShotValue = 1
    private var shotStartedMs = 0L

    private enum class HoopState { WAIT_ABOVE, ARMED }
    private var hoopState = HoopState.WAIT_ABOVE
    private var armedAtMs = 0L
    private var lastScoreMs = 0L

    fun resetSession() {
        playerA = null
        playerB = null
        lastPossessor = null
        wasBallNearPlayer = false
        currentShotPlayer = null
        currentShotValue = 1
        hoopState = HoopState.WAIT_ABOVE
        armedAtMs = 0L
        onDebugEvent("TRACKER_RESET")
    }

    fun update(detections: List<AiDetection>, width: Int, height: Int, nowMs: Long): TrackerSnapshot {
        if (width <= 0 || height <= 0) return snapshot(null, "画像待機")

        val people = detections.filter { it.label == "person" }
            .map { normalize(it.box, width, height) }
            .sortedByDescending { it.width() * it.height() }
            .take(2)
        updatePlayers(people, nowMs)

        val ball = detections.filter { it.label == "sports ball" }
            .maxByOrNull { it.score }
            ?.let { normalize(it.box, width, height) }

        var status = if (playerA != null && playerB != null) "A/B追跡中" else "2人を認識中"

        if (ball != null) {
            val possessor = nearestPossessor(ball)
            val near = possessor != null
            if (near) lastPossessor = possessor

            if (wasBallNearPlayer && !near && lastPossessor != null) {
                currentShotPlayer = lastPossessor
                currentShotValue = calculateShotValue(lastPossessor!!)
                shotStartedMs = nowMs
                status = "${currentShotPlayer} シュート候補 ${currentShotValue}点"
                onDebugEvent("SHOT_CANDIDATE player=${currentShotPlayer} value=${currentShotValue}")
            }
            wasBallNearPlayer = near

            detectHoopCrossing(ball, nowMs)
        } else {
            if (nowMs - shotStartedMs > 2500L) currentShotPlayer = null
        }

        return snapshot(ball, status)
    }

    private fun updatePlayers(people: List<RectF>, nowMs: Long) {
        if (people.size < 2) return
        if (playerA == null || playerB == null) {
            val sorted = people.sortedBy { centerX(it) }
            playerA = PlayerTrack('A', sorted[0], nowMs)
            playerB = PlayerTrack('B', sorted[1], nowMs)
            return
        }

        val a = playerA!!
        val b = playerB!!
        val p0 = people[0]
        val p1 = people[1]
        val direct = distance(a.box, p0) + distance(b.box, p1)
        val crossed = distance(a.box, p1) + distance(b.box, p0)
        if (direct <= crossed) {
            a.box = p0; b.box = p1
        } else {
            a.box = p1; b.box = p0
        }
        a.lastSeenMs = nowMs
        b.lastSeenMs = nowMs
    }

    private fun nearestPossessor(ball: RectF): Char? {
        val bx = centerX(ball)
        val by = centerY(ball)
        val candidates = listOfNotNull(playerA, playerB)
        var best: PlayerTrack? = null
        var bestScore = Float.MAX_VALUE
        for (p in candidates) {
            val expanded = RectF(
                p.box.left - p.box.width() * 0.30f,
                p.box.top - p.box.height() * 0.20f,
                p.box.right + p.box.width() * 0.30f,
                p.box.bottom + p.box.height() * 0.10f
            )
            if (!expanded.contains(bx, by)) continue
            val d = hypot((bx - p.centerX).toDouble(), (by - p.centerY).toDouble()).toFloat()
            if (d < bestScore) { best = p; bestScore = d }
        }
        return best?.id
    }

    private fun detectHoopCrossing(ball: RectF, nowMs: Long) {
        val hoop = hoopRect ?: return
        val x = centerX(ball)
        val y = centerY(ball)
        val xMargin = hoop.width() * 0.55f
        val inLaneX = x >= hoop.left - xMargin && x <= hoop.right + xMargin

        when (hoopState) {
            HoopState.WAIT_ABOVE -> {
                if (inLaneX && y < hoop.top) {
                    hoopState = HoopState.ARMED
                    armedAtMs = nowMs
                    onDebugEvent("HOOP_ARMED ballX=$x ballY=$y")
                }
            }
            HoopState.ARMED -> {
                if (nowMs - armedAtMs > 1600L) {
                    hoopState = HoopState.WAIT_ABOVE
                    onDebugEvent("HOOP_TIMEOUT")
                    return
                }
                if (inLaneX && y > hoop.bottom) {
                    val shooter = currentShotPlayer ?: lastPossessor
                    if (shooter != null && nowMs - lastScoreMs > 1800L) {
                        lastScoreMs = nowMs
                        onDebugEvent("AUTO_SCORE player=$shooter points=$currentShotValue")
                        onAutomaticScore(shooter, currentShotValue)
                    }
                    currentShotPlayer = null
                    hoopState = HoopState.WAIT_ABOVE
                }
            }
        }
    }

    private fun calculateShotValue(player: Char): Int {
        if (threePointLine.size < 2) return 1
        val track = (if (player == 'A') playerA else playerB) ?: return 1
        val footX = track.footX
        val footY = track.footY
        val lineY = interpolateLineY(footX) ?: return 1
        return if (footY > lineY + 0.012f) 2 else 1
    }

    private fun interpolateLineY(x: Float): Float? {
        val pts = threePointLine.sortedBy { it.x }
        if (pts.size < 2) return null
        if (x <= pts.first().x) return pts.first().y
        if (x >= pts.last().x) return pts.last().y
        for (i in 0 until pts.lastIndex) {
            val a = pts[i]
            val b = pts[i + 1]
            if (x in a.x..b.x) {
                val t = (x - a.x) / (b.x - a.x).coerceAtLeast(0.0001f)
                return a.y + (b.y - a.y) * t
            }
        }
        return null
    }

    private fun normalize(box: RectF, width: Int, height: Int): RectF = RectF(
        (box.left / width).coerceIn(0f, 1f),
        (box.top / height).coerceIn(0f, 1f),
        (box.right / width).coerceIn(0f, 1f),
        (box.bottom / height).coerceIn(0f, 1f)
    )

    private fun centerX(r: RectF) = (r.left + r.right) / 2f
    private fun centerY(r: RectF) = (r.top + r.bottom) / 2f
    private fun distance(a: RectF, b: RectF): Float = hypot(
        (centerX(a) - centerX(b)).toDouble(),
        (centerY(a) - centerY(b)).toDouble()
    ).toFloat()

    private fun snapshot(ball: RectF?, status: String) = TrackerSnapshot(
        playerA = playerA?.box,
        playerB = playerB?.box,
        ball = ball,
        lastPossessor = lastPossessor,
        shotPlayer = currentShotPlayer,
        shotValue = currentShotValue,
        status = status
    )
}
