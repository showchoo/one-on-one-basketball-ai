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

    private data class BallSample(
        val x: Float,
        val y: Float,
        val timeMs: Long,
        val source: String = "FULL",
        val score: Float = 0f
    )
    private val ballHistory = mutableListOf<BallSample>()
    private var lastSelectedBall: RectF? = null
    private var lastSelectedBallMs = 0L

    private enum class HoopState { WAIT_ABOVE, ARMED }
    private var hoopState = HoopState.WAIT_ABOVE
    private var armedAtMs = 0L
    private var armedBallY = 0f
    private var lastScoreMs = 0L

    fun resetSession() {
        playerA = null
        playerB = null
        lastPossessor = null
        wasBallNearPlayer = false
        currentShotPlayer = null
        currentShotValue = 1
        shotStartedMs = 0L
        hoopState = HoopState.WAIT_ABOVE
        armedAtMs = 0L
        armedBallY = 0f
        lastScoreMs = 0L
        ballHistory.clear()
        lastSelectedBall = null
        lastSelectedBallMs = 0L
        onDebugEvent("TRACKER_RESET")
    }

    fun update(detections: List<AiDetection>, width: Int, height: Int, nowMs: Long): TrackerSnapshot {
        if (width <= 0 || height <= 0) return snapshot(null, "画像待機")

        val people = detections.filter { it.label == "person" }
            .map { normalize(it.box, width, height) }
            .sortedByDescending { it.width() * it.height() }
            .take(2)
        updatePlayers(people, nowMs)

        val ballDetections = detections
            .filter { it.label == "sports ball" }

        val selectedBallDetection = selectBallDetection(ballDetections, width, height, nowMs)
        val ball = selectedBallDetection?.let { normalize(it.box, width, height) }

        var status = if (playerA != null && playerB != null) "A/B追跡中" else "2人を認識中"

        if (ball != null) {
            val bx = centerX(ball)
            val by = centerY(ball)
            addBallSample(
                bx,
                by,
                nowMs,
                selectedBallDetection?.source ?: "FULL",
                selectedBallDetection?.score ?: 0f
            )
            lastSelectedBall = ball
            lastSelectedBallMs = nowMs

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
            pruneBallHistory(nowMs)
            if (nowMs - lastSelectedBallMs > 900L) lastSelectedBall = null
            if (nowMs - shotStartedMs > 4500L) currentShotPlayer = null
        }

        return snapshot(ball, status)
    }

    private fun selectBallDetection(
        detections: List<AiDetection>,
        width: Int,
        height: Int,
        nowMs: Long
    ): AiDetection? {
        if (detections.isEmpty()) return null

        val hoop = hoopRect
        if (hoop != null) {
            val scoringCandidates = detections.filter { det ->
                val box = normalize(det.box, width, height)
                val x = centerX(box)
                val y = centerY(box)
                val w = hoop.width()
                val h = hoop.height()
                x >= hoop.left - w * 1.25f &&
                    x <= hoop.right + w * 1.25f &&
                    y >= hoop.top - h * 4.5f &&
                    y <= hoop.bottom + h * 5.5f
            }

            if (scoringCandidates.isNotEmpty()) {
                val hcX = centerX(hoop)
                val hcY = centerY(hoop)
                return scoringCandidates.maxByOrNull { det ->
                    val box = normalize(det.box, width, height)
                    val dist = hypot(
                        (centerX(box) - hcX).toDouble(),
                        (centerY(box) - hcY).toDouble()
                    ).toFloat()
                    val sourceBonus = if (det.source == "HOOP_ROI") 0.45f else 0f
                    sourceBonus + det.score + (0.25f - dist).coerceAtLeast(0f)
                }
            }
        }

        val previous = lastSelectedBall
        if (previous != null && nowMs - lastSelectedBallMs <= 800L) {
            val px = centerX(previous)
            val py = centerY(previous)
            val continuity = detections.map { det ->
                val box = normalize(det.box, width, height)
                val d = hypot(
                    (centerX(box) - px).toDouble(),
                    (centerY(box) - py).toDouble()
                ).toFloat()
                det to d
            }.filter { it.second <= 0.32f }

            if (continuity.isNotEmpty()) {
                return continuity.minByOrNull { pair ->
                    val sourceBonus = if (pair.first.source == "HOOP_ROI") -0.04f else 0f
                    pair.second + sourceBonus - pair.first.score * 0.12f
                }?.first
            }
        }

        return detections.maxByOrNull { det ->
            det.score + if (det.source == "HOOP_ROI") 0.08f else 0f
        }
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
            a.box = p0
            b.box = p1
        } else {
            a.box = p1
            b.box = p0
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
            // 手元の小さいボールを取りこぼさないよう、人物領域を広めに見る。
            val expanded = RectF(
                p.box.left - p.box.width() * 0.45f,
                p.box.top - p.box.height() * 0.25f,
                p.box.right + p.box.width() * 0.45f,
                p.box.bottom + p.box.height() * 0.15f
            )
            if (!expanded.contains(bx, by)) continue

            val d = hypot((bx - p.centerX).toDouble(), (by - p.centerY).toDouble()).toFloat()
            if (d < bestScore) {
                best = p
                bestScore = d
            }
        }
        return best?.id
    }

    private fun closestPlayerToBall(ballX: Float, ballY: Float): Char? {
        return listOfNotNull(playerA, playerB)
            .minByOrNull { player ->
                hypot(
                    (ballX - player.centerX).toDouble(),
                    (ballY - player.centerY).toDouble()
                )
            }
            ?.id
    }

    private fun addBallSample(
        x: Float,
        y: Float,
        nowMs: Long,
        source: String,
        score: Float
    ) {
        ballHistory += BallSample(x, y, nowMs, source, score)
        pruneBallHistory(nowMs)
        while (ballHistory.size > 24) ballHistory.removeAt(0)
    }

    private fun pruneBallHistory(nowMs: Long) {
        ballHistory.removeAll { nowMs - it.timeMs > 3000L }
    }

    private fun detectHoopCrossing(ball: RectF, nowMs: Long) {
        val hoop = hoopRect ?: return
        val x = centerX(ball)
        val y = centerY(ball)

        val hoopCenterY = centerY(hoop)
        val xMargin = hoop.width() * 0.80f
        val laneLeft = hoop.left - xMargin
        val laneRight = hoop.right + xMargin
        val inLaneX = x in laneLeft..laneRight

        // 手動設定したリング矩形は実リングより大きめなので、
        // 中心線を基準に「上→下」の軌道を判定する。
        val aboveThreshold = hoopCenterY + maxOf(hoop.height() * 0.05f, 0.004f)
        val belowThreshold = hoopCenterY + maxOf(hoop.height() * 0.18f, 0.008f)
        val approachTop = hoop.top - maxOf(hoop.height() * 4.0f, 0.065f)

        if (hoopState == HoopState.WAIT_ABOVE && inferDirectRimCrossing(hoop, nowMs)) {
            if (currentShotPlayer == null && lastPossessor == null) {
                val fallback = closestPlayerToBall(x, y)
                if (fallback != null) {
                    currentShotPlayer = fallback
                    currentShotValue = calculateShotValue(fallback)
                    shotStartedMs = nowMs
                    onDebugEvent(
                        "SHOOTER_FALLBACK player=$fallback value=$currentShotValue ballX=$x ballY=$y"
                    )
                }
            }
            onDebugEvent("HOOP_CROSS_INFERRED ballX=$x ballY=$y")
            registerAutomaticScore(nowMs, x, y)
            return
        }

        when (hoopState) {
            HoopState.WAIT_ABOVE -> {
                if (inLaneX && y <= aboveThreshold && y >= approachTop) {
                    if (currentShotPlayer == null && lastPossessor == null) {
                        val fallback = closestPlayerToBall(x, y)
                        if (fallback != null) {
                            currentShotPlayer = fallback
                            currentShotValue = calculateShotValue(fallback)
                            shotStartedMs = nowMs
                            onDebugEvent(
                                "SHOOTER_FALLBACK player=$fallback value=$currentShotValue ballX=$x ballY=$y"
                            )
                        }
                    }

                    hoopState = HoopState.ARMED
                    armedAtMs = nowMs
                    armedBallY = y
                    onDebugEvent(
                        "HOOP_ARMED ballX=$x ballY=$y above=$aboveThreshold below=$belowThreshold"
                    )
                }
            }

            HoopState.ARMED -> {
                if (nowMs - armedAtMs > 2600L) {
                    hoopState = HoopState.WAIT_ABOVE
                    onDebugEvent("HOOP_TIMEOUT")
                    return
                }

                val minimumDrop = maxOf(hoop.height() * 0.18f, 0.008f)
                val crossedDownward =
                    inLaneX &&
                        y >= belowThreshold &&
                        y - armedBallY >= minimumDrop

                if (crossedDownward) {
                    val hadAboveSample = ballHistory.any { sample ->
                        sample.timeMs >= armedAtMs - 350L &&
                            sample.timeMs <= nowMs &&
                            sample.x in laneLeft..laneRight &&
                            sample.y <= aboveThreshold
                    }

                    if (hadAboveSample) {
                        registerAutomaticScore(nowMs, x, y)
                    }
                }
            }
        }
    }

    private fun inferDirectRimCrossing(hoop: RectF, nowMs: Long): Boolean {
        if (ballHistory.size < 2) return false
        val current = ballHistory.last()
        val previous = ballHistory[ballHistory.lastIndex - 1]
        val dt = current.timeMs - previous.timeMs
        if (dt !in 1L..500L) return false

        val hoopCenterY = centerY(hoop)
        val belowThreshold = hoopCenterY + maxOf(hoop.height() * 0.18f, 0.008f)
        if (previous.y >= hoopCenterY || current.y < belowThreshold) return false
        if (current.y <= previous.y) return false

        if (previous.source != "HOOP_ROI" && current.source != "HOOP_ROI") return false

        val dy = current.y - previous.y
        if (dy <= 0.0001f) return false
        val t = ((hoopCenterY - previous.y) / dy).coerceIn(0f, 1f)
        val crossingX = previous.x + (current.x - previous.x) * t
        val tightMargin = hoop.width() * 0.35f

        val crossesMouth =
            crossingX >= hoop.left - tightMargin &&
                crossingX <= hoop.right + tightMargin

        if (crossesMouth) {
            onDebugEvent(
                "RIM_SEGMENT prev=${previous.x}|${previous.y} current=${current.x}|${current.y} crossingX=$crossingX dt=$dt"
            )
        }
        return crossesMouth
    }

    private fun registerAutomaticScore(nowMs: Long, ballX: Float, ballY: Float) {
        val shooter = currentShotPlayer ?: lastPossessor

        if (shooter == null) {
            onDebugEvent("HOOP_CROSSED_NO_SHOOTER ballX=$ballX ballY=$ballY")
            hoopState = HoopState.WAIT_ABOVE
            return
        }

        if (nowMs - lastScoreMs <= 1200L) {
            onDebugEvent("HOOP_CROSSED_COOLDOWN player=$shooter")
            hoopState = HoopState.WAIT_ABOVE
            return
        }

        lastScoreMs = nowMs
        onDebugEvent(
            "AUTO_SCORE player=$shooter points=$currentShotValue ballX=$ballX ballY=$ballY"
        )
        onAutomaticScore(shooter, currentShotValue)

        currentShotPlayer = null
        hoopState = HoopState.WAIT_ABOVE
        ballHistory.clear()
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
