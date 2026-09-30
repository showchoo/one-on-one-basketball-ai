package jp.showchoo.oneononeai

import android.graphics.PointF
import android.graphics.RectF
import kotlin.math.abs
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

    private data class BallSample(
        val x: Float,
        val y: Float,
        val timeMs: Long,
        val score: Float
    )

    private val ballHistory = ArrayDeque<BallSample>()
    private var lastBallBox: RectF? = null
    private var lastBallSeenMs = 0L

    private var releaseCandidatePlayer: Char? = null
    private var releaseCandidateValue = 1
    private var releaseCandidateAtMs = 0L
    private var wasBallNearPlayer = false

    private enum class ShotState {
        IDLE,
        ABOVE_RIM,
        BELOW_RIM
    }

    private var shotState = ShotState.IDLE
    private var shotStartedMs = 0L
    private var ballBelowMs = 0L
    private var savedHoop: RectF? = null
    private var shotPlayer: Char? = null
    private var shotValue = 1
    private var cooldownUntilMs = 0L

    val needsFastBallTracking: Boolean
        get() = shotState != ShotState.IDLE

    fun resetSession() {
        playerA = null
        playerB = null
        lastPossessor = null
        ballHistory.clear()
        lastBallBox = null
        lastBallSeenMs = 0L
        releaseCandidatePlayer = null
        releaseCandidateValue = 1
        releaseCandidateAtMs = 0L
        wasBallNearPlayer = false
        shotState = ShotState.IDLE
        shotStartedMs = 0L
        ballBelowMs = 0L
        savedHoop = null
        shotPlayer = null
        shotValue = 1
        cooldownUntilMs = 0L
        onDebugEvent("TRACKER_RESET_V030")
    }

    fun update(
        detections: List<AiDetection>,
        width: Int,
        height: Int,
        nowMs: Long
    ): TrackerSnapshot {
        if (width <= 0 || height <= 0) {
            return snapshot(null, "画像待機")
        }

        val people = detections
            .filter { it.label == "person" }
            .map { normalize(it.box, width, height) }
            .sortedByDescending { it.width() * it.height() }
            .take(2)
        updatePlayers(people, nowMs)

        updateHoopFromDetector(detections, width, height)

        val ballCandidates = detections
            .filter { it.label == "sports ball" }
            .map { detection ->
                detection to normalize(detection.box, width, height)
            }

        val selected = selectBall(ballCandidates, nowMs)
        val ball = selected?.second

        if (selected != null) {
            val detection = selected.first
            val detectedBall = selected.second
            val bx = centerX(detectedBall)
            val by = centerY(detectedBall)

            ballHistory.addLast(
                BallSample(
                    x = bx,
                    y = by,
                    timeMs = nowMs,
                    score = detection.score
                )
            )
            while (ballHistory.size > 120) ballHistory.removeFirst()
            while (ballHistory.isNotEmpty() && nowMs - ballHistory.first().timeMs > 8000L) {
                ballHistory.removeFirst()
            }

            lastBallBox = detectedBall
            lastBallSeenMs = nowMs

            updatePossessionAndReleaseCandidate(detectedBall, nowMs)
        } else if (nowMs - lastBallSeenMs > 900L) {
            lastBallBox = null
        }

        val status = updateShotState(ball, nowMs)
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
            a.box = p0
            b.box = p1
        } else {
            a.box = p1
            b.box = p0
        }

        a.lastSeenMs = nowMs
        b.lastSeenMs = nowMs
    }

    private fun updateHoopFromDetector(
        detections: List<AiDetection>,
        width: Int,
        height: Int
    ) {
        val detected = detections
            .filter { it.label == "hoop" }
            .maxByOrNull { it.score }
            ?.let { normalize(it.box, width, height) }
            ?: return

        val current = hoopRect
        if (current == null) {
            hoopRect = detected
            onDebugEvent("HOOP_AUTO_ACQUIRED")
            return
        }

        val centerDistance = hypot(
            (centerX(current) - centerX(detected)).toDouble(),
            (centerY(current) - centerY(detected)).toDouble()
        ).toFloat()

        if (centerDistance <= 0.15f) {
            val alpha = 0.20f
            hoopRect = RectF(
                lerp(current.left, detected.left, alpha),
                lerp(current.top, detected.top, alpha),
                lerp(current.right, detected.right, alpha),
                lerp(current.bottom, detected.bottom, alpha)
            )
        }
    }

    private fun selectBall(
        candidates: List<Pair<AiDetection, RectF>>,
        nowMs: Long
    ): Pair<AiDetection, RectF>? {
        if (candidates.isEmpty()) return null
        if (candidates.size == 1) return candidates.first()

        val hoop = savedHoop ?: hoopRect
        if (shotState != ShotState.IDLE && hoop != null) {
            val hcX = centerX(hoop)
            val hcY = centerY(hoop)
            return candidates.maxByOrNull { pair ->
                val box = pair.second
                val dist = hypot(
                    (centerX(box) - hcX).toDouble(),
                    (centerY(box) - hcY).toDouble()
                ).toFloat()
                pair.first.score + (0.35f - dist).coerceAtLeast(0f)
            }
        }

        val previous = lastBallBox
        if (previous != null && nowMs - lastBallSeenMs <= 700L) {
            val px = centerX(previous)
            val py = centerY(previous)
            val continuous = candidates
                .map { pair ->
                    val d = hypot(
                        (centerX(pair.second) - px).toDouble(),
                        (centerY(pair.second) - py).toDouble()
                    ).toFloat()
                    pair to d
                }
                .filter { it.second <= 0.28f }

            if (continuous.isNotEmpty()) {
                return continuous.minByOrNull { it.second - it.first.first.score * 0.10f }?.first
            }
        }

        return candidates.maxByOrNull { it.first.score }
    }

    private fun updatePossessionAndReleaseCandidate(ball: RectF, nowMs: Long) {
        val possessor = nearestPossessor(ball)
        val near = possessor != null

        if (near) {
            lastPossessor = possessor
        }

        if (wasBallNearPlayer && !near && lastPossessor != null) {
            val recent = ballHistory.takeLast(4)
            val movingUp = recent.size >= 2 &&
                recent.last().y < recent.first().y - 0.006f

            if (movingUp) {
                releaseCandidatePlayer = lastPossessor
                releaseCandidateValue = calculateShotValue(lastPossessor!!)
                releaseCandidateAtMs = nowMs
                onDebugEvent(
                    "RELEASE_CANDIDATE player=$releaseCandidatePlayer value=$releaseCandidateValue"
                )
            }
        }

        wasBallNearPlayer = near
    }

    private fun updateShotState(ball: RectF?, nowMs: Long): String {
        if (nowMs < cooldownUntilMs) {
            return "COOLDOWN"
        }

        val hoop = savedHoop ?: hoopRect ?: return "リング待機"

        when (shotState) {
            ShotState.IDLE -> {
                if (ball == null) {
                    return if (playerA != null && playerB != null) "A/B追跡中" else "2人を認識中"
                }

                if (isBallAboveHoopAndRising(ball, hoop)) {
                    savedHoop = RectF(hoop)
                    shotStartedMs = nowMs
                    ballBelowMs = 0L

                    val recentReleaseValid =
                        releaseCandidatePlayer != null &&
                            nowMs - releaseCandidateAtMs <= 2500L

                    shotPlayer = if (recentReleaseValid) {
                        releaseCandidatePlayer
                    } else {
                        lastPossessor ?: closestPlayerToBall(centerX(ball), centerY(ball))
                    }

                    shotValue = if (recentReleaseValid) {
                        releaseCandidateValue
                    } else {
                        shotPlayer?.let { calculateShotValue(it) } ?: 1
                    }

                    shotState = ShotState.ABOVE_RIM
                    onDebugEvent(
                        "SHOT_ABOVE_RIM player=$shotPlayer value=$shotValue ball=" +
                            centerX(ball) + "|" + centerY(ball)
                    )
                    return "SHOT / ABOVE RIM"
                }

                return if (playerA != null && playerB != null) "A/B追跡中" else "2人を認識中"
            }

            ShotState.ABOVE_RIM -> {
                if (nowMs - shotStartedMs > 3000L) {
                    resetShot("SHOT_TIMEOUT_ABOVE")
                    return "SHOT TIMEOUT"
                }

                val saved = savedHoop ?: hoop
                if (ball != null && centerY(ball) > centerY(saved)) {
                    shotState = ShotState.BELOW_RIM
                    ballBelowMs = nowMs
                    onDebugEvent(
                        "SHOT_BELOW_RIM ball=" + centerX(ball) + "|" + centerY(ball)
                    )
                    return "SHOT / BELOW RIM"
                }

                if (ball == null && nowMs - lastBallSeenMs >= 450L) {
                    val last = ballHistory.lastOrNull()
                    if (last != null && isNearRim(last.x, last.y, saved)) {
                        shotState = ShotState.BELOW_RIM
                        ballBelowMs = nowMs
                        onDebugEvent("SHOT_BALL_VANISHED_NEAR_RIM")
                        return "SHOT / RIM OCCLUSION"
                    }
                }

                return "SHOT / ABOVE RIM"
            }

            ShotState.BELOW_RIM -> {
                if (nowMs - ballBelowMs < 350L) {
                    return "SHOT / VERIFYING"
                }

                val made = classifyMake(nowMs)
                if (made) {
                    val scorer = shotPlayer
                    if (scorer != null) {
                        onDebugEvent(
                            "AUTO_SCORE_V030 player=$scorer points=$shotValue"
                        )
                        onAutomaticScore(scorer, shotValue)
                    } else {
                        onDebugEvent("MAKE_NO_SHOOTER")
                    }
                    resetShot("SHOT_MAKE")
                    cooldownUntilMs = nowMs + 1200L
                    return "MAKE"
                }

                resetShot("SHOT_MISS")
                cooldownUntilMs = nowMs + 650L
                return "MISS"
            }
        }
    }

    private fun isBallAboveHoopAndRising(ball: RectF, hoop: RectF): Boolean {
        val bx = centerX(ball)
        val by = centerY(ball)
        val hx = centerX(hoop)
        val hy = centerY(hoop)

        val inX = bx > hx - hoop.width() * 4.0f &&
            bx < hx + hoop.width() * 4.0f
        val inY = by > hy - hoop.height() * 2.5f &&
            by < hy

        if (!inX || !inY) return false

        val recent = ballHistory.takeLast(4)
        if (recent.size < 3) return false

        var upwardSteps = 0
        for (i in 1 until recent.size) {
            if (recent[i].y < recent[i - 1].y) upwardSteps++
        }

        val netUp = recent.last().y < recent.first().y - 0.005f
        return upwardSteps >= 1 && netUp
    }

    private fun classifyMake(nowMs: Long): Boolean {
        val hoop = savedHoop ?: return false
        val hx = centerX(hoop)
        val hy = centerY(hoop)
        val hw = hoop.width()
        val hh = hoop.height()

        val shotSamples = ballHistory.filter {
            it.timeMs >= shotStartedMs - 250L &&
                it.timeMs <= nowMs
        }
        if (shotSamples.isEmpty()) return false

        val postRim = shotSamples.filter {
            it.timeMs >= ballBelowMs &&
                it.timeMs <= ballBelowMs + 900L &&
                it.y >= hy
        }

        var trajectoryMake = false
        if (postRim.size >= 2) {
            val maxLateral = postRim.maxOf { abs(it.x - hx) }
            val avgX = postRim.map { it.x }.average().toFloat()
            val movedDown = postRim.last().y >= postRim.first().y
            val stayedNarrow = maxLateral < hw * 0.85f
            val centered = abs(avgX - hx) < hw * 0.70f
            trajectoryMake = stayedNarrow && movedDown && centered
        }

        val lastNearAbove = shotSamples.lastOrNull {
            it.y > hy - hh * 1.8f &&
                it.y < hy + hh * 0.10f &&
                abs(it.x - hx) < hw * 1.5f
        }

        val firstBelow = shotSamples.firstOrNull {
            it.timeMs >= shotStartedMs &&
                it.y > hy + hh * 0.35f
        }

        var disappearanceMake = false
        if (lastNearAbove != null && firstBelow != null) {
            val gapMs = firstBelow.timeMs - lastNearAbove.timeMs
            val centeredBelow = abs(firstBelow.x - hx) < hw * 0.75f
            disappearanceMake = gapMs >= 150L && centeredBelow
        }

        var interpolationMake = false
        val ordered = shotSamples.sortedBy { it.timeMs }
        for (i in 1 until ordered.size) {
            val a = ordered[i - 1]
            val b = ordered[i]
            if (a.y < hy && b.y > hy && b.y > a.y) {
                val dy = b.y - a.y
                if (dy > 0.0001f) {
                    val t = ((hy - a.y) / dy).coerceIn(0f, 1f)
                    val crossingX = a.x + (b.x - a.x) * t
                    if (abs(crossingX - hx) < hw * 0.65f) {
                        interpolationMake = true
                        break
                    }
                }
            }
        }

        val vanishedNearRim =
            postRim.isEmpty() &&
                nowMs - lastBallSeenMs >= 350L &&
                shotSamples.lastOrNull()?.let {
                    abs(it.x - hx) < hw * 0.8f &&
                        abs(it.y - hy) < hh * 1.8f
                } == true

        onDebugEvent(
            "MAKE_CLASSIFY trajectory=$trajectoryMake disappearance=$disappearanceMake " +
                "interpolation=$interpolationMake vanish=$vanishedNearRim post=" + postRim.size
        )

        return trajectoryMake || disappearanceMake || interpolationMake || vanishedNearRim
    }

    private fun resetShot(reason: String) {
        onDebugEvent(reason)
        shotState = ShotState.IDLE
        shotStartedMs = 0L
        ballBelowMs = 0L
        savedHoop = null
        shotPlayer = null
        shotValue = 1
        releaseCandidatePlayer = null
        releaseCandidateAtMs = 0L
    }

    private fun isNearRim(x: Float, y: Float, hoop: RectF): Boolean {
        val hx = centerX(hoop)
        val hy = centerY(hoop)
        return abs(x - hx) < hoop.width() * 1.2f &&
            abs(y - hy) < hoop.height() * 2.0f
    }

    private fun nearestPossessor(ball: RectF): Char? {
        val bx = centerX(ball)
        val by = centerY(ball)
        val candidates = listOfNotNull(playerA, playerB)

        return candidates
            .mapNotNull { player ->
                val expanded = RectF(
                    player.box.left - player.box.width() * 0.40f,
                    player.box.top - player.box.height() * 0.20f,
                    player.box.right + player.box.width() * 0.40f,
                    player.box.bottom + player.box.height() * 0.12f
                )
                if (!expanded.contains(bx, by)) return@mapNotNull null

                val d = hypot(
                    (bx - player.centerX).toDouble(),
                    (by - player.centerY).toDouble()
                ).toFloat()
                player.id to d
            }
            .minByOrNull { it.second }
            ?.first
    }

    private fun closestPlayerToBall(ballX: Float, ballY: Float): Char? =
        listOfNotNull(playerA, playerB)
            .minByOrNull { player ->
                hypot(
                    (ballX - player.centerX).toDouble(),
                    (ballY - player.centerY).toDouble()
                )
            }
            ?.id

    private fun calculateShotValue(player: Char): Int {
        if (threePointLine.size < 2) return 1
        val track = if (player == 'A') playerA else playerB
        track ?: return 1

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

    private fun normalize(box: RectF, width: Int, height: Int): RectF =
        RectF(
            (box.left / width).coerceIn(0f, 1f),
            (box.top / height).coerceIn(0f, 1f),
            (box.right / width).coerceIn(0f, 1f),
            (box.bottom / height).coerceIn(0f, 1f)
        )

    private fun centerX(r: RectF) = (r.left + r.right) / 2f
    private fun centerY(r: RectF) = (r.top + r.bottom) / 2f

    private fun distance(a: RectF, b: RectF): Float =
        hypot(
            (centerX(a) - centerX(b)).toDouble(),
            (centerY(a) - centerY(b)).toDouble()
        ).toFloat()

    private fun lerp(a: Float, b: Float, t: Float): Float = a + (b - a) * t

    private fun snapshot(ball: RectF?, status: String) =
        TrackerSnapshot(
            playerA = playerA?.box,
            playerB = playerB?.box,
            ball = ball,
            lastPossessor = lastPossessor,
            shotPlayer = shotPlayer,
            shotValue = shotValue,
            status = status
        )
}
