package jp.showchoo.oneononeai

import android.graphics.PointF
import android.graphics.RectF
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.max

/**
 * v0.4 scoring state machine.
 *
 * Receives persistent A/B identities from PlayerIdentityTracker and a
 * high-frequency ball stream from FastBallTracker. YOLO is no longer the
 * frame-by-frame scoring clock.
 */
class BasketballTracker(
    private val onAutomaticScore: (Char, Int) -> Unit,
    private val onDebugEvent: (String) -> Unit = {}
) {
    var hoopRect: RectF? = null
        private set
    var threePointLine: List<PointF> = emptyList()

    private var manualHoopLocked = false
    private var lastPossessor: Char? = null
    private var releasePlayer: Char? = null
    private var releaseValue = 1
    private var releaseAtMs = 0L
    private var wasBallNearPlayer = false

    private enum class ShotState {
        IDLE,
        ABOVE_RIM,
        BELOW_RIM,
        COOLDOWN
    }

    private data class BallSample(
        val x: Float,
        val y: Float,
        val timeMs: Long,
        val confidence: Float,
        val source: String
    ) {
        val isMeasured: Boolean get() = source != "FAST_PREDICT"
    }

    private val history = ArrayDeque<BallSample>()
    private var shotState = ShotState.IDLE
    private var shotPlayer: Char? = null
    private var shotValue = 1
    private var shotStartedMs = 0L
    private var aboveRimMs = 0L
    private var belowRimMs = 0L
    private var cooldownUntilMs = 0L
    private var savedHoop: RectF? = null

    val needsFastBallTracking: Boolean
        get() = shotState == ShotState.ABOVE_RIM || shotState == ShotState.BELOW_RIM

    fun setManualHoop(rect: RectF) {
        hoopRect = RectF(rect)
        manualHoopLocked = true
        onDebugEvent("HOOP_MANUAL_LOCK")
    }

    fun resetSession() {
        lastPossessor = null
        releasePlayer = null
        releaseValue = 1
        releaseAtMs = 0L
        wasBallNearPlayer = false
        history.clear()
        shotState = ShotState.IDLE
        shotPlayer = null
        shotValue = 1
        shotStartedMs = 0L
        aboveRimMs = 0L
        belowRimMs = 0L
        cooldownUntilMs = 0L
        savedHoop = null
        onDebugEvent("TRACKER_RESET_V040")
    }

    fun updateYoloDetections(
        detections: List<AiDetection>,
        imageWidth: Int,
        imageHeight: Int,
        nowMs: Long
    ) {
        if (manualHoopLocked) return

        val detected = detections
            .filter { it.label == "hoop" }
            .maxByOrNull { it.score }
            ?.let { normalize(it.box, imageWidth, imageHeight) }
            ?: return

        val current = hoopRect
        if (current == null) {
            hoopRect = RectF(detected)
            onDebugEvent("HOOP_AUTO_ACQUIRED_V040")
            return
        }

        val distance = hypot(
            (centerX(current) - centerX(detected)).toDouble(),
            (centerY(current) - centerY(detected)).toDouble()
        ).toFloat()

        if (distance <= 0.10f) {
            val alpha = 0.10f
            hoopRect = RectF(
                lerp(current.left, detected.left, alpha),
                lerp(current.top, detected.top, alpha),
                lerp(current.right, detected.right, alpha),
                lerp(current.bottom, detected.bottom, alpha)
            )
        }
    }

    fun updateFrame(
        players: PlayerIdentityTracker.Snapshot,
        ball: FastBallTracker.Result?,
        nowMs: Long
    ): TrackerSnapshot {
        if (ball != null) {
            history.addLast(
                BallSample(
                    x = centerX(ball.box),
                    y = centerY(ball.box),
                    timeMs = nowMs,
                    confidence = ball.confidence,
                    source = ball.source
                )
            )
            trimHistory(nowMs)

            if (ball.source != "FAST_PREDICT") {
                updatePossession(
                    ball.box,
                    players.playerA,
                    players.playerB,
                    nowMs
                )
            }
        } else {
            trimHistory(nowMs)
        }

        val status = updateShotState(
            players = players,
            ball = ball,
            nowMs = nowMs
        )

        return TrackerSnapshot(
            playerA = players.playerA,
            playerB = players.playerB,
            ball = ball?.box,
            lastPossessor = lastPossessor,
            shotPlayer = shotPlayer,
            shotValue = shotValue,
            status = status
        )
    }

    private fun updatePossession(
        ball: RectF,
        a: RectF?,
        b: RectF?,
        nowMs: Long
    ) {
        val possessor = nearestPossessor(ball, a, b)
        val near = possessor != null

        if (near) {
            lastPossessor = possessor
        }

        if (wasBallNearPlayer && !near && lastPossessor != null) {
            val recent = history
                .filter { it.isMeasured }
                .takeLast(5)

            val rising =
                recent.size >= 2 &&
                    recent.last().y < recent.first().y - 0.004f

            if (rising) {
                releasePlayer = lastPossessor
                releaseValue = calculateShotValue(lastPossessor!!, a, b)
                releaseAtMs = nowMs
                onDebugEvent(
                    "RELEASE_V040 player=$releasePlayer value=$releaseValue"
                )
            }
        }

        wasBallNearPlayer = near
    }

    private fun updateShotState(
        players: PlayerIdentityTracker.Snapshot,
        ball: FastBallTracker.Result?,
        nowMs: Long
    ): String {
        val hoop = savedHoop ?: hoopRect ?: return "RIM SEARCH"
        val bx = ball?.let { centerX(it.box) }
        val by = ball?.let { centerY(it.box) }

        if (shotState == ShotState.COOLDOWN) {
            if (nowMs >= cooldownUntilMs) {
                resetShotState("COOLDOWN_END")
            } else {
                return "COOLDOWN"
            }
        }

        when (shotState) {
            ShotState.IDLE -> {
                if (ball == null) return "A/B LOCK / BALL SEARCH"

                if (shouldArmShot(ball, hoop, nowMs)) {
                    savedHoop = RectF(hoop)
                    shotStartedMs = nowMs
                    aboveRimMs = nowMs
                    shotPlayer =
                        if (releasePlayer != null && nowMs - releaseAtMs <= 2800L) {
                            releasePlayer
                        } else {
                            lastPossessor ?: closestPlayerToBall(
                                ball.box,
                                players.playerA,
                                players.playerB
                            )
                        }

                    shotValue =
                        if (releasePlayer != null && nowMs - releaseAtMs <= 2800L) {
                            releaseValue
                        } else {
                            shotPlayer?.let {
                                calculateShotValue(
                                    it,
                                    players.playerA,
                                    players.playerB
                                )
                            } ?: 1
                        }

                    shotState = ShotState.ABOVE_RIM
                    onDebugEvent(
                        "SHOT_ARMED_V040 player=$shotPlayer value=$shotValue source=" +
                            ball.source + " conf=" + ball.confidence
                    )
                    return "SHOT / ABOVE"
                }

                return "A/B LOCK / BALL TRACK"
            }

            ShotState.ABOVE_RIM -> {
                if (nowMs - shotStartedMs > 3200L) {
                    resetShotState("SHOT_TIMEOUT_ABOVE_V040")
                    return "MISS / TIMEOUT"
                }

                val saved = savedHoop ?: hoop

                if (
                    bx != null &&
                    by != null &&
                    by >= centerY(saved) + max(saved.height() * 0.08f, 0.004f)
                ) {
                    belowRimMs = nowMs
                    shotState = ShotState.BELOW_RIM
                    onDebugEvent(
                        "SHOT_BELOW_V040 x=$bx y=$by source=" + (ball?.source ?: "NONE")
                    )
                    return "SHOT / VERIFY"
                }

                return if (ball?.source == "FAST_PREDICT") {
                    "SHOT / PREDICT"
                } else {
                    "SHOT / ABOVE"
                }
            }

            ShotState.BELOW_RIM -> {
                val saved = savedHoop ?: hoop

                if (nowMs - belowRimMs < 120L) {
                    return "SHOT / VERIFY"
                }

                val made = detectMake(saved, nowMs)
                if (made) {
                    val scorer = shotPlayer
                    if (scorer != null) {
                        onDebugEvent(
                            "AUTO_SCORE_V040 player=$scorer points=$shotValue"
                        )
                        onAutomaticScore(scorer, shotValue)
                    } else {
                        onDebugEvent("MAKE_NO_SHOOTER_V040")
                    }

                    cooldownUntilMs = nowMs + 1100L
                    shotState = ShotState.COOLDOWN
                    history.clear()
                    return "MAKE"
                }

                if (nowMs - belowRimMs >= 700L) {
                    resetShotState("SHOT_MISS_V040")
                    return "MISS"
                }

                return "SHOT / VERIFY"
            }

            ShotState.COOLDOWN -> return "COOLDOWN"
        }
    }

    private fun shouldArmShot(
        ball: FastBallTracker.Result,
        hoop: RectF,
        nowMs: Long
    ): Boolean {
        if (ball.source == "FAST_PREDICT") return false

        val x = centerX(ball.box)
        val y = centerY(ball.box)
        val hx = centerX(hoop)
        val hy = centerY(hoop)

        val nearHoopX = abs(x - hx) <= hoop.width() * 4.8f
        val aboveHoop =
            y <= hy + hoop.height() * 0.04f &&
                y >= hoop.top - max(hoop.height() * 4.0f, 0.07f)

        if (!nearHoopX || !aboveHoop) return false

        val measured = history
            .filter { it.isMeasured && nowMs - it.timeMs <= 900L }
            .takeLast(6)

        if (measured.size < 2) return false

        val rising =
            measured.last().y < measured.first().y - 0.0035f

        val recentRelease =
            releasePlayer != null &&
                nowMs - releaseAtMs <= 2800L

        return rising || recentRelease
    }

    private fun detectMake(hoop: RectF, nowMs: Long): Boolean {
        val hx = centerX(hoop)
        val hy = centerY(hoop)
        val hw = hoop.width()

        val samples = history
            .filter {
                it.timeMs >= aboveRimMs - 120L &&
                    it.timeMs <= nowMs
            }
            .sortedBy { it.timeMs }

        if (samples.size < 2) return false
        if (samples.none { it.isMeasured }) return false

        var bestCrossingX: Float? = null
        var bestPairMeasured = false

        for (i in 1 until samples.size) {
            val a = samples[i - 1]
            val b = samples[i]

            if (
                a.y <= hy &&
                b.y >= hy &&
                b.y > a.y &&
                b.timeMs - a.timeMs <= 420L
            ) {
                val dy = b.y - a.y
                if (dy <= 0.0001f) continue

                val t = ((hy - a.y) / dy).coerceIn(0f, 1f)
                val crossingX = a.x + (b.x - a.x) * t

                if (
                    bestCrossingX == null ||
                    abs(crossingX - hx) < abs(bestCrossingX - hx)
                ) {
                    bestCrossingX = crossingX
                    bestPairMeasured = a.isMeasured || b.isMeasured
                }
            }
        }

        val crossingX = bestCrossingX
        val mouthMargin = hw * 0.18f
        val directCross =
            crossingX != null &&
                bestPairMeasured &&
                crossingX >= hoop.left - mouthMargin &&
                crossingX <= hoop.right + mouthMargin

        val belowSamples = samples.filter {
            it.timeMs >= belowRimMs &&
                it.y >= hy
        }

        val continuedDown =
            belowSamples.size >= 2 &&
                belowSamples.last().y >= belowSamples.first().y - 0.003f

        val stayedNearRim =
            belowSamples.takeLast(4).all {
                abs(it.x - hx) <= hw * 0.95f
            }

        val result = directCross && continuedDown && stayedNearRim

        if (crossingX != null) {
            onDebugEvent(
                "MAKE_CHECK_V040 crossX=$crossingX direct=$directCross " +
                    "down=$continuedDown near=$stayedNearRim samples=" + samples.size
            )
        }

        return result
    }

    private fun resetShotState(reason: String) {
        onDebugEvent(reason)
        shotState = ShotState.IDLE
        shotPlayer = null
        shotValue = 1
        shotStartedMs = 0L
        aboveRimMs = 0L
        belowRimMs = 0L
        savedHoop = null
        releasePlayer = null
        releaseAtMs = 0L
    }

    private fun trimHistory(nowMs: Long) {
        while (history.size > 180) history.removeFirst()
        while (history.isNotEmpty() && nowMs - history.first().timeMs > 6000L) {
            history.removeFirst()
        }
    }

    private fun nearestPossessor(
        ball: RectF,
        a: RectF?,
        b: RectF?
    ): Char? {
        val bx = centerX(ball)
        val by = centerY(ball)

        fun cost(player: RectF?): Float? {
            player ?: return null
            val expanded = RectF(
                player.left - player.width() * 0.40f,
                player.top - player.height() * 0.18f,
                player.right + player.width() * 0.40f,
                player.bottom + player.height() * 0.13f
            )
            if (!expanded.contains(bx, by)) return null
            return hypot(
                (bx - centerX(player)).toDouble(),
                (by - centerY(player)).toDouble()
            ).toFloat()
        }

        val ca = cost(a)
        val cb = cost(b)

        return when {
            ca == null && cb == null -> null
            cb == null -> 'A'
            ca == null -> 'B'
            ca <= cb -> 'A'
            else -> 'B'
        }
    }

    private fun closestPlayerToBall(
        ball: RectF,
        a: RectF?,
        b: RectF?
    ): Char? {
        val bx = centerX(ball)
        val by = centerY(ball)

        val da = a?.let {
            hypot(
                (bx - centerX(it)).toDouble(),
                (by - centerY(it)).toDouble()
            ).toFloat()
        }

        val db = b?.let {
            hypot(
                (bx - centerX(it)).toDouble(),
                (by - centerY(it)).toDouble()
            ).toFloat()
        }

        return when {
            da == null && db == null -> null
            db == null -> 'A'
            da == null -> 'B'
            da <= db -> 'A'
            else -> 'B'
        }
    }

    private fun calculateShotValue(
        player: Char,
        a: RectF?,
        b: RectF?
    ): Int {
        if (threePointLine.size < 2) return 1
        val box = if (player == 'A') a else b
        box ?: return 1

        val footX = centerX(box)
        val footY = box.bottom
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
    private fun lerp(a: Float, b: Float, t: Float) = a + (b - a) * t
}
