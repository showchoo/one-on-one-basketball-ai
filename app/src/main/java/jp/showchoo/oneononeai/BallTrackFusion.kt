package jp.showchoo.oneononeai

import android.graphics.RectF
import kotlin.math.hypot
import kotlin.math.max

class BallTrackFusion(
    private val onDebugEvent: (String) -> Unit = {}
) {
    data class Result(
        val box: RectF,
        val confidence: Float,
        val source: String,
        val trackingMs: Long = 0L,
        val ageSinceVerifiedMs: Long
    )

    data class SearchRoi(
        val rect: RectF,
        val reason: String
    )

    private data class Pending(
        val box: RectF,
        val score: Float,
        val timeMs: Long,
        val source: String
    )

    private var confirmedBox: RectF? = null
    private var vx = 0f
    private var vy = 0f
    private var lastVerifiedMs = 0L
    private var lastUpdateMs = 0L
    private var lastScore = 0f
    private var lastSource = ""
    private val pending = mutableListOf<Pending>()
    private var searchIndex = 0
    private var motionSearchIndex = 0
    private var unlockedSearchCount = 0

    companion object {
        private const val PREDICT_VISIBLE_MS = 360L
        private const val TRACK_EXPIRE_MS = 650L
    }

    @Synchronized
    fun reset() {
        confirmedBox = null
        vx = 0f
        vy = 0f
        lastVerifiedMs = 0L
        lastUpdateMs = 0L
        lastScore = 0f
        lastSource = ""
        pending.clear()
        searchIndex = 0
        motionSearchIndex = 0
        unlockedSearchCount = 0
    }

    @Synchronized
    fun observe(
        detections: List<AiDetection>,
        captureTimeMs: Long,
        receivedTimeMs: Long,
        source: String
    ): Boolean {
        if (detections.isEmpty()) return false

        val candidates = detections
            .filter { it.label == "sports ball" }
            .sortedByDescending { it.score }
        if (candidates.isEmpty()) return false

        val current = confirmedBox
        val chosen = if (current != null && lastVerifiedMs > 0L) {
            val predictedAtCapture = predictBox(captureTimeMs, allowExpired = true)
            val px = predictedAtCapture?.let { centerX(it) } ?: centerX(current)
            val py = predictedAtCapture?.let { centerY(it) } ?: centerY(current)

            candidates
                .map {
                    val d = hypot(
                        (centerX(it.box) - px).toDouble(),
                        (centerY(it.box) - py).toDouble()
                    ).toFloat()
                    it to d
                }
                .filter { pair ->
                    // Once locked, prediction already compensates for velocity.
                    // Keep the residual gate deliberately tight so a false
                    // basketball-looking object inside a large ROI cannot take over.
                    val gate = when (source) {
                        "BALL_MOTION_ROI" -> 0.12f
                        "BALL_ROI_YOLO" -> 0.10f
                        else -> 0.14f
                    }
                    pair.second <= gate
                }
                .minByOrNull { it.second - it.first.score * 0.08f }
                ?.first
        } else {
            candidates.firstOrNull()
        } ?: return false

        val minScore = when (source) {
            "BALL_MOTION_ROI" -> 0.10f
            "BALL_ROI_YOLO" -> 0.11f
            else -> 0.20f
        }
        if (chosen.score < minScore) return false

        if (current == null) {
            // v0.5.2 could lock from a single false positive. Never acquire from
            // one observation now. Keep several short-lived hypotheses because
            // MOTION_1..4 are interleaved and the same real ball may not be
            // inspected on two consecutive ROI passes.
            pending.removeAll {
                val age = captureTimeMs - it.timeMs
                age < 0L || age > 900L
            }

            val match = pending
                .mapNotNull { p ->
                    val dtMs = captureTimeMs - p.timeMs
                    if (dtMs !in 35L..900L) {
                        null
                    } else {
                        val dynamicGate =
                            (0.055f + dtMs / 1000f * 0.42f).coerceAtMost(0.28f)
                        val d = distance(p.box, chosen.box)
                        if (d <= dynamicGate) Triple(p, dtMs, d) else null
                    }
                }
                .minByOrNull { it.third }

            if (match != null) {
                val p = match.first
                val dt = match.second.coerceAtLeast(1L) / 1000f
                vx = ((centerX(chosen.box) - centerX(p.box)) / dt).coerceIn(-3f, 3f)
                vy = ((centerY(chosen.box) - centerY(p.box)) / dt).coerceIn(-3f, 3f)
                acquire(chosen.box, chosen.score, captureTimeMs, receivedTimeMs, source)
                pending.clear()
                onDebugEvent(
                    "BALL_ACQUIRE_TWO_HIT score=${chosen.score} source=$source dtMs=${match.second} d=${match.third}"
                )
                return true
            }

            pending += Pending(RectF(chosen.box), chosen.score, captureTimeMs, source)
            while (pending.size > 6) pending.removeAt(0)
            onDebugEvent(
                "BALL_PENDING score=${chosen.score} source=$source hypotheses=${pending.size}"
            )
            return false
        }

        val predictedAtCapture = predictBox(captureTimeMs, allowExpired = true) ?: current
        val d = distance(predictedAtCapture, chosen.box)

        val hardJump = d > 0.16f
        if (hardJump && chosen.score < 0.48f) {
            onDebugEvent("BALL_REJECT_JUMP d=$d score=${chosen.score} source=$source")
            return false
        }

        updateConfirmed(
            observation = chosen.box,
            score = chosen.score,
            captureTimeMs = captureTimeMs,
            receivedTimeMs = receivedTimeMs,
            source = source
        )
        return true
    }

    @Synchronized
    fun predict(nowMs: Long): Result? {
        val box = confirmedBox ?: return null
        val age = nowMs - lastVerifiedMs

        if (age > TRACK_EXPIRE_MS) {
            confirmedBox = null
            pending.clear()
            onDebugEvent("BALL_TRACK_EXPIRED age=$age")
            return null
        }

        if (age > PREDICT_VISIBLE_MS) {
            return null
        }

        val predicted = predictBox(nowMs, allowExpired = true) ?: RectF(box)
        val confidence = (
            lastScore * (1f - age.toFloat() / (PREDICT_VISIBLE_MS * 1.35f))
        ).coerceIn(0.10f, 1f)

        return Result(
            box = predicted,
            confidence = confidence,
            source = if (age <= 80L) lastSource else "BALL_PREDICT",
            ageSinceVerifiedMs = age
        )
    }

    @Synchronized
    fun nextSearchRoi(
        players: PlayerIdentityTracker.Snapshot,
        hoop: RectF?,
        nowMs: Long,
        motionProposals: List<MotionBallProposer.Proposal>
    ): SearchRoi {
        val active = predictBox(nowMs, allowExpired = false)
        if (active != null) {
            val speed = hypot(vx.toDouble(), vy.toDouble()).toFloat()
            val w = (0.20f + speed * 0.055f).coerceIn(0.20f, 0.34f)
            val h = (0.26f + speed * 0.07f).coerceIn(0.26f, 0.42f)
            return SearchRoi(
                centeredRoi(centerX(active), centerY(active), w, h),
                "LOCKED"
            )
        }

        // v0.5.1 accidentally inspected only motionProposals.firstOrNull().
        // Player limbs often outrank the ball, so candidates 2-4 must also
        // get a chance. Periodically force contextual search so constant
        // player motion cannot starve PLAYER/HOOP/TILE search forever.
        val motionPoolSize = minOf(4, motionProposals.size)
        val forceContextScan = unlockedSearchCount % 5 == 4
        unlockedSearchCount = (unlockedSearchCount + 1) % 100000

        if (motionPoolSize > 0 && !forceContextScan) {
            val motionIndex = motionSearchIndex % motionPoolSize
            motionSearchIndex = (motionSearchIndex + 1) % 100000
            val motion = motionProposals[motionIndex]
            return SearchRoi(
                motion.roi,
                "MOTION_${motionIndex + 1}"
            )
        }

        val choices = mutableListOf<SearchRoi>()
        players.playerA?.let {
            choices += SearchRoi(expandPlayerRoi(it), "PLAYER_A")
        }
        players.playerB?.let {
            choices += SearchRoi(expandPlayerRoi(it), "PLAYER_B")
        }
        hoop?.let {
            choices += SearchRoi(
                centeredRoi(
                    centerX(it),
                    centerY(it) - 0.06f,
                    0.32f,
                    0.44f
                ),
                "HOOP"
            )
        }

        choices += SearchRoi(RectF(0.00f, 0.00f, 0.54f, 0.58f), "TILE_LT")
        choices += SearchRoi(RectF(0.46f, 0.00f, 1.00f, 0.58f), "TILE_RT")
        choices += SearchRoi(RectF(0.00f, 0.42f, 0.54f, 1.00f), "TILE_LB")
        choices += SearchRoi(RectF(0.46f, 0.42f, 1.00f, 1.00f), "TILE_RB")

        val choice = choices[searchIndex % choices.size]
        searchIndex = (searchIndex + 1) % 100000
        return SearchRoi(clamp(choice.rect), choice.reason)
    }

    @Synchronized
    fun isLocked(nowMs: Long): Boolean =
        confirmedBox != null && nowMs - lastVerifiedMs <= PREDICT_VISIBLE_MS

    private fun acquire(
        observation: RectF,
        score: Float,
        captureTimeMs: Long,
        receivedTimeMs: Long,
        source: String
    ) {
        confirmedBox = RectF(observation)
        lastVerifiedMs = receivedTimeMs
        lastUpdateMs = captureTimeMs
        lastScore = score
        lastSource = source
        onDebugEvent(
            "BALL_ACQUIRED score=$score source=$source latency=" +
                (receivedTimeMs - captureTimeMs)
        )
    }

    private fun updateConfirmed(
        observation: RectF,
        score: Float,
        captureTimeMs: Long,
        receivedTimeMs: Long,
        source: String
    ) {
        val current = confirmedBox ?: run {
            acquire(observation, score, captureTimeMs, receivedTimeMs, source)
            return
        }

        val predictedAtCapture = predictBox(captureTimeMs, allowExpired = true) ?: current
        val dt = (captureTimeMs - lastUpdateMs).coerceIn(20L, 700L) / 1000f

        val measuredVx =
            ((centerX(observation) - centerX(predictedAtCapture)) / dt).coerceIn(-3f, 3f)
        val measuredVy =
            ((centerY(observation) - centerY(predictedAtCapture)) / dt).coerceIn(-3f, 3f)

        val alpha = if (score >= 0.30f) 0.50f else 0.30f
        vx = vx * (1f - alpha) + measuredVx * alpha
        vy = vy * (1f - alpha) + measuredVy * alpha

        val latencySec = (receivedTimeMs - captureTimeMs).coerceIn(0L, 500L) / 1000f
        val projectedX = (
            centerX(observation) + vx * latencySec
        ).coerceIn(0.01f, 0.99f)
        val projectedY = (
            centerY(observation) + vy * latencySec + 0.5f * 0.72f * latencySec * latencySec
        ).coerceIn(0.01f, 0.99f)

        val smoothed = centeredRoi(
            projectedX,
            projectedY,
            observation.width().coerceIn(0.008f, 0.10f),
            observation.height().coerceIn(0.008f, 0.10f)
        )

        confirmedBox = smoothed
        lastVerifiedMs = receivedTimeMs
        lastUpdateMs = receivedTimeMs
        lastScore = score
        lastSource = source
        pending.clear()

        onDebugEvent(
            "BALL_VERIFY score=$score source=$source latency=" +
                (receivedTimeMs - captureTimeMs) +
                " x=$projectedX y=$projectedY"
        )
    }

    private fun predictBox(
        timeMs: Long,
        allowExpired: Boolean
    ): RectF? {
        val current = confirmedBox ?: return null
        if (!allowExpired && timeMs - lastVerifiedMs > TRACK_EXPIRE_MS) return null

        val dt = (timeMs - lastUpdateMs).coerceIn(0L, 650L) / 1000f
        val x = (
            centerX(current) + vx * dt
        ).coerceIn(0.01f, 0.99f)
        val y = (
            centerY(current) + vy * dt + 0.5f * 0.72f * dt * dt
        ).coerceIn(0.01f, 0.99f)

        return centeredRoi(
            x,
            y,
            current.width(),
            current.height()
        )
    }

    private fun expandPlayerRoi(player: RectF): RectF {
        val cx = centerX(player)
        val cy = centerY(player) - player.height() * 0.08f
        val w = max(0.28f, player.width() * 2.8f).coerceAtMost(0.52f)
        val h = max(0.42f, player.height() * 1.45f).coerceAtMost(0.72f)
        return centeredRoi(cx, cy, w, h)
    }

    private fun centeredRoi(
        cx: Float,
        cy: Float,
        width: Float,
        height: Float
    ): RectF {
        val w = width.coerceIn(0.008f, 1f)
        val h = height.coerceIn(0.008f, 1f)
        var left = cx - w / 2f
        var top = cy - h / 2f
        left = left.coerceIn(0f, 1f - w)
        top = top.coerceIn(0f, 1f - h)
        return RectF(left, top, left + w, top + h)
    }

    private fun clamp(r: RectF): RectF =
        RectF(
            r.left.coerceIn(0f, 1f),
            r.top.coerceIn(0f, 1f),
            r.right.coerceIn(0f, 1f),
            r.bottom.coerceIn(0f, 1f)
        )

    private fun distance(a: RectF, b: RectF): Float =
        hypot(
            (centerX(a) - centerX(b)).toDouble(),
            (centerY(a) - centerY(b)).toDouble()
        ).toFloat()

    private fun centerX(r: RectF) = (r.left + r.right) / 2f
    private fun centerY(r: RectF) = (r.top + r.bottom) / 2f
}
