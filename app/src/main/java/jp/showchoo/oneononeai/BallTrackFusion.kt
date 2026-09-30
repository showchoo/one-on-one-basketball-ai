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
    private var pending: Pending? = null
    private var searchIndex = 0

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
        pending = null
        searchIndex = 0
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
                    val gate = if (source == "BALL_ROI_YOLO") 0.16f else 0.22f
                    pair.second <= gate
                }
                .minByOrNull { it.second - it.first.score * 0.08f }
                ?.first
        } else {
            candidates.firstOrNull()
        } ?: return false

        val minScore = if (source == "BALL_ROI_YOLO") 0.09f else 0.13f
        if (chosen.score < minScore) return false

        if (current == null) {
            val p = pending
            if (chosen.score >= 0.32f) {
                acquire(chosen.box, chosen.score, captureTimeMs, receivedTimeMs, source)
                pending = null
                return true
            }

            if (p != null &&
                captureTimeMs - p.timeMs in 20L..520L &&
                distance(p.box, chosen.box) <= 0.12f
            ) {
                val dt = (captureTimeMs - p.timeMs).coerceAtLeast(1L) / 1000f
                vx = ((centerX(chosen.box) - centerX(p.box)) / dt).coerceIn(-3f, 3f)
                vy = ((centerY(chosen.box) - centerY(p.box)) / dt).coerceIn(-3f, 3f)
                acquire(chosen.box, chosen.score, captureTimeMs, receivedTimeMs, source)
                pending = null
                onDebugEvent("BALL_ACQUIRE_TWO_HIT score=${chosen.score} source=$source")
                return true
            }

            pending = Pending(RectF(chosen.box), chosen.score, captureTimeMs, source)
            onDebugEvent("BALL_PENDING score=${chosen.score} source=$source")
            return false
        }

        val predictedAtCapture = predictBox(captureTimeMs, allowExpired = true) ?: current
        val d = distance(predictedAtCapture, chosen.box)

        val hardJump = d > 0.24f
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
            pending = null
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
        nowMs: Long
    ): RectF {
        val active = predictBox(nowMs, allowExpired = false)
        if (active != null) {
            val speed = hypot(vx.toDouble(), vy.toDouble()).toFloat()
            val w = (0.22f + speed * 0.06f).coerceIn(0.22f, 0.38f)
            val h = (0.28f + speed * 0.08f).coerceIn(0.28f, 0.46f)
            return centeredRoi(centerX(active), centerY(active), w, h)
        }

        val choices = mutableListOf<RectF>()
        players.playerA?.let { choices += expandPlayerRoi(it) }
        players.playerB?.let { choices += expandPlayerRoi(it) }
        hoop?.let {
            choices += centeredRoi(
                centerX(it),
                centerY(it) - 0.06f,
                0.34f,
                0.46f
            )
        }

        choices += RectF(0.00f, 0.00f, 0.58f, 0.62f)
        choices += RectF(0.42f, 0.00f, 1.00f, 0.62f)
        choices += RectF(0.00f, 0.38f, 0.58f, 1.00f)
        choices += RectF(0.42f, 0.38f, 1.00f, 1.00f)

        val roi = choices[searchIndex % choices.size]
        searchIndex = (searchIndex + 1) % 100000
        return clamp(roi)
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
        pending = null

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
