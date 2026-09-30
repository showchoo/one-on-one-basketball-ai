package jp.showchoo.oneononeai

import android.graphics.RectF
import kotlin.math.hypot
import kotlin.math.max

class BallContextTracker(
    private val onDebugEvent: (String) -> Unit = {}
) {
    enum class Mode { SEARCH_POSSESSION, POSSESSED, FLIGHT, RIM, LOOSE }

    private var mode = Mode.SEARCH_POSSESSION
    private var possessor: Char? = null
    private var lastBallSeenMs = 0L
    private var lastNearPlayerMs = 0L
    private var lastPossessor: Char? = null
    private var flightUntilMs = 0L

    fun reset() {
        mode = Mode.SEARCH_POSSESSION
        possessor = null
        lastBallSeenMs = 0L
        lastNearPlayerMs = 0L
        lastPossessor = null
        flightUntilMs = 0L
        onDebugEvent("BALL_CONTEXT_RESET")
    }

    fun currentMode(): Mode = mode

    fun allowsFreeSpace(nowMs: Long): Boolean =
        mode == Mode.FLIGHT || mode == Mode.RIM || nowMs <= flightUntilMs

    fun update(
        players: PlayerIdentityTracker.Snapshot,
        ball: BallTrackFusion.Result?,
        hoop: RectF?,
        nowMs: Long
    ) {
        if (ball == null) {
            if (mode == Mode.FLIGHT || mode == Mode.RIM) {
                if (nowMs > flightUntilMs) transition(Mode.LOOSE, "flight_timeout")
            } else if (lastBallSeenMs > 0L && nowMs - lastBallSeenMs > 1200L) {
                transition(Mode.SEARCH_POSSESSION, "ball_lost")
                possessor = null
            }
            return
        }

        lastBallSeenMs = nowMs
        val owner = nearestPlayer(ball.box, players.playerA, players.playerB)
        val nearHoop = hoop?.let { isNearHoop(ball.box, it) } ?: false

        if (owner != null) {
            lastNearPlayerMs = nowMs
            lastPossessor = owner
            possessor = owner
            flightUntilMs = 0L
            transition(Mode.POSSESSED, "near_player_" + owner)
            return
        }

        if (nearHoop && (mode == Mode.FLIGHT || nowMs - lastNearPlayerMs <= 1800L)) {
            flightUntilMs = max(flightUntilMs, nowMs + 900L)
            transition(Mode.RIM, "near_hoop")
            return
        }

        if (mode == Mode.POSSESSED || nowMs - lastNearPlayerMs <= 650L) {
            possessor = null
            flightUntilMs = nowMs + 1800L
            transition(Mode.FLIGHT, "released_from_" + (lastPossessor ?: '?'))
            return
        }

        if (mode == Mode.FLIGHT || mode == Mode.RIM) {
            if (nowMs <= flightUntilMs) transition(Mode.FLIGHT, "continue_flight")
            else transition(Mode.LOOSE, "free_ball_timeout")
        } else {
            transition(Mode.LOOSE, "unowned_ball")
        }
    }

    fun candidateAllowed(
        box: RectF,
        players: PlayerIdentityTracker.Snapshot,
        hoop: RectF?,
        nowMs: Long,
        alreadyLocked: Boolean
    ): Boolean {
        if (alreadyLocked) return true
        if (allowsFreeSpace(nowMs)) return true
        if (nearestPlayer(box, players.playerA, players.playerB) != null) return true
        if (hoop != null && isNearHoop(box, hoop)) return true
        return false
    }

    private fun transition(next: Mode, reason: String) {
        if (next == mode) return
        val old = mode
        mode = next
        onDebugEvent("BALL_CONTEXT " + old + "->" + next + " reason=" + reason)
    }

    private fun nearestPlayer(ball: RectF, a: RectF?, b: RectF?): Char? {
        val bx = centerX(ball)
        val by = centerY(ball)

        fun cost(player: RectF?): Float? {
            player ?: return null
            val expanded = RectF(
                player.left - player.width() * 0.48f,
                player.top - player.height() * 0.26f,
                player.right + player.width() * 0.48f,
                player.bottom + player.height() * 0.16f
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

    private fun isNearHoop(ball: RectF, hoop: RectF): Boolean {
        val bx = centerX(ball)
        val by = centerY(ball)
        val hx = centerX(hoop)
        val hy = centerY(hoop)
        val xGate = max(hoop.width() * 4.5f, 0.10f)
        val yGate = max(hoop.height() * 5.0f, 0.12f)
        return kotlin.math.abs(bx - hx) <= xGate &&
            kotlin.math.abs(by - hy) <= yGate
    }

    private fun centerX(r: RectF) = (r.left + r.right) / 2f
    private fun centerY(r: RectF) = (r.top + r.bottom) / 2f
}
