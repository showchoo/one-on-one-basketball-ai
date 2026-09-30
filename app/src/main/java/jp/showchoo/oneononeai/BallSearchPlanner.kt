package jp.showchoo.oneononeai

import android.graphics.RectF
import kotlin.math.hypot
import kotlin.math.max

/**
 * v0.6 search policy.
 *
 * Detection, association and search are deliberately separated. The detector
 * answers "is there a basketball in this crop?", BallTrackFusion associates
 * observations over time, and this class decides which crop is worth spending
 * the next inference on.
 */
class BallSearchPlanner {
    data class SearchRoi(
        val rect: RectF,
        val reason: String
    )

    private var motionIndex = 0
    private var contextIndex = 0
    private var searchCount = 0

    fun reset() {
        motionIndex = 0
        contextIndex = 0
        searchCount = 0
    }

    fun next(
        players: PlayerIdentityTracker.Snapshot,
        hoop: RectF?,
        motionProposals: List<MotionBallProposer.Proposal>,
        active: BallTrackFusion.SearchAnchor?,
        focusPlayer: Char?,
        rimPriority: Boolean
    ): SearchRoi {
        searchCount = (searchCount + 1) % 100000

        if (active != null) {
            if (active.provisional) {
                // Re-check a fresh unconfirmed hit immediately. A slightly
                // larger crop tolerates ball motion during the ~200-300ms
                // inference delay on entry-level phones.
                return SearchRoi(
                    centeredRoi(
                        centerX(active.box),
                        centerY(active.box),
                        0.18f,
                        0.24f
                    ),
                    "VERIFY_PENDING"
                )
            }

            val w = (0.13f + active.speed * 0.040f).coerceIn(0.13f, 0.25f)
            val h = (0.18f + active.speed * 0.055f).coerceIn(0.18f, 0.31f)
            return SearchRoi(
                centeredRoi(centerX(active.box), centerY(active.box), w, h),
                "LOCKED"
            )
        }

        if (rimPriority && hoop != null && searchCount % 2 == 0) {
            return SearchRoi(
                centeredRoi(
                    centerX(hoop),
                    centerY(hoop) - hoop.height() * 1.10f,
                    max(0.22f, hoop.width() * 5.5f).coerceAtMost(0.42f),
                    max(0.28f, hoop.height() * 6.5f).coerceAtMost(0.52f)
                ),
                "RIM_FOCUS"
            )
        }

        val rankedMotion = motionProposals
            .map { proposal ->
                proposal to contextualMotionScore(
                    proposal = proposal,
                    players = players,
                    hoop = hoop,
                    focusPlayer = focusPlayer
                )
            }
            .sortedByDescending { it.second }
            .take(5)

        // Most search passes inspect motion, but not blindly. The rank is
        // contextual: motion beside a player or close to the rim is preferred,
        // while motion deep inside a player's torso is penalized.
        val useMotion = rankedMotion.isNotEmpty() && searchCount % 4 != 0
        if (useMotion) {
            val i = motionIndex % rankedMotion.size
            motionIndex = (motionIndex + 1) % 100000
            return SearchRoi(
                RectF(rankedMotion[i].first.roi),
                "MOTION_R" + (i + 1)
            )
        }

        val choices = mutableListOf<SearchRoi>()

        val focused = when (focusPlayer) {
            'A' -> players.playerA
            'B' -> players.playerB
            else -> null
        }
        if (focused != null) {
            choices += SearchRoi(playerBallRoi(focused), "POSSESSION_" + focusPlayer)
        }

        players.playerA?.let {
            if (focusPlayer != 'A') {
                choices += SearchRoi(playerBallRoi(it), "PLAYER_A")
            }
        }
        players.playerB?.let {
            if (focusPlayer != 'B') {
                choices += SearchRoi(playerBallRoi(it), "PLAYER_B")
            }
        }

        hoop?.let {
            choices += SearchRoi(
                centeredRoi(
                    centerX(it),
                    centerY(it) - it.height() * 0.65f,
                    max(0.20f, it.width() * 4.5f).coerceAtMost(0.36f),
                    max(0.24f, it.height() * 5.0f).coerceAtMost(0.44f)
                ),
                "HOOP"
            )
        }

        // Sparse fallback coverage. These are intentionally last because v0.5
        // showed that unrestricted full-court acquisition creates false locks.
        choices += SearchRoi(RectF(0.05f, 0.05f, 0.50f, 0.55f), "TILE_LT")
        choices += SearchRoi(RectF(0.50f, 0.05f, 0.95f, 0.55f), "TILE_RT")
        choices += SearchRoi(RectF(0.05f, 0.45f, 0.50f, 0.95f), "TILE_LB")
        choices += SearchRoi(RectF(0.50f, 0.45f, 0.95f, 0.95f), "TILE_RB")

        val choice = choices[contextIndex % choices.size]
        contextIndex = (contextIndex + 1) % 100000
        return choice
    }

    private fun contextualMotionScore(
        proposal: MotionBallProposer.Proposal,
        players: PlayerIdentityTracker.Snapshot,
        hoop: RectF?,
        focusPlayer: Char?
    ): Float {
        var score = proposal.score

        fun addPlayerContext(player: RectF?, focused: Boolean) {
            player ?: return
            val cx = proposal.centerX
            val cy = proposal.centerY

            if (insidePlayerCore(cx, cy, player)) {
                score -= 0.55f
                return
            }

            val d = distanceToRect(cx, cy, expand(player, 0.70f, 0.22f))
            if (d <= 0.015f) score += if (focused) 0.62f else 0.42f
            else if (d <= 0.06f) score += if (focused) 0.34f else 0.22f
        }

        addPlayerContext(players.playerA, focusPlayer == 'A')
        addPlayerContext(players.playerB, focusPlayer == 'B')

        hoop?.let {
            val d = hypot(
                (proposal.centerX - centerX(it)).toDouble(),
                (proposal.centerY - centerY(it)).toDouble()
            ).toFloat()
            if (d <= 0.18f) score += 0.30f
        }

        return score
    }

    private fun playerBallRoi(player: RectF): RectF {
        val cx = centerX(player)
        val cy = centerY(player) + player.height() * 0.05f
        val w = max(0.22f, player.width() * 2.35f).coerceAtMost(0.42f)
        val h = max(0.34f, player.height() * 1.35f).coerceAtMost(0.62f)
        return centeredRoi(cx, cy, w, h)
    }

    private fun insidePlayerCore(x: Float, y: Float, player: RectF): Boolean {
        val left = player.left + player.width() * 0.18f
        val right = player.right - player.width() * 0.18f
        val top = player.top + player.height() * 0.08f
        val bottom = player.top + player.height() * 0.72f
        return x in left..right && y in top..bottom
    }

    private fun expand(r: RectF, xFactor: Float, yFactor: Float): RectF =
        RectF(
            r.left - r.width() * xFactor,
            r.top - r.height() * yFactor,
            r.right + r.width() * xFactor,
            r.bottom + r.height() * yFactor
        )

    private fun distanceToRect(x: Float, y: Float, r: RectF): Float {
        val dx = when {
            x < r.left -> r.left - x
            x > r.right -> x - r.right
            else -> 0f
        }
        val dy = when {
            y < r.top -> r.top - y
            y > r.bottom -> y - r.bottom
            else -> 0f
        }
        return hypot(dx.toDouble(), dy.toDouble()).toFloat()
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

    private fun centerX(r: RectF) = (r.left + r.right) / 2f
    private fun centerY(r: RectF) = (r.top + r.bottom) / 2f
}
