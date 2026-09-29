package jp.showchoo.oneononeai

enum class CommentaryMode(val buttonLabel: String) {
    LIVE("実況: ON"),
    SCORE_ONLY("実況: 得点のみ"),
    OFF("実況: OFF");

    fun next(): CommentaryMode = when (this) {
        LIVE -> SCORE_ONLY
        SCORE_ONLY -> OFF
        OFF -> LIVE
    }
}

data class ScoreEvent(
    val player: Char,
    val points: Int,
    val previousScoreA: Int,
    val previousScoreB: Int,
    val scoreA: Int,
    val scoreB: Int,
    val targetScore: Int,
    val gameOver: Boolean
)

class CommentaryEngine(
    var mode: CommentaryMode = CommentaryMode.LIVE
) {
    private var lastScorer: Char? = null
    private var scoringStreak = 0

    fun reset() {
        lastScorer = null
        scoringStreak = 0
    }

    fun onGameStart(targetScore: Int): String? {
        reset()
        return when (mode) {
            CommentaryMode.OFF -> null
            CommentaryMode.SCORE_ONLY -> "ゲームスタート"
            CommentaryMode.LIVE -> "ゲームスタート。${targetScore}点先取です"
        }
    }

    fun onScore(event: ScoreEvent): String? {
        updateStreak(event.player)

        return when (mode) {
            CommentaryMode.OFF -> null
            CommentaryMode.SCORE_ONLY -> scoreOnly(event)
            CommentaryMode.LIVE -> liveCommentary(event)
        }
    }

    private fun scoreOnly(e: ScoreEvent): String {
        return "プレイヤー ${e.player}、${e.points}ポイント。${e.scoreA}対${e.scoreB}"
    }

    private fun liveCommentary(e: ScoreEvent): String {
        if (e.gameOver) {
            return "決まった！プレイヤー ${e.player}、${e.scoreA}対${e.scoreB}で勝利！"
        }

        val previousLeader = leader(e.previousScoreA, e.previousScoreB)
        val currentLeader = leader(e.scoreA, e.scoreB)

        val parts = mutableListOf<String>()
        parts += if (e.points == 2) {
            "プレイヤー ${e.player}、外から決めた！2ポイント！"
        } else {
            "プレイヤー ${e.player}、決めた！"
        }

        when {
            e.scoreA == 9 && e.scoreB == 9 && e.targetScore == 10 ->
                parts += "9対9。次の1本で決着！"

            e.scoreA == e.scoreB ->
                parts += "${e.scoreA}対${e.scoreB}、同点！"

            previousLeader != null && currentLeader != null && previousLeader != currentLeader ->
                parts += "逆転！${e.scoreA}対${e.scoreB}！"

            else ->
                parts += "${e.scoreA}対${e.scoreB}。"
        }

        val scorerScore = if (e.player == 'A') e.scoreA else e.scoreB
        if (scorerScore == e.targetScore - 1) {
            parts += "プレイヤー ${e.player}、ゲームポイント！"
        } else if (scoringStreak >= 3) {
            parts += "${scoringStreak}連続得点！"
        }

        return parts.joinToString(" ")
    }

    private fun updateStreak(player: Char) {
        if (lastScorer == player) {
            scoringStreak += 1
        } else {
            lastScorer = player
            scoringStreak = 1
        }
    }

    private fun leader(a: Int, b: Int): Char? = when {
        a > b -> 'A'
        b > a -> 'B'
        else -> null
    }
}
