package jp.showchoo.oneononeai

class GameEngine(
    private val targetScore: Int = 10,
    private val onScoreChanged: (Int, Int) -> Unit,
    private val onAnnouncement: (String) -> Unit,
    private val onGameOver: (Char, Int, Int) -> Unit
) {
    var scoreA: Int = 0
        private set
    var scoreB: Int = 0
        private set
    var running: Boolean = false
        private set

    fun start() {
        if (scoreA >= targetScore || scoreB >= targetScore) reset()
        running = true
        onAnnouncement("ゲームスタート")
    }

    fun reset() {
        scoreA = 0
        scoreB = 0
        running = false
        onScoreChanged(scoreA, scoreB)
    }

    fun addScore(player: Char, points: Int) {
        if (!running) return
        val safePoints = points.coerceIn(1, 2)
        if (player == 'A') scoreA += safePoints else if (player == 'B') scoreB += safePoints else return
        onScoreChanged(scoreA, scoreB)
        onAnnouncement("プレイヤー $player、${safePoints}ポイント。${scoreA}対${scoreB}")
        if (scoreA >= targetScore || scoreB >= targetScore) {
            running = false
            val winner = if (scoreA >= targetScore) 'A' else 'B'
            onAnnouncement("ゲーム。プレイヤー $winner の勝ち。${scoreA}対${scoreB}")
            onGameOver(winner, scoreA, scoreB)
        }
    }
}
