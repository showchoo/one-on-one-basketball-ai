package jp.showchoo.oneononeai

class GameEngine(
    private val targetScore: Int = 10,
    private val onScoreChanged: (Int, Int) -> Unit,
    private val onGameStarted: (Int) -> Unit,
    private val onScoreEvent: (ScoreEvent) -> Unit,
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
        onGameStarted(targetScore)
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
        val previousA = scoreA
        val previousB = scoreB

        when (player) {
            'A' -> scoreA += safePoints
            'B' -> scoreB += safePoints
            else -> return
        }

        onScoreChanged(scoreA, scoreB)

        val gameOver = scoreA >= targetScore || scoreB >= targetScore
        onScoreEvent(
            ScoreEvent(
                player = player,
                points = safePoints,
                previousScoreA = previousA,
                previousScoreB = previousB,
                scoreA = scoreA,
                scoreB = scoreB,
                targetScore = targetScore,
                gameOver = gameOver
            )
        )

        if (gameOver) {
            running = false
            val winner = if (scoreA >= targetScore) 'A' else 'B'
            onGameOver(winner, scoreA, scoreB)
        }
    }
}
