package jp.showchoo.oneononeai

import java.util.ArrayDeque
import kotlin.random.Random

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
    companion object {
        const val BASE_VARIANTS_PER_SCORE_TYPE = 10_000
        private const val RECENT_COMMENTARY_LIMIT = 30
        private const val MAX_REROLL_ATTEMPTS = 12
    }

    private var lastScorer: Char? = null
    private var scoringStreak = 0
    private val recentCommentary = ArrayDeque<String>()

    private val openers = listOf(
        "プレイヤー {player}、", "{player}、", "ここでプレイヤー {player}、", "ここで {player}、",
        "さあプレイヤー {player}、", "さあ {player}、", "スコアを動かす {player}、", "得点するのは {player}、",
        "プレイヤー {player}が、", "ここは {player}が、", "攻めるプレイヤー {player}、", "攻める {player}、",
        "ポイントを狙う {player}、", "スコアを伸ばしたい {player}、", "この場面で {player}、",
        "プレイヤー {player}の攻撃、", "ボールを持つ {player}、", "得点を狙う {player}、",
        "ここから {player}、", "プレイヤー {player}、ここで"
    )

    private val onePointCalls = listOf(
        "決めた！1ポイント！", "しっかり決めて1ポイント！", "確実に1点を追加！", "1ポイントをもぎ取った！",
        "得点成功！1ポイント！", "リングを射抜いて1ポイント！", "シュート成功、1ポイント！",
        "きっちり1ポイントを加えた！", "ここは落とさない、1ポイント！", "得点を奪った！1ポイント！",
        "1点を積み上げた！", "スコアを1つ伸ばした！", "ナイスフィニッシュ、1ポイント！",
        "確実な得点、1ポイント！", "1ポイントを追加した！", "シュートを沈めた！1ポイント！",
        "決め切った！1ポイント！", "得点をものにした！1ポイント！", "リングを捉えた！1ポイント！",
        "ポイント獲得、1点追加！", "1ポイント成功！", "この一本を決めた！1ポイント！",
        "しっかりスコア、1ポイント！", "得点を加えた！1ポイント！", "一本成功、1ポイント！"
    )

    private val twoPointCalls = listOf(
        "外から決めた！2ポイント！", "ロングレンジ成功！2ポイント！", "外のシュートを沈めた！2ポイント！",
        "遠い位置から決め切った！2ポイント！", "2ポイントを奪った！", "外から一気に2点追加！",
        "ロングショット成功、2ポイント！", "距離のある一本を決めた！2ポイント！", "外角から成功！2ポイント！",
        "2ポイントシュートを沈めた！", "外からリングを射抜いた！2ポイント！",
        "鮮やかなロングレンジ、2ポイント！", "外の一本をものにした！2ポイント！", "遠距離から得点！2ポイント！",
        "大きな2ポイントを加えた！", "外からしっかり決めた！2ポイント！", "ロングレンジを決め切った！2ポイント！",
        "2点を一気に積み上げた！", "外から得点成功！2ポイント！", "距離をものともせず2ポイント！",
        "2ポイント成功！", "外の一本が決まった！2ポイント！", "ロングショットを沈めて2点追加！",
        "外からスコア！2ポイント！", "遠いところから決めた！2ポイント！"
    )

    private val reactions = listOf(
        "ナイスショット！", "見事です！", "いい一本！", "これは大きい！", "しっかり決めました！",
        "鮮やかな得点！", "いいリズムです！", "確実に仕留めました！", "ここは強い！", "得点につなげました！",
        "スコアを伸ばします！", "この一本は効きます！", "いい形で決めました！", "きっちり得点！",
        "落ち着いて決めました！", "これはナイスプレー！", "得点を重ねます！", "リングを捉えました！",
        "いいシュートです！", "しっかりポイントを取りました！"
    )

    private val tieCalls = listOf(
        "{a}対{b}、同点！", "これで{a}対{b}、並びました！", "スコアは{a}対{b}、振り出しです！",
        "{a}対{b}、ゲームはイーブン！", "追いついた！{a}対{b}！", "ここで同点、{a}対{b}！",
        "スコアが並びました、{a}対{b}！", "{a}対{b}、再び同点です！"
    )

    private val leadChangeCalls = listOf(
        "逆転！{a}対{b}！", "ここでひっくり返した！{a}対{b}！", "リードが入れ替わった！{a}対{b}！",
        "{a}対{b}、逆転です！", "ついに前へ出た！{a}対{b}！", "ゲームが動いた！逆転して{a}対{b}！",
        "ここでリードチェンジ！{a}対{b}！", "スコアをひっくり返した！{a}対{b}！"
    )

    private val normalScoreCalls = listOf(
        "スコアは{a}対{b}。", "{a}対{b}です。", "これで{a}対{b}。", "現在{a}対{b}。",
        "スコア、{a}対{b}。", "{a}対{b}となりました。", "得点は{a}対{b}。", "これでスコアは{a}対{b}。"
    )

    private val gamePointCalls = listOf(
        "プレイヤー {player}、ゲームポイント！", "{player}、あと1点で勝利！", "ゲームポイントは {player}！",
        "{player}、勝利まであと1ポイント！", "次を取れば {player} の勝利！", "{player}、決着まであと1点！"
    )

    private val streakCalls = listOf(
        "{streak}連続得点！", "{player}、これで{streak}連続得点！", "{streak}本続けて決めています！",
        "{player}の連続得点、{streak}まで伸びました！", "止まらない {player}、{streak}連続得点！",
        "{player}、{streak}得点連続です！"
    )

    private val deuceCalls = listOf(
        "9対9。次の1本で決着！", "9対9！次を取った方が勝ち！", "ついに9対9、次の一本が勝負です！",
        "9対9の大接戦！次で決まります！", "スコアは9対9。勝負の一本へ！", "並んだ、9対9！次の得点で決着です！"
    )

    private val winCalls = listOf(
        "決まった！プレイヤー {player}、{a}対{b}で勝利！", "ゲームセット！{player}が{a}対{b}で勝ちました！",
        "勝負あり！プレイヤー {player}、{a}対{b}で勝利！", "フィニッシュ！{player}が{a}対{b}でゲームを取りました！",
        "決着！{a}対{b}、勝者はプレイヤー {player}！", "{player}が決め切った！{a}対{b}で勝利！",
        "これでゲーム終了！{player}、{a}対{b}で勝利です！", "最後も決めた！プレイヤー {player}が{a}対{b}で勝利！",
        "ゲームを制したのは {player}！最終スコア{a}対{b}！", "勝利の一本！{player}が{a}対{b}で決着をつけました！"
    )

    private val gameStartCalls = listOf(
        "ゲームスタート。{target}点先取です", "試合開始！{target}点先取で勝負です",
        "さあゲームスタート。先に{target}点を取った方が勝ちです", "1オン1、スタート！勝利条件は{target}点先取",
        "ゲーム開始。{target}点を先に取れば勝利です", "勝負開始！{target}点先取でいきます"
    )

    fun reset() {
        lastScorer = null
        scoringStreak = 0
        recentCommentary.clear()
    }

    fun onGameStart(targetScore: Int): String? {
        reset()
        return when (mode) {
            CommentaryMode.OFF -> null
            CommentaryMode.SCORE_ONLY -> "ゲームスタート"
            CommentaryMode.LIVE -> rememberAndReturn(
                pick(gameStartCalls).replace("{target}", targetScore.toString())
            )
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
            return variedCommentary { fillScoreAndPlayer(pick(winCalls), e) }
        }

        return variedCommentary {
            val previousLeader = leader(e.previousScoreA, e.previousScoreB)
            val currentLeader = leader(e.scoreA, e.scoreB)
            val opener = pick(openers).replace("{player}", e.player.toString())
            val scoringCall = pick(if (e.points == 2) twoPointCalls else onePointCalls)
            val reaction = pick(reactions)
            val parts = mutableListOf("$opener $scoringCall $reaction")

            parts += when {
                e.scoreA == 9 && e.scoreB == 9 && e.targetScore == 10 -> pick(deuceCalls)
                e.scoreA == e.scoreB -> fillScore(pick(tieCalls), e)
                previousLeader != null && currentLeader != null && previousLeader != currentLeader ->
                    fillScore(pick(leadChangeCalls), e)
                else -> fillScore(pick(normalScoreCalls), e)
            }

            val scorerScore = if (e.player == 'A') e.scoreA else e.scoreB
            if (scorerScore == e.targetScore - 1) {
                parts += pick(gamePointCalls).replace("{player}", e.player.toString())
            } else if (scoringStreak >= 3) {
                parts += pick(streakCalls)
                    .replace("{player}", e.player.toString())
                    .replace("{streak}", scoringStreak.toString())
            }
            parts.joinToString(" ")
        }
    }

    private fun variedCommentary(builder: () -> String): String {
        var candidate = builder()
        repeat(MAX_REROLL_ATTEMPTS) {
            if (!recentCommentary.contains(candidate)) return rememberAndReturn(candidate)
            candidate = builder()
        }
        return rememberAndReturn(candidate)
    }

    private fun rememberAndReturn(commentary: String): String {
        recentCommentary.addLast(commentary)
        while (recentCommentary.size > RECENT_COMMENTARY_LIMIT) recentCommentary.removeFirst()
        return commentary
    }

    private fun fillScore(template: String, e: ScoreEvent): String =
        template.replace("{a}", e.scoreA.toString()).replace("{b}", e.scoreB.toString())

    private fun fillScoreAndPlayer(template: String, e: ScoreEvent): String =
        fillScore(template, e).replace("{player}", e.player.toString())

    private fun <T> pick(items: List<T>): T = items[Random.nextInt(items.size)]

    private fun updateStreak(player: Char) {
        if (lastScorer == player) scoringStreak += 1
        else {
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
