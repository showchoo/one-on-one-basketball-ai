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
        "{player}、いった！", "{player}、攻める！", "{player}、ここで仕掛ける！", "{player}、アタック！",
        "{player}、迷いなし！", "{player}、勝負！", "{player}、ここだ！", "{player}、一気にいく！",
        "{player}、流れを取りにいく！", "{player}、強気だ！", "{player}、仕留めにいく！", "{player}、狙ってる！",
        "{player}、そのまま行く！", "{player}、止まらない！", "{player}、ここで一本！", "{player}、プルアップ！",
        "{player}、バケツ狙い！", "{player}、ここで決めるか！", "{player}、アグレッシブ！", "{player}、レッツゴー！"
    )

    private val onePointCalls = listOf(
        "決めた！ワン！", "バケツ！1ポイント！", "沈めた！ワン！", "ナイスフィニッシュ！1ポイント！",
        "ねじ込んだ！1ポイント！", "落とさない！ワン！", "クリーン！1ポイント！", "きっちりワン！",
        "入れた！1ポイント！", "フィニッシュ！ワン！", "強い！1ポイント！", "この一本は入る！ワン！",
        "決め切った！1ポイント！", "バスケット！ワン！", "取った！1ポイント！", "一本返した！ワン！",
        "そのまま沈めた！1ポイント！", "ナイスバケツ！ワン！", "ここは確実！1ポイント！", "リングイン！ワン！",
        "ワン追加！", "一本ゲット！1ポイント！", "スコア！ワン！", "決まった！1ポイント！", "しっかりバケツ！ワン！"
    )

    private val twoPointCalls = listOf(
        "外からドン！ツー！", "バケツ！2ポイント！", "ロング、沈めた！ツー！", "ディープから決めた！2ポイント！",
        "でかい！ツー！", "外から刺した！2ポイント！", "プルアップ、入った！ツー！", "レンジ関係なし！2ポイント！",
        "外が落ちた！ツー！", "ロングバケツ！2ポイント！", "ディープ！決めた！ツー！", "そこから入れる！2ポイント！",
        "外からクリーン！ツー！", "ツーを奪った！", "ロングレンジ、バケツ！2ポイント！", "遠くても関係ない！ツー！",
        "外から一発！2ポイント！", "ツー追加！", "ディープレンジ成功！2ポイント！", "外から沈めた！ツー！",
        "ビッグショット！2ポイント！", "レンジ広い！ツー！", "外の一本、決まった！2ポイント！", "ツー！でかい！",
        "ロングを刺した！2ポイント！"
    )

    private val reactions = listOf(
        "ナイス！", "でかい！", "いいね！", "クリーン！", "これは効く！",
        "バケツ！", "熱い！", "止められない！", "強い！", "その一本！",
        "いいリズム！", "乗ってきた！", "クラッチ！", "いい仕事！", "落ち着いてる！",
        "その調子！", "流れ来てる！", "ナイスバケツ！", "キレてる！", "レッツゴー！"
    )

    private val tieCalls = listOf(
        "{a}対{b}、タイ！", "{a}対{b}、並んだ！", "{a}対{b}、イーブン！",
        "追いついた！{a}対{b}！", "ここでタイ！{a}対{b}！", "{a}対{b}、振り出し！",
        "スコア並んだ、{a}対{b}！", "{a}対{b}、まだ分からない！"
    )

    private val leadChangeCalls = listOf(
        "逆転！{a}対{b}！", "ひっくり返した！{a}対{b}！", "リード奪った！{a}対{b}！",
        "{a}対{b}、前に出た！", "ここでチェンジ！{a}対{b}！", "流れ変えた！{a}対{b}！",
        "逆転バケツ！{a}対{b}！", "スコアを返した！{a}対{b}！"
    )

    private val normalScoreCalls = listOf(
        "{a}対{b}！", "スコア、{a}対{b}！", "これで{a}対{b}！", "現在{a}対{b}！",
        "{a}対{b}、まだ続く！", "スコアは{a}対{b}！", "{a}対{b}、勝負はここから！", "これでスコア{a}対{b}！"
    )

    private val gamePointCalls = listOf(
        "{player}、ゲームポイント！", "{player}、あと一本！", "次で終わるぞ、{player}！",
        "{player}、マッチポイント！", "{player}、フィニッシュまであとワン！", "次を取れば{player}！"
    )

    private val streakCalls = listOf(
        "{streak}連続！止まらない！", "{player}、{streak}連続バケツ！", "{streak}本連続！熱い！",
        "{player}、完全に乗ってる！{streak}連続！", "オンファイヤー！{player}、{streak}連続！",
        "{player}、{streak}連続！誰か止めろ！"
    )

    private val deuceCalls = listOf(
        "9対9！次で終わる！", "9対9！ラストバケツ！", "9対9、次の一本が全部！",
        "タイゲーム！9対9！", "9対9！クラッチタイム！", "並んだ！9対9、次で決着！"
    )

    private val winCalls = listOf(
        "ゲーム！{player}が取った！{a}対{b}！", "終わり！{player}、{a}対{b}！",
        "ゲームセット！{player}の勝ち！{a}対{b}！", "クラッチ！{player}が締めた！{a}対{b}！",
        "フィニッシュ！勝者{player}！{a}対{b}！", "{player}が持っていった！{a}対{b}！",
        "決着！{player}！最終{a}対{b}！", "ラストバケツ！{player}の勝ち！{a}対{b}！",
        "勝負あり！{player}がゲームを取った！{a}対{b}！", "That's game！{player}！{a}対{b}！"
    )

    private val gameStartCalls = listOf(
        "レッツゴー！{target}点先取！", "1オン1、スタート！{target}まで！",
        "チェックボール！{target}点先取！", "さあやろう！先に{target}！",
        "ゲームオン！{target}点先取！", "ストリート1オン1、いくぞ！{target}まで！"
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
