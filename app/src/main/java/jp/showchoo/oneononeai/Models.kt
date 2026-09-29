package jp.showchoo.oneononeai

import android.graphics.RectF

data class AiDetection(
    val label: String,
    val score: Float,
    val box: RectF
)

data class PlayerTrack(
    val id: Char,
    var box: RectF,
    var lastSeenMs: Long
) {
    val centerX: Float get() = (box.left + box.right) / 2f
    val centerY: Float get() = (box.top + box.bottom) / 2f
    val footX: Float get() = centerX
    val footY: Float get() = box.bottom
}

data class TrackerSnapshot(
    val playerA: RectF?,
    val playerB: RectF?,
    val ball: RectF?,
    val lastPossessor: Char?,
    val shotPlayer: Char?,
    val shotValue: Int,
    val status: String
)
