package jp.showchoo.oneononeai

import android.content.Context
import android.media.MediaPlayer

class StreetMcPlayer(
    private val context: Context
) {
    private var player: MediaPlayer? = null
    private var released = false

    fun playReadyTipoff() {
        playSequence(
            R.raw.mc_ready,
            R.raw.mc_tipoff
        )
    }

    fun playVictory() {
        playSequence(R.raw.mc_victory)
    }

    fun playDemo() {
        playSequence(
            R.raw.mc_ready,
            R.raw.mc_tipoff,
            R.raw.mc_three,
            R.raw.mc_and_one,
            R.raw.mc_slam_dunk,
            R.raw.mc_victory
        )
    }

    fun stop() {
        player?.setOnCompletionListener(null)
        runCatching { player?.stop() }
        player?.release()
        player = null
    }

    fun release() {
        released = true
        stop()
    }

    private fun playSequence(vararg rawIds: Int) {
        if (released || rawIds.isEmpty()) return
        stop()
        playAt(rawIds, 0)
    }

    private fun playAt(rawIds: IntArray, index: Int) {
        if (released || index !in rawIds.indices) return

        val next = MediaPlayer.create(context, rawIds[index]) ?: return
        player = next
        next.setOnCompletionListener {
            it.release()
            if (player === it) player = null
            playAt(rawIds, index + 1)
        }
        next.setOnErrorListener { mp, _, _ ->
            mp.release()
            if (player === mp) player = null
            playAt(rawIds, index + 1)
            true
        }
        next.start()
    }
}
