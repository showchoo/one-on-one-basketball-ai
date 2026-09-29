package jp.showchoo.oneononeai

import android.content.Context
import android.graphics.RectF
import android.os.Build
import java.io.BufferedWriter
import java.io.File
import java.io.FileWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class DebugLogger(context: Context) {
    private val lock = Any()
    private val logDir = File(context.getExternalFilesDir(null) ?: context.filesDir, "debug_logs")
    private val timestampFormat = SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS", Locale.US)
    private val fileNameFormat = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US)

    val currentFile: File
    private var writer: BufferedWriter? = null

    init {
        logDir.mkdirs()
        currentFile = File(logDir, "basketball_debug_${fileNameFormat.format(Date())}.csv")
        writer = BufferedWriter(FileWriter(currentFile, true))
        writer?.write(
            "timestamp,event_type,status,inference_ms,detection_count," +
                "player_a,player_b,ball,last_possessor,shot_player,shot_value," +
                "score_a,score_b,points,detail\n"
        )
        logEvent(
            eventType = "APP_START",
            detail = "device=${Build.MANUFACTURER} ${Build.MODEL}; android=${Build.VERSION.RELEASE}"
        )
    }

    fun logFrame(
        snapshot: TrackerSnapshot,
        inferenceMs: Long,
        detectionCount: Int,
        scoreA: Int,
        scoreB: Int
    ) {
        writeRow(
            eventType = "FRAME",
            status = snapshot.status,
            inferenceMs = inferenceMs.toString(),
            detectionCount = detectionCount.toString(),
            playerA = rect(snapshot.playerA),
            playerB = rect(snapshot.playerB),
            ball = rect(snapshot.ball),
            lastPossessor = snapshot.lastPossessor?.toString().orEmpty(),
            shotPlayer = snapshot.shotPlayer?.toString().orEmpty(),
            shotValue = snapshot.shotValue.toString(),
            scoreA = scoreA.toString(),
            scoreB = scoreB.toString()
        )
    }

    fun logEvent(
        eventType: String,
        status: String = "",
        scoreA: Int? = null,
        scoreB: Int? = null,
        points: Int? = null,
        detail: String = ""
    ) {
        writeRow(
            eventType = eventType,
            status = status,
            scoreA = scoreA?.toString().orEmpty(),
            scoreB = scoreB?.toString().orEmpty(),
            points = points?.toString().orEmpty(),
            detail = detail
        )
    }

    fun flush() {
        synchronized(lock) {
            writer?.flush()
        }
    }

    fun close() {
        synchronized(lock) {
            writer?.flush()
            writer?.close()
            writer = null
        }
    }

    private fun writeRow(
        eventType: String,
        status: String = "",
        inferenceMs: String = "",
        detectionCount: String = "",
        playerA: String = "",
        playerB: String = "",
        ball: String = "",
        lastPossessor: String = "",
        shotPlayer: String = "",
        shotValue: String = "",
        scoreA: String = "",
        scoreB: String = "",
        points: String = "",
        detail: String = ""
    ) {
        val values = listOf(
            timestampFormat.format(Date()),
            eventType,
            status,
            inferenceMs,
            detectionCount,
            playerA,
            playerB,
            ball,
            lastPossessor,
            shotPlayer,
            shotValue,
            scoreA,
            scoreB,
            points,
            detail
        )

        synchronized(lock) {
            writer?.write(values.joinToString(",") { csv(it) })
            writer?.newLine()
            writer?.flush()
        }
    }

    private fun rect(r: RectF?): String {
        if (r == null) return ""
        return String.format(
            Locale.US,
            "%.4f|%.4f|%.4f|%.4f",
            r.left,
            r.top,
            r.right,
            r.bottom
        )
    }

    private fun csv(value: String): String {
        val escaped = value.replace("\"", "\"\"")
        return "\"$escaped\""
    }
}
