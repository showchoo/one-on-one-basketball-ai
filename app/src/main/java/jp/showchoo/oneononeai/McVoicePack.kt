package jp.showchoo.oneononeai

import android.content.Context
import android.media.AudioAttributes
import android.media.MediaPlayer
import android.net.Uri
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.util.zip.ZipInputStream
import kotlin.random.Random

class McVoicePack(private val context: Context) {
    private val root = File(context.filesDir, "mc_voice_pack")
    private val manifestFile = File(root, "manifest.json")
    private var clips: Map<String, List<String>> = emptyMap()
    private var player: MediaPlayer? = null
    private val queue = ArrayDeque<String>()

    init {
        reload()
    }

    val name: String
        get() = runCatching {
            if (!manifestFile.exists()) return@runCatching "MC Voice"
            JSONObject(manifestFile.readText()).optString("name", "MC Voice")
        }.getOrDefault("MC Voice")

    val clipCount: Int
        get() = clips.values.sumOf { it.size }

    fun has(key: String): Boolean = clipFiles(key).any { it.exists() }

    fun importZip(uri: Uri): Result<String> = runCatching {
        context.contentResolver.openInputStream(uri)?.use { input ->
            installFromZipStream(input)
        } ?: throw IllegalArgumentException("Cannot open voice pack")
        name
    }

    fun play(key: String): Boolean {
        val file = pickFile(key) ?: return false
        stop()
        return playFile(file)
    }

    fun playSequence(vararg keys: String): Boolean {
        val playable = keys.mapNotNull { key ->
            pickFile(key)?.absolutePath
        }
        if (playable.isEmpty()) return false

        stop()
        queue.addAll(playable)
        playNext()
        return true
    }

    fun playGameStart(): Boolean =
        playSequence("are_you_ready", "tip_off")

    fun playScoreEvent(event: ScoreEvent): Boolean {
        if (event.gameOver) {
            return playFirstAvailable(
                "game_over_" + event.player.lowercaseChar(),
                "game_over"
            )
        }

        val wasTied = event.previousScoreA == event.previousScoreB
        val isTied = event.scoreA == event.scoreB
        val previousLeader = leader(event.previousScoreA, event.previousScoreB)
        val currentLeader = leader(event.scoreA, event.scoreB)
        val scorerScore = if (event.player == 'A') event.scoreA else event.scoreB

        if (scorerScore == event.targetScore - 1) {
            if (playFirstAvailable(
                    "game_point_" + event.player.lowercaseChar(),
                    "game_point"
                )
            ) return true
        }

        if (isTied && !wasTied) {
            if (playFirstAvailable("tie")) return true
        }

        if (
            previousLeader != null &&
            currentLeader != null &&
            previousLeader != currentLeader
        ) {
            if (playFirstAvailable(
                    "lead_change_" + event.player.lowercaseChar(),
                    "lead_change"
                )
            ) return true
        }

        return playFirstAvailable(
            "score_" + event.player.lowercaseChar() + "_" + event.points,
            "score_" + event.points,
            "score_" + event.player.lowercaseChar(),
            "score"
        )
    }

    fun previewAll(): Boolean =
        playSequence(
            "are_you_ready",
            "tip_off",
            "three",
            "and_one",
            "slam_dunk",
            "game_over"
        )

    fun stop() {
        queue.clear()
        player?.setOnCompletionListener(null)
        player?.release()
        player = null
    }

    fun release() = stop()

    private fun playFirstAvailable(vararg keys: String): Boolean {
        for (key in keys) {
            if (play(key)) return true
        }
        return false
    }

    private fun installFromZipStream(input: InputStream) {
        val temp = File(context.cacheDir, "mc_voice_pack_import")
        if (temp.exists()) temp.deleteRecursively()
        temp.mkdirs()

        var extractedBytes = 0L
        ZipInputStream(input).use { zip ->
            while (true) {
                val entry = zip.nextEntry ?: break
                val safeName = entry.name.replace("\\", "/")
                if (safeName.startsWith("/") || safeName.contains("../")) {
                    throw IllegalArgumentException("Invalid voice pack path")
                }
                if (entry.isDirectory) {
                    zip.closeEntry()
                    continue
                }

                val ext = safeName.substringAfterLast('.', "").lowercase()
                if (ext !in setOf("json", "mp3", "wav", "ogg", "m4a")) {
                    zip.closeEntry()
                    continue
                }

                val out = File(temp, safeName).canonicalFile
                if (!out.path.startsWith(temp.canonicalPath + File.separator)) {
                    throw IllegalArgumentException("Invalid voice pack path")
                }
                out.parentFile?.mkdirs()

                FileOutputStream(out).use { output ->
                    val buffer = ByteArray(32 * 1024)
                    while (true) {
                        val read = zip.read(buffer)
                        if (read <= 0) break
                        extractedBytes += read
                        if (extractedBytes > 100L * 1024L * 1024L) {
                            throw IllegalArgumentException("Voice pack too large")
                        }
                        output.write(buffer, 0, read)
                    }
                }
                zip.closeEntry()
            }
        }

        val manifest = File(temp, "manifest.json")
        if (!manifest.exists()) throw IllegalArgumentException("manifest.json not found")

        val parsed = parseManifest(temp, manifest)
        if (parsed.isEmpty()) throw IllegalArgumentException("No clips in voice pack")

        stop()
        if (root.exists()) root.deleteRecursively()
        root.mkdirs()
        temp.copyRecursively(root, overwrite = true)
        temp.deleteRecursively()
        reload()
    }

    private fun parseManifest(base: File, manifest: File): Map<String, List<String>> {
        val json = JSONObject(manifest.readText())
        val clipObject = json.optJSONObject("clips")
            ?: throw IllegalArgumentException("clips not found")

        val parsed = mutableMapOf<String, List<String>>()
        val keys = clipObject.keys()

        while (keys.hasNext()) {
            val key = keys.next()
            val value = clipObject.opt(key)
            val names = when (value) {
                is String -> listOf(value)
                is JSONArray -> buildList {
                    for (i in 0 until value.length()) {
                        val item = value.optString(i)
                        if (item.isNotBlank()) add(item)
                    }
                }
                else -> emptyList()
            }

            val safeNames = names.filter { relative ->
                val file = File(base, relative).canonicalFile
                file.path.startsWith(base.canonicalPath + File.separator) && file.exists()
            }
            if (safeNames.isNotEmpty()) parsed[key] = safeNames
        }

        return parsed
    }

    private fun reload() {
        clips = runCatching {
            if (!manifestFile.exists()) return@runCatching emptyMap()
            parseManifest(root, manifestFile)
        }.getOrDefault(emptyMap())
    }

    private fun clipFiles(key: String): List<File> =
        clips[key].orEmpty().mapNotNull { relative ->
            runCatching {
                val file = File(root, relative).canonicalFile
                if (!file.path.startsWith(root.canonicalPath + File.separator)) null else file
            }.getOrNull()
        }

    private fun pickFile(key: String): File? {
        val files = clipFiles(key).filter { it.exists() }
        if (files.isEmpty()) return null
        return files[Random.nextInt(files.size)]
    }

    private fun playNext() {
        val path = queue.removeFirstOrNull() ?: return
        val file = File(path)
        if (!file.exists()) {
            playNext()
            return
        }
        playFile(file, continueQueue = true)
    }

    private fun playFile(file: File, continueQueue: Boolean = false): Boolean {
        return runCatching {
            player?.release()
            player = MediaPlayer().apply {
                setAudioAttributes(
                    AudioAttributes.Builder()
                        .setUsage(AudioAttributes.USAGE_GAME)
                        .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                        .build()
                )
                setDataSource(file.absolutePath)
                setOnCompletionListener {
                    it.release()
                    if (player === it) player = null
                    if (continueQueue) playNext()
                }
                prepare()
                start()
            }
            true
        }.getOrDefault(false)
    }

    private fun leader(a: Int, b: Int): Char? = when {
        a > b -> 'A'
        b > a -> 'B'
        else -> null
    }
}
