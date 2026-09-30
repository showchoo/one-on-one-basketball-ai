package jp.showchoo.oneononeai

import android.content.Context
import android.media.MediaPlayer
import android.net.Uri
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.util.zip.ZipInputStream

class McVoicePack(private val context: Context) {
    private val root = File(context.filesDir, "mc_voice_pack")
    private val manifestFile = File(root, "manifest.json")
    private var clips: Map<String, String> = emptyMap()
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

    fun has(key: String): Boolean = clipFile(key)?.exists() == true

    fun importZip(uri: Uri): Result<String> = runCatching {
        val temp = File(context.cacheDir, "mc_voice_pack_import")
        if (temp.exists()) temp.deleteRecursively()
        temp.mkdirs()

        context.contentResolver.openInputStream(uri)?.use { input ->
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
                    FileOutputStream(out).use { output -> zip.copyTo(output) }
                    zip.closeEntry()
                }
            }
        } ?: throw IllegalArgumentException("Cannot open voice pack")

        val manifest = File(temp, "manifest.json")
        if (!manifest.exists()) throw IllegalArgumentException("manifest.json not found")

        val json = JSONObject(manifest.readText())
        val clipObject = json.optJSONObject("clips")
            ?: throw IllegalArgumentException("clips not found")

        val parsed = mutableMapOf<String, String>()
        val keys = clipObject.keys()
        while (keys.hasNext()) {
            val key = keys.next()
            val fileName = clipObject.optString(key)
            val file = File(temp, fileName).canonicalFile
            if (!file.path.startsWith(temp.canonicalPath + File.separator) || !file.exists()) {
                throw IllegalArgumentException("Missing clip: $key")
            }
            parsed[key] = fileName
        }
        if (parsed.isEmpty()) throw IllegalArgumentException("No clips in voice pack")

        stop()
        if (root.exists()) root.deleteRecursively()
        root.mkdirs()
        temp.copyRecursively(root, overwrite = true)
        temp.deleteRecursively()
        reload()
        name
    }

    fun play(key: String): Boolean {
        val file = clipFile(key) ?: return false
        if (!file.exists()) return false
        stop()
        return playFile(file)
    }

    fun playSequence(vararg keys: String): Boolean {
        val playable = keys.filter { has(it) }
        if (playable.isEmpty()) return false
        stop()
        queue.addAll(playable)
        playNext()
        return true
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

    private fun reload() {
        clips = runCatching {
            if (!manifestFile.exists()) return@runCatching emptyMap()
            val obj = JSONObject(manifestFile.readText()).optJSONObject("clips")
                ?: return@runCatching emptyMap()
            buildMap {
                val keys = obj.keys()
                while (keys.hasNext()) {
                    val key = keys.next()
                    put(key, obj.optString(key))
                }
            }
        }.getOrDefault(emptyMap())
    }

    private fun clipFile(key: String): File? {
        val relative = clips[key] ?: return null
        val file = File(root, relative).canonicalFile
        if (!file.path.startsWith(root.canonicalPath + File.separator)) return null
        return file
    }

    private fun playNext() {
        val key = queue.removeFirstOrNull() ?: return
        val file = clipFile(key)
        if (file == null || !file.exists()) {
            playNext()
            return
        }
        playFile(file, continueQueue = true)
    }

    private fun playFile(file: File, continueQueue: Boolean = false): Boolean {
        return runCatching {
            player?.release()
            player = MediaPlayer().apply {
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
}
