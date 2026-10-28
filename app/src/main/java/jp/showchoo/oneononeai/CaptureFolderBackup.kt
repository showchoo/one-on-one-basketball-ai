package jp.showchoo.oneononeai

import android.content.Context
import android.content.Intent
import android.net.Uri
import androidx.documentfile.provider.DocumentFile
import java.io.File
import java.io.FileInputStream
import java.io.IOException
import java.util.concurrent.Executors

/**
 * Save ZIPs to a user-granted Google Drive document-tree URI.
 * Writing to Android's document provider is not proof of cloud sync.
 */
class CaptureFolderBackup(private val context: Context) {
    data class Receipt(val name: String, val bytes: Long, val date: String)
    private val prefs = context.getSharedPreferences("capture_folder_backup", Context.MODE_PRIVATE)
    private val worker = Executors.newSingleThreadExecutor()

    fun hasDestination(): Boolean = destinationUri() != null

    fun destinationUri(): Uri? {
        val uri = prefs.getString("tree_uri", null)?.let(Uri::parse) ?: return null
        if (!isDrive(uri)) return null
        val granted = context.contentResolver.persistedUriPermissions.any {
            it.uri == uri && it.isWritePermission
        }
        return if (granted) uri else null
    }

    private fun isDrive(uri: Uri): Boolean =
        uri.scheme == "content" &&
            (uri.authority == "com.google.android.apps.docs.storage" ||
                uri.authority == "com.google.android.apps.docs.storage.legacy")

    /** A one-time, narrow folder grant; no full-Drive OAuth scope. */
    fun setDestination(uri: Uri) {
        require(isDrive(uri)) { "Googleドライブのフォルダを選択してください" }
        context.contentResolver.takePersistableUriPermission(
            uri, Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_GRANT_WRITE_URI_PERMISSION
        )
        val folder = DocumentFile.fromTreeUri(context, uri)
        require(folder != null && folder.isDirectory && folder.canWrite()) {
            "選択したGoogleドライブのフォルダには書き込めません"
        }
        prefs.edit().putString("tree_uri", uri.toString()).apply()
    }

    fun copy(zip: File, onComplete: (Result<Receipt>) -> Unit) {
        worker.execute {
            onComplete(runCatching {
                if (!zip.isFile || zip.length() == 0L) throw IOException("ZIPが見つかりません")
                val root = destinationUri()?.let { DocumentFile.fromTreeUri(context, it) }
                    ?: throw IOException("保存先フォルダの権限がありません")
                if (!root.isDirectory || !root.canWrite()) throw IOException("保存先に書き込めません")
                val stamp = Regex("^session_(\\d{8})").find(zip.name)?.groupValues?.get(1)
                    ?: throw IOException("撮影日を判別できません")
                val date = "${stamp.substring(0, 4)}-${stamp.substring(4, 6)}-${stamp.substring(6, 8)}"
                val folder = root.findFile(date) ?: root.createDirectory(date)
                    ?: throw IOException("撮影日フォルダを作れません")
                if (!folder.isDirectory || !folder.canWrite()) throw IOException("撮影日フォルダが書込不可")
                if (folder.findFile(zip.name) != null) {
                    throw IOException("同名ZIPが既にあります。上書きを避けました")
                }
                val dest = folder.createFile("application/zip", zip.name)
                    ?: throw IOException("保存先にZIPを作れません")
                try {
                    FileInputStream(zip).use { input ->
                        val output = context.contentResolver.openOutputStream(dest.uri, "w")
                            ?: throw IOException("書込ストリームを開けません")
                        output.use { input.copyTo(it, 128 * 1024) }
                    }
                    // Cloud providers may report 0 while syncing.
                    val size = dest.length()
                    if (size > 0 && size != zip.length()) {
                        throw IOException("サイズ不一致: $size / ${zip.length()}")
                    }
                } catch (e: Exception) {
                    runCatching { dest.delete() } // Only newly created partial backup.
                    throw e
                }
                Receipt(zip.name, zip.length(), date)
            })
        }
    }

    fun close() { worker.shutdown() }
}
