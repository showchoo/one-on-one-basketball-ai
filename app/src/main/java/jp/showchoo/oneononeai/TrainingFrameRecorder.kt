package jp.showchoo.oneononeai

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Build
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

/**
 * Saves UNLABELED, full-resolution camera frames for later human annotation.
 * Recording and ZIP exporting require explicit button presses. No network uploads.
 */
class TrainingFrameRecorder(context: Context) {
    companion object {
        private const val FRAME_INTERVAL_MS = 500L
        private const val MAX_FRAMES = 1200
        private const val MAX_DURATION_MS = 5L * 60L * 1000L
        private const val MAX_BYTES = 250L * 1024L * 1024L
        private const val JPEG_QUALITY = 88
    }

    private data class Session(
        val dir: File,
        val saved: AtomicInteger = AtomicInteger(),
        val bytes: AtomicLong = AtomicLong(),
        val failures: AtomicInteger = AtomicInteger()
    )

    private val root = File(context.getExternalFilesDir(null) ?: context.filesDir, "training_frames")
    private val exportDir = File(context.getExternalFilesDir(null) ?: context.filesDir, "training_exports")
    private val worker = Executors.newSingleThreadExecutor()
    private val writing = AtomicBoolean(false)
    private val deviceName = "${Build.MANUFACTURER} ${Build.MODEL}".replace(",", " ")
    @Volatile private var session: Session? = null
    @Volatile var isRecording: Boolean = false
        private set
    private var lastCaptureElapsedMs = Long.MIN_VALUE
    private var firstFrameElapsedMs = Long.MIN_VALUE
    @Volatile var automaticStopReason: String? = null
        private set

    val savedCount: Int get() = session?.saved?.get() ?: 0
    val failureCount: Int get() = session?.failures?.get() ?: 0
    val hasSession: Boolean get() = session != null

    @Synchronized
    fun start(): File {
        check(!isRecording) { "Recording is already active" }
        root.mkdirs()
        val name = SimpleDateFormat("yyyyMMdd_HHmmss_SSS", Locale.US).format(Date()) +
            "_${System.nanoTime().toString().takeLast(5)}"
        val dir = File(root, "session_$name")
        if (!dir.mkdirs()) throw IOException("Cannot create training session")
        if (!File(dir, "images").mkdirs()) throw IOException("Cannot create image folder")
        File(dir, "frames.csv").writeText(
            "filename,timestamp_utc_ms,capture_elapsed_ms,image_width,image_height," +
                "rotation_applied_deg,device\n"
        )
        File(dir, "README.txt").writeText(
            "Training images for 1on1 Basketball AI\n" +
                "Each JPEG is a full camera frame rotated into its upright display orientation.\n" +
                "images/ contains UNLABELED images; no basketball bounding boxes are provided.\n" +
                "frames.csv gives timestamps and rotation. Coordinates for later labels " +
                "must be measured on each saved JPEG.\n" +
                "Captured only after explicitly enabling collection in the app.\n" +
                "Device: $deviceName\n" +
                "Interval: ${FRAME_INTERVAL_MS}ms; JPEG quality: $JPEG_QUALITY; " +
                "limit: 5 minutes / $MAX_FRAMES frames / 250 MiB.\n"
        )
        session = Session(dir)
        lastCaptureElapsedMs = Long.MIN_VALUE
        firstFrameElapsedMs = Long.MIN_VALUE
        automaticStopReason = null
        isRecording = true
        return dir
    }

    @Synchronized
    fun stop(): Int {
        isRecording = false
        return savedCount
    }

    /**
     * Stops capture and reports the final JPEG count only after all queued
     * JPEG writes and frames.csv appends have finished on the serial worker.
     * This confirms local device persistence, NOT cloud backup.
     */
    @Synchronized
    fun finishCapture(onComplete: (Result<Int>) -> Unit) {
        isRecording = false
        val current = session
        if (current == null) {
            onComplete(Result.failure(IllegalStateException("No training session")))
            return
        }
        runCatching {
            worker.execute {
                onComplete(Result.success(current.saved.get()))
            }
        }.onFailure { onComplete(Result.failure(it)) }
    }

    /**
     * Called from the CameraX analyzer. At most one bitmap copy is in flight;
     * JPEG compression and disk I/O take place off the analyzer thread.
     */
    @Synchronized
    fun captureIfDue(bitmap: Bitmap, rotationDegrees: Int, captureElapsedMs: Long) {
        val current = session ?: return
        if (!isRecording) return
        if (firstFrameElapsedMs == Long.MIN_VALUE) {
            firstFrameElapsedMs = captureElapsedMs
        }
        if (captureElapsedMs - firstFrameElapsedMs >= MAX_DURATION_MS) {
            isRecording = false
            automaticStopReason = "FIVE_MINUTES"
            return
        }
        if (current.saved.get() >= MAX_FRAMES || current.bytes.get() >= MAX_BYTES) {
            isRecording = false
            automaticStopReason = "SIZE_OR_FRAME_LIMIT"
            return
        }
        if (writing.get()) return
        if (lastCaptureElapsedMs != Long.MIN_VALUE &&
            captureElapsedMs - lastCaptureElapsedMs < FRAME_INTERVAL_MS
        ) return
        if (!writing.compareAndSet(false, true)) return

        val snapshot = try {
            bitmap.copy(Bitmap.Config.ARGB_8888, false)
                ?: throw IOException("Could not copy camera image")
        } catch (_: Exception) {
            current.failures.incrementAndGet()
            writing.set(false)
            return
        }
        lastCaptureElapsedMs = captureElapsedMs
        val utcMs = System.currentTimeMillis()
        val rotation = ((rotationDegrees % 360) + 360) % 360
        try {
            worker.execute {
                try {
                    val upright = if (rotation == 0) {
                        snapshot
                    } else {
                        Bitmap.createBitmap(
                            snapshot, 0, 0, snapshot.width, snapshot.height,
                            Matrix().apply { postRotate(rotation.toFloat()) }, true
                        )
                    }
                    try {
                        val index = current.saved.get() + 1
                        val fileName = "frame_%05d_%d.jpg".format(Locale.US, index, utcMs)
                        val output = File(File(current.dir, "images"), fileName)
                        try {
                            FileOutputStream(output).use { stream ->
                                if (!upright.compress(Bitmap.CompressFormat.JPEG, JPEG_QUALITY, stream)) {
                                    throw IOException("JPEG compression returned false")
                                }
                            }
                            // Metadata is only appended after the JPEG was completely written.
                            File(current.dir, "frames.csv").appendText(
                                "images/$fileName,$utcMs,$captureElapsedMs," +
                                    "${upright.width},${upright.height},$rotation,$deviceName\n"
                            )
                            current.bytes.addAndGet(output.length())
                            current.saved.incrementAndGet()
                        } catch (e: Exception) {
                            output.delete()
                            throw e
                        }
                    } finally {
                        if (upright !== snapshot && !upright.isRecycled) upright.recycle()
                    }
                } catch (_: Exception) {
                    current.failures.incrementAndGet()
                } finally {
                    if (!snapshot.isRecycled) snapshot.recycle()
                    writing.set(false)
                }
            }
        } catch (_: Exception) {
            if (!snapshot.isRecycled) snapshot.recycle()
            current.failures.incrementAndGet()
            writing.set(false)
        }
    }

    /**
     * Stops capture, waits for pending image writes in the serial executor,
     * then streams a ZIP without loading JPEG files into memory.
     */
    @Synchronized
    fun exportZip(onComplete: (Result<File>) -> Unit) {
        isRecording = false
        val current = session
        if (current == null) {
            onComplete(Result.failure(IllegalStateException("No training session")))
            return
        }
        worker.execute {
            val result = runCatching {
                if (current.saved.get() == 0) {
                    throw IllegalStateException("No frames captured yet")
                }
                exportDir.mkdirs()
                val zipFile = File(exportDir, "${current.dir.name}.zip")
                val partial = File(exportDir, "${current.dir.name}.zip.part")
                try {
                    ZipOutputStream(FileOutputStream(partial).buffered()).use { zip ->
                        current.dir.walkTopDown()
                            .filter { it.isFile }
                            .sortedBy { it.relativeTo(current.dir).invariantSeparatorsPath }
                            .forEach { file ->
                                val entryName = file.relativeTo(current.dir).invariantSeparatorsPath
                                zip.putNextEntry(ZipEntry(entryName))
                                FileInputStream(file).use { input -> input.copyTo(zip) }
                                zip.closeEntry()
                            }
                    }
                    if (zipFile.exists() && !zipFile.delete()) {
                        throw IOException("Cannot replace old ZIP")
                    }
                    if (!partial.renameTo(zipFile)) throw IOException("ZIP rename failed")
                    zipFile
                } catch (e: Exception) {
                    partial.delete()
                    throw e
                }
            }
            onComplete(result)
        }
    }

    @Synchronized
    fun close() {
        isRecording = false
        worker.shutdown()
    }
}
