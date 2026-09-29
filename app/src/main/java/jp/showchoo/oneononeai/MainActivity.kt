package jp.showchoo.oneononeai

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.os.SystemClock
import android.speech.tts.TextToSpeech
import android.util.Size
import android.view.WindowManager
import android.widget.Button
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity(), TextToSpeech.OnInitListener {
    private lateinit var previewView: PreviewView
    private lateinit var overlayView: OverlayView
    private lateinit var playerAText: TextView
    private lateinit var playerBText: TextView
    private lateinit var statusText: TextView
    private lateinit var perfText: TextView

    private lateinit var cameraExecutor: ExecutorService
    @Volatile private var detector: ObjectDetectorEngine? = null
    private var lastInferenceAt = 0L
    private var tts: TextToSpeech? = null

    private lateinit var game: GameEngine
    private lateinit var tracker: BasketballTracker

    private val cameraPermission = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) startCamera() else statusText.text = "カメラ権限が必要です"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        setContentView(R.layout.activity_main)

        previewView = findViewById(R.id.previewView)
        overlayView = findViewById(R.id.overlayView)
        playerAText = findViewById(R.id.playerA)
        playerBText = findViewById(R.id.playerB)
        statusText = findViewById(R.id.statusText)
        perfText = findViewById(R.id.perfText)
        previewView.scaleType = PreviewView.ScaleType.FIT_CENTER

        tts = TextToSpeech(this, this)
        cameraExecutor = Executors.newSingleThreadExecutor()

        game = GameEngine(
            targetScore = 10,
            onScoreChanged = { a, b -> updateScoreUi(a, b) },
            onAnnouncement = { speak(it) },
            onGameOver = { winner, a, b -> statusText.text = "GAME: $winner WIN  $a-$b" }
        )
        tracker = BasketballTracker { player, points ->
            runOnUiThread { game.addScore(player, points) }
        }

        bindControls()
        cameraExecutor.execute {
            try {
                detector = ObjectDetectorEngine(applicationContext)
                runOnUiThread { statusText.text = "AI準備完了 / リング位置を設定" }
            } catch (e: Exception) {
                runOnUiThread { statusText.text = "AI初期化失敗: ${e.message}" }
            }
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            cameraPermission.launch(Manifest.permission.CAMERA)
        }
    }

    private fun bindControls() {
        findViewById<Button>(R.id.calibrateHoopButton).setOnClickListener {
            statusText.text = "リング中央をタップ"
            overlayView.calibrateHoop { rect ->
                tracker.hoopRect = rect
                overlayView.setCalibration(tracker.hoopRect, tracker.threePointLine)
                statusText.text = "リング設定済み / 3Pラインを設定"
            }
        }
        findViewById<Button>(R.id.calibrateThreeButton).setOnClickListener {
            statusText.text = "3Pラインを左から5点タップ"
            overlayView.calibrateThreePointLine { points ->
                tracker.threePointLine = points
                overlayView.setCalibration(tracker.hoopRect, tracker.threePointLine)
                statusText.text = "3Pライン設定済み"
            }
        }
        findViewById<Button>(R.id.startButton).setOnClickListener {
            if (tracker.hoopRect == null) {
                statusText.text = "先にリング位置を設定してください"
            } else {
                tracker.resetSession()
                game.start()
                statusText.text = if (tracker.threePointLine.size >= 2) "GAME RUNNING / 自動1・2点" else "GAME RUNNING / 3P未設定なので1点固定"
            }
        }
        findViewById<Button>(R.id.resetButton).setOnClickListener {
            game.reset()
            tracker.resetSession()
            statusText.text = "リセットしました"
        }
        findViewById<Button>(R.id.a1Button).setOnClickListener { game.addScore('A', 1) }
        findViewById<Button>(R.id.a2Button).setOnClickListener { game.addScore('A', 2) }
        findViewById<Button>(R.id.b1Button).setOnClickListener { game.addScore('B', 1) }
        findViewById<Button>(R.id.b2Button).setOnClickListener { game.addScore('B', 2) }
    }

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get()
            val preview = Preview.Builder()
                .setTargetResolution(Size(1280, 720))
                .build()
                .also { it.setSurfaceProvider(previewView.surfaceProvider) }

            val analysis = ImageAnalysis.Builder()
                .setTargetResolution(Size(1280, 720))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()

            analysis.setAnalyzer(cameraExecutor) { image ->
                val now = SystemClock.uptimeMillis()
                if (now - lastInferenceAt < 180L) {
                    image.close()
                    return@setAnalyzer
                }
                lastInferenceAt = now
                val d = detector
                if (d == null) {
                    image.close()
                    return@setAnalyzer
                }
                try {
                    val bitmap = Bitmap.createBitmap(image.width, image.height, Bitmap.Config.ARGB_8888)
                    val buffer = image.planes[0].buffer
                    buffer.rewind()
                    bitmap.copyPixelsFromBuffer(buffer)
                    val result = d.detect(bitmap, image.imageInfo.rotationDegrees)
                    val snapshot = tracker.update(
                        result.detections,
                        result.imageWidth,
                        result.imageHeight,
                        SystemClock.uptimeMillis()
                    )
                    runOnUiThread {
                        overlayView.setCalibration(tracker.hoopRect, tracker.threePointLine)
                        overlayView.update(snapshot, result.imageWidth, result.imageHeight)
                        perfText.text = "AI ${result.inferenceMs} ms | ${result.detections.size} obj"
                        if (game.running) statusText.text = snapshot.status
                    }
                } catch (e: Exception) {
                    runOnUiThread { perfText.text = "AI error: ${e.javaClass.simpleName}" }
                } finally {
                    image.close()
                }
            }

            provider.unbindAll()
            provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
        }, ContextCompat.getMainExecutor(this))
    }

    private fun updateScoreUi(a: Int, b: Int) {
        playerAText.text = "A  $a"
        playerBText.text = "$b  B"
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            tts?.language = Locale.JAPAN
            tts?.setSpeechRate(1.05f)
        }
    }

    private fun speak(text: String) {
        tts?.speak(text, TextToSpeech.QUEUE_ADD, null, "score-${SystemClock.uptimeMillis()}")
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        tts?.stop()
        tts?.shutdown()
    }
}
