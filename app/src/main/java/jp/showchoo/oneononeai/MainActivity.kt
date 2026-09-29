package jp.showchoo.oneononeai

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.os.SystemClock
import android.speech.tts.TextToSpeech
import android.speech.tts.Voice
import android.util.Size
import android.view.WindowManager
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.LinearLayout
import android.widget.SeekBar
import android.widget.Spinner
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
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
    companion object {
        private const val PREFS_NAME = "commentary_voice"
        private const val PREF_VOICE_NAME = "voice_name"
        private const val PREF_SPEECH_RATE = "speech_rate"
        private const val PREF_PITCH = "pitch"
        private const val DEFAULT_SPEECH_RATE = 1.05f
        private const val DEFAULT_PITCH = 1.0f
    }

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
    private var ttsReady = false

    private lateinit var game: GameEngine
    private lateinit var tracker: BasketballTracker
    private lateinit var commentary: CommentaryEngine
    private lateinit var commentaryButton: Button

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

        commentary = CommentaryEngine(CommentaryMode.LIVE)
        game = GameEngine(
            targetScore = 10,
            onScoreChanged = { a, b -> updateScoreUi(a, b) },
            onGameStarted = { target ->
                commentary.onGameStart(target)?.let { speak(it) }
            },
            onScoreEvent = { event ->
                commentary.onScore(event)?.let { speak(it) }
            },
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
        commentaryButton = findViewById(R.id.commentaryButton)
        commentaryButton.text = commentary.mode.buttonLabel
        commentaryButton.setOnClickListener {
            commentary.mode = commentary.mode.next()
            commentaryButton.text = commentary.mode.buttonLabel
            if (commentary.mode == CommentaryMode.OFF) tts?.stop()
            statusText.text = commentary.mode.buttonLabel
        }

        findViewById<Button>(R.id.voiceSettingsButton).setOnClickListener {
            showVoiceSettings()
        }

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
            commentary.reset()
            tts?.stop()
            statusText.text = "リセットしました"
        }
        findViewById<Button>(R.id.a1Button).setOnClickListener { game.addScore('A', 1) }
        findViewById<Button>(R.id.a2Button).setOnClickListener { game.addScore('A', 2) }
        findViewById<Button>(R.id.b1Button).setOnClickListener { game.addScore('B', 1) }
        findViewById<Button>(R.id.b2Button).setOnClickListener { game.addScore('B', 2) }
    }

    private fun showVoiceSettings() {
        if (!ttsReady) {
            statusText.text = "音声エンジン準備中です"
            return
        }

        val prefs = getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        val currentRate = prefs.getFloat(PREF_SPEECH_RATE, DEFAULT_SPEECH_RATE)
        val currentPitch = prefs.getFloat(PREF_PITCH, DEFAULT_PITCH)
        val savedVoiceName = prefs.getString(PREF_VOICE_NAME, null)

        val allJapaneseVoices = tts?.voices
            ?.filter { it.locale.language == Locale.JAPANESE.language }
            ?.sortedBy { it.name }
            .orEmpty()

        val offlineVoices = allJapaneseVoices.filter { !it.isNetworkConnectionRequired }
        val voices = if (offlineVoices.isNotEmpty()) offlineVoices else allJapaneseVoices

        val container = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            val pad = (20 * resources.displayMetrics.density).toInt()
            setPadding(pad, pad / 2, pad, 0)
        }

        val voiceLabel = TextView(this).apply { text = "音声" }
        container.addView(voiceLabel)

        val voiceSpinner = Spinner(this)
        val voiceLabels = if (voices.isEmpty()) {
            listOf("端末の標準日本語音声")
        } else {
            voices.mapIndexed { index, voice ->
                val local = if (voice.isNetworkConnectionRequired) "オンライン" else "オフライン"
                "音声 ${index + 1}  [$local]"
            }
        }
        voiceSpinner.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, voiceLabels)
        if (voices.isNotEmpty()) {
            val currentIndex = voices.indexOfFirst { it.name == savedVoiceName || it.name == tts?.voice?.name }
            if (currentIndex >= 0) voiceSpinner.setSelection(currentIndex)
        }
        container.addView(voiceSpinner)

        val rateLabel = TextView(this)
        val rateSeek = SeekBar(this).apply {
            max = 100
            progress = rateToProgress(currentRate)
        }
        fun updateRateLabel() {
            rateLabel.text = "話す速さ: %.2fx".format(progressToRate(rateSeek.progress))
        }
        updateRateLabel()
        rateSeek.setOnSeekBarChangeListener(simpleSeekListener { updateRateLabel() })
        container.addView(rateLabel)
        container.addView(rateSeek)

        val pitchLabel = TextView(this)
        val pitchSeek = SeekBar(this).apply {
            max = 100
            progress = pitchToProgress(currentPitch)
        }
        fun updatePitchLabel() {
            pitchLabel.text = "声の高さ: %.2fx".format(progressToPitch(pitchSeek.progress))
        }
        updatePitchLabel()
        pitchSeek.setOnSeekBarChangeListener(simpleSeekListener { updatePitchLabel() })
        container.addView(pitchLabel)
        container.addView(pitchSeek)

        val testButton = Button(this).apply {
            text = "試聴"
            setOnClickListener {
                applyPreviewVoice(
                    voices.getOrNull(voiceSpinner.selectedItemPosition),
                    progressToRate(rateSeek.progress),
                    progressToPitch(pitchSeek.progress)
                )
                speak("実況音声のテストです。プレイヤーA、外から決めた！2ポイント！")
            }
        }
        container.addView(testButton)

        AlertDialog.Builder(this)
            .setTitle("実況音声設定")
            .setView(container)
            .setPositiveButton("保存") { _, _ ->
                val selectedVoice = voices.getOrNull(voiceSpinner.selectedItemPosition)
                val rate = progressToRate(rateSeek.progress)
                val pitch = progressToPitch(pitchSeek.progress)

                prefs.edit()
                    .putString(PREF_VOICE_NAME, selectedVoice?.name)
                    .putFloat(PREF_SPEECH_RATE, rate)
                    .putFloat(PREF_PITCH, pitch)
                    .apply()

                applySavedTtsSettings()
                statusText.text = "実況音声設定を保存しました"
            }
            .setNegativeButton("キャンセル") { _, _ ->
                applySavedTtsSettings()
            }
            .setOnCancelListener {
                applySavedTtsSettings()
            }
            .show()
    }

    private fun simpleSeekListener(onChanged: () -> Unit) = object : SeekBar.OnSeekBarChangeListener {
        override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) = onChanged()
        override fun onStartTrackingTouch(seekBar: SeekBar?) = Unit
        override fun onStopTrackingTouch(seekBar: SeekBar?) = Unit
    }

    private fun rateToProgress(rate: Float): Int =
        (((rate.coerceIn(0.70f, 1.50f) - 0.70f) / 0.80f) * 100f).toInt()

    private fun progressToRate(progress: Int): Float = 0.70f + (progress / 100f) * 0.80f

    private fun pitchToProgress(pitch: Float): Int =
        (((pitch.coerceIn(0.70f, 1.30f) - 0.70f) / 0.60f) * 100f).toInt()

    private fun progressToPitch(progress: Int): Float = 0.70f + (progress / 100f) * 0.60f

    private fun applyPreviewVoice(voice: Voice?, rate: Float, pitch: Float) {
        tts?.language = Locale.JAPAN
        if (voice != null) tts?.voice = voice
        tts?.setSpeechRate(rate)
        tts?.setPitch(pitch)
    }

    private fun applySavedTtsSettings() {
        val engine = tts ?: return
        val prefs = getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        val voiceName = prefs.getString(PREF_VOICE_NAME, null)
        val rate = prefs.getFloat(PREF_SPEECH_RATE, DEFAULT_SPEECH_RATE)
        val pitch = prefs.getFloat(PREF_PITCH, DEFAULT_PITCH)

        engine.language = Locale.JAPAN
        val selectedVoice = engine.voices
            ?.firstOrNull { it.name == voiceName && it.locale.language == Locale.JAPANESE.language }
        if (selectedVoice != null) engine.voice = selectedVoice
        engine.setSpeechRate(rate)
        engine.setPitch(pitch)
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
            ttsReady = true
            applySavedTtsSettings()
        } else {
            ttsReady = false
            statusText.text = "音声エンジンを初期化できませんでした"
        }
    }

    private fun speak(text: String) {
        tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, "commentary-${SystemClock.uptimeMillis()}")
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        tts?.stop()
        tts?.shutdown()
    }
}
