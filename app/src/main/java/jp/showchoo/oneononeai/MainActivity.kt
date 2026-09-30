package jp.showchoo.oneononeai

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.RectF
import android.os.Bundle
import android.os.SystemClock
import android.speech.tts.TextToSpeech
import android.speech.tts.Voice
import android.util.Size
import android.view.View
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
import androidx.core.content.FileProvider
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference

class MainActivity : AppCompatActivity(), TextToSpeech.OnInitListener {
    companion object {
        private const val PREFS_NAME = "commentary_voice"
        private const val PREF_VOICE_NAME = "voice_name"
        private const val PREF_SPEECH_RATE = "speech_rate"
        private const val PREF_PITCH = "pitch"
        private const val DEFAULT_SPEECH_RATE = 1.10f
        private const val DEFAULT_PITCH = 0.92f
    }

    private lateinit var previewView: PreviewView
    private lateinit var overlayView: OverlayView
    private lateinit var playerAText: TextView
    private lateinit var playerBText: TextView
    private lateinit var statusText: TextView
    private lateinit var perfText: TextView

    private data class YoloPacket(
        val result: ObjectDetectorEngine.Result,
        val captureTimeMs: Long,
        val receivedTimeMs: Long
    )

    private data class BallPacket(
        val result: BallRoiDetectorEngine.Result,
        val captureTimeMs: Long,
        val receivedTimeMs: Long,
        val reason: String,
        val motionCount: Int,
        val playersAtCapture: PlayerIdentityTracker.Snapshot,
        val motionAtCapture: List<MotionBallProposer.Proposal>,
        val hoopAtCapture: RectF?,
        val wasLockedAtCapture: Boolean
    )

    private data class BallValidationResult(
        val geometryAccepted: List<AiDetection>,
        val accepted: List<AiDetection>,
        val diagnostics: List<OverlayView.RawBallCandidate>,
        val dropSummary: String
    )

    private lateinit var cameraExecutor: ExecutorService
    private lateinit var yoloExecutor: ExecutorService
    private lateinit var ballExecutor: ExecutorService
    @Volatile private var detector: ObjectDetectorEngine? = null
    @Volatile private var ballDetector: BallRoiDetectorEngine? = null
    private var analysisBitmap: Bitmap? = null

    private lateinit var ballFusion: BallTrackFusion
    private lateinit var ballContextTracker: BallContextTracker
    private lateinit var ballSearchPlanner: BallSearchPlanner
    private lateinit var motionBallProposer: MotionBallProposer
    private lateinit var playerIdentityTracker: PlayerIdentityTracker
    private val yoloBusy = AtomicBoolean(false)
    private val ballBusy = AtomicBoolean(false)
    private val pendingYolo = AtomicReference<YoloPacket?>(null)
    private val pendingBall = AtomicReference<BallPacket?>(null)
    private var lastYoloLaunchAt = 0L
    private var lastBallLaunchAt = 0L
    private var lastLogAt = 0L
    private var lastUiAt = 0L
    private var latestYoloInferenceMs = 0L
    private var latestBallInferenceMs = 0L
    private var latestYoloDetectionCount = 0
    private var latestRawBallCandidates: List<OverlayView.RawBallCandidate> = emptyList()
    private var latestRawBallCandidatesAt = 0L

    private var tts: TextToSpeech? = null
    private var ttsReady = false

    private lateinit var game: GameEngine
    private lateinit var tracker: BasketballTracker
    private lateinit var commentary: CommentaryEngine
    private lateinit var commentaryButton: Button
    private lateinit var mcVoicePackButton: Button
    private lateinit var controlsToggleButton: Button
    private lateinit var advancedControls: LinearLayout
    private lateinit var debugLogger: DebugLogger
    private lateinit var mcVoicePack: McVoicePack

    private val cameraPermission = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) startCamera() else statusText.text = "カメラ権限が必要です"
    }

    private val mcPackPicker = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri == null) return@registerForActivityResult
        runCatching {
            contentResolver.takePersistableUriPermission(
                uri,
                Intent.FLAG_GRANT_READ_URI_PERMISSION
            )
        }
        val result = mcVoicePack.importZip(uri)
        result.onSuccess { packName ->
            updateMcVoiceButton()
            statusText.text = "MC音声: $packName"
            debugLogger.logEvent(
                "MC_VOICE_PACK_IMPORTED",
                detail = "name=$packName; clips=${mcVoicePack.clipCount}"
            )
            tts?.stop()
            mcVoicePack.previewAll()
        }.onFailure { error ->
            statusText.text = "MC音声パック読込失敗"
            debugLogger.logEvent("MC_VOICE_PACK_ERROR", detail = error.toString())
        }
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

        debugLogger = DebugLogger(applicationContext)
        mcVoicePack = McVoicePack(applicationContext)
        tts = TextToSpeech(this, this)
        cameraExecutor = Executors.newSingleThreadExecutor()
        yoloExecutor = Executors.newSingleThreadExecutor()
        ballExecutor = Executors.newSingleThreadExecutor()
        motionBallProposer = MotionBallProposer()
        ballSearchPlanner = BallSearchPlanner()
        ballFusion = BallTrackFusion { event ->
            debugLogger.logEvent(
                eventType = "BALL_FUSION_EVENT",
                scoreA = if (::game.isInitialized) game.scoreA else null,
                scoreB = if (::game.isInitialized) game.scoreB else null,
                detail = event
            )
        }
        ballContextTracker = BallContextTracker { event ->
            debugLogger.logEvent(
                eventType = "BALL_CONTEXT_EVENT",
                scoreA = if (::game.isInitialized) game.scoreA else null,
                scoreB = if (::game.isInitialized) game.scoreB else null,
                detail = event
            )
        }
        playerIdentityTracker = PlayerIdentityTracker { event ->
            debugLogger.logEvent(
                eventType = "PLAYER_TRACKER_EVENT",
                scoreA = if (::game.isInitialized) game.scoreA else null,
                scoreB = if (::game.isInitialized) game.scoreB else null,
                detail = event
            )
        }

        commentary = CommentaryEngine(CommentaryMode.LIVE)
        game = GameEngine(
            targetScore = 10,
            onScoreChanged = { a, b -> updateScoreUi(a, b) },
            onGameStarted = { target ->
                debugLogger.logEvent("GAME_START", scoreA = game.scoreA, scoreB = game.scoreB, detail = "target=$target")
                if (commentary.mode != CommentaryMode.OFF) {
                    tts?.stop()
                    val mcPlayed = mcVoicePack.playGameStart()
                    if (!mcPlayed) {
                        commentary.onGameStart(target)?.let { speak(it) }
                    }
                }
            },
            onScoreEvent = { event ->
                debugLogger.logEvent(
                    eventType = "SCORE_EVENT",
                    scoreA = event.scoreA,
                    scoreB = event.scoreB,
                    points = event.points,
                    detail = "player=${event.player}; gameOver=${event.gameOver}"
                )
                if (commentary.mode != CommentaryMode.OFF) {
                    tts?.stop()
                    val mcPlayed = mcVoicePack.playScoreEvent(event)
                    if (!mcPlayed) {
                        commentary.onScore(event)?.let { speak(it) }
                    } else {
                        debugLogger.logEvent(
                            "MC_VOICE_EVENT",
                            scoreA = event.scoreA,
                            scoreB = event.scoreB,
                            points = event.points,
                            detail = "player=${event.player}; gameOver=${event.gameOver}; pack=${mcVoicePack.name}"
                        )
                    }
                }
            },
            onGameOver = { winner, a, b ->
                debugLogger.logEvent("GAME_OVER", scoreA = a, scoreB = b, detail = "winner=$winner")
                statusText.text = "GAME: $winner WIN  $a-$b"
            }
        )
        tracker = BasketballTracker(
            onAutomaticScore = { player, points ->
                debugLogger.logEvent(
                    eventType = "AUTO_SCORE_REQUEST",
                    scoreA = game.scoreA,
                    scoreB = game.scoreB,
                    points = points,
                    detail = "player=$player"
                )
                runOnUiThread { game.addScore(player, points) }
            },
            onDebugEvent = { event ->
                debugLogger.logEvent(
                    eventType = "TRACKER_EVENT",
                    scoreA = game.scoreA,
                    scoreB = game.scoreB,
                    detail = event
                )
            }
        )

        bindControls()
        yoloExecutor.execute {
            try {
                detector = ObjectDetectorEngine(applicationContext)
                runOnUiThread {
                    statusText.text = "v0.5 SCENE AI 準備完了"
                }
            } catch (e: Exception) {
                runOnUiThread {
                    statusText.text = "Scene AI初期化失敗: ${e.message}"
                }
            }
        }

        ballExecutor.execute {
            try {
                ballDetector = BallRoiDetectorEngine(applicationContext)
                runOnUiThread {
                    statusText.text = "v0.5 BALL ROI AI 準備完了"
                }
            } catch (e: Exception) {
                runOnUiThread {
                    statusText.text = "Ball AI初期化失敗: ${e.message}"
                }
            }
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            cameraPermission.launch(Manifest.permission.CAMERA)
        }
    }

    private fun bindControls() {
        advancedControls = findViewById(R.id.advancedControls)
        controlsToggleButton = findViewById(R.id.controlsToggleButton)
        controlsToggleButton.setOnClickListener {
            val expand = advancedControls.visibility != View.VISIBLE
            setControlsExpanded(expand)
        }

        commentaryButton = findViewById(R.id.commentaryButton)
        commentaryButton.text = commentary.mode.buttonLabel
        commentaryButton.setOnClickListener {
            commentary.mode = commentary.mode.next()
            commentaryButton.text = commentary.mode.buttonLabel
            if (commentary.mode == CommentaryMode.OFF) {
                tts?.stop()
                mcVoicePack.stop()
            }
            debugLogger.logEvent("COMMENTARY_MODE", detail = commentary.mode.name)
            statusText.text = commentary.mode.buttonLabel
        }

        findViewById<Button>(R.id.voiceSettingsButton).setOnClickListener {
            showVoiceSettings()
        }

        mcVoicePackButton = findViewById(R.id.mcVoicePackButton)
        updateMcVoiceButton()
        mcVoicePackButton.setOnClickListener {
            showMcVoicePackMenu()
        }

        findViewById<Button>(R.id.debugLogButton).setOnClickListener {
            shareDebugLog()
        }

        findViewById<Button>(R.id.calibrateHoopButton).setOnClickListener {
            statusText.text = "リング中央をタップ"
            overlayView.calibrateHoop { rect ->
                tracker.setManualHoop(rect)
                debugLogger.logEvent("HOOP_CALIBRATED", detail = "rect=${rect.left}|${rect.top}|${rect.right}|${rect.bottom}")
                overlayView.setCalibration(tracker.hoopRect, tracker.threePointLine)
                statusText.text = "リング設定済み / 3Pラインを設定"
            }
        }
        findViewById<Button>(R.id.calibrateThreeButton).setOnClickListener {
            statusText.text = "3Pラインを左から5点タップ"
            overlayView.calibrateThreePointLine { points ->
                tracker.threePointLine = points
                debugLogger.logEvent(
                    "THREE_POINT_CALIBRATED",
                    detail = points.joinToString(";") { "${it.x}|${it.y}" }
                )
                overlayView.setCalibration(tracker.hoopRect, tracker.threePointLine)
                statusText.text = "3Pライン設定済み"
            }
        }
        findViewById<Button>(R.id.startButton).setOnClickListener {
            tracker.resetSession()
            ballFusion.reset()
            ballContextTracker.reset()
            ballSearchPlanner.reset()
            motionBallProposer.reset()
            playerIdentityTracker.reset()
            latestRawBallCandidates = emptyList()
            latestRawBallCandidatesAt = 0L
            debugLogger.logEvent(
                "START_PRESSED",
                detail = "threePointPoints=${tracker.threePointLine.size}; hoopPreset=${tracker.hoopRect != null}"
            )
            game.start()
            setControlsExpanded(false)
            statusText.text = when {
                tracker.hoopRect == null -> "LIVE / AIリング検出中"
                tracker.threePointLine.size >= 2 -> "LIVE / 自動1・2点判定"
                else -> "LIVE / 3P未設定・1点固定"
            }
        }
        findViewById<Button>(R.id.resetButton).setOnClickListener {
            debugLogger.logEvent("RESET_PRESSED", scoreA = game.scoreA, scoreB = game.scoreB)
            game.reset()
            tracker.resetSession()
            ballFusion.reset()
            ballContextTracker.reset()
            ballSearchPlanner.reset()
            motionBallProposer.reset()
            playerIdentityTracker.reset()
            commentary.reset()
            tts?.stop()
            setControlsExpanded(true)
            statusText.text = "SETUP / リセットしました"
        }
        findViewById<Button>(R.id.a1Button).setOnClickListener { addManualScore('A', 1) }
        findViewById<Button>(R.id.a2Button).setOnClickListener { addManualScore('A', 2) }
        findViewById<Button>(R.id.b1Button).setOnClickListener { addManualScore('B', 1) }
        findViewById<Button>(R.id.b2Button).setOnClickListener { addManualScore('B', 2) }
    }

    private fun setControlsExpanded(expanded: Boolean) {
        advancedControls.visibility = if (expanded) View.VISIBLE else View.GONE
        controlsToggleButton.text = if (expanded) "操作を閉じる" else "操作を開く"
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

        val voices = sortedJapaneseVoices()

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
                val source = if (voice.isNetworkConnectionRequired) "HUMAN / 高品質通信" else "OFFLINE"
                val quality = when {
                    voice.quality >= 500 -> "最高"
                    voice.quality >= 400 -> "高"
                    voice.quality >= 300 -> "標準"
                    else -> "軽量"
                }
                "音声 ${index + 1}  [$source・品質:$quality]"
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

        val presetRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
        }

        val humanPresetButton = Button(this).apply {
            text = "HUMAN"
            setOnClickListener {
                val bestHuman = bestHumanVoice(voices)
                val bestHumanIndex = voices.indexOf(bestHuman)
                if (bestHumanIndex >= 0) voiceSpinner.setSelection(bestHumanIndex)
                rateSeek.progress = rateToProgress(1.10f)
                pitchSeek.progress = pitchToProgress(0.92f)
                applyPreviewVoice(bestHuman, 1.10f, 0.92f)
                speak("エー、外からドン。ツー。でかい一本。")
            }
        }
        presetRow.addView(humanPresetButton)

        val streetPresetButton = Button(this).apply {
            text = "STREET"
            setOnClickListener {
                rateSeek.progress = rateToProgress(1.18f)
                pitchSeek.progress = pitchToProgress(0.88f)
                applyPreviewVoice(
                    voices.getOrNull(voiceSpinner.selectedItemPosition),
                    1.18f,
                    0.88f
                )
                speak("エー、いった！外からドン！ツー！でかい！")
            }
        }
        presetRow.addView(streetPresetButton)

        val arenaPresetButton = Button(this).apply {
            text = "ARENA"
            setOnClickListener {
                rateSeek.progress = rateToProgress(1.08f)
                pitchSeek.progress = pitchToProgress(0.96f)
                applyPreviewVoice(
                    voices.getOrNull(voiceSpinner.selectedItemPosition),
                    1.08f,
                    0.96f
                )
                speak("エー、決めた！ツーポイント！ナイスショット！")
            }
        }
        presetRow.addView(arenaPresetButton)
        container.addView(presetRow)

        val mcTestButton = Button(this).apply {
            text = "MC音声パックを試聴"
            setOnClickListener {
                tts?.stop()
                if (!mcVoicePack.previewAll()) {
                    mcPackPicker.launch(arrayOf("application/zip", "application/octet-stream"))
                }
            }
        }
        container.addView(mcTestButton)

        val testButton = Button(this).apply {
            text = "現在のTTS設定を試聴"
            setOnClickListener {
                applyPreviewVoice(
                    voices.getOrNull(voiceSpinner.selectedItemPosition),
                    progressToRate(rateSeek.progress),
                    progressToPitch(pitchSeek.progress)
                )
                speak("エー、攻める！バケツ！ツー！流れ来てる！")
            }
        }
        container.addView(testButton)

        AlertDialog.Builder(this)
            .setTitle("実況音声 / HUMAN・STREET")
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
                debugLogger.logEvent(
                    "VOICE_SETTINGS_SAVED",
                    detail = "voice=${selectedVoice?.name.orEmpty()}; rate=$rate; pitch=$pitch"
                )
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

    private fun updateMcVoiceButton() {
        if (!::mcVoicePackButton.isInitialized) return
        val scoreReady = mcVoicePack.has("score_1") && mcVoicePack.has("score_2")
        mcVoicePackButton.text = if (scoreReady) "MC: ON" else "MC読込"
    }

    private fun showMcVoicePackMenu() {
        val hasPack = mcVoicePack.has("are_you_ready") ||
            mcVoicePack.has("game_over") ||
            mcVoicePack.has("tip_off")

        if (!hasPack) {
            mcPackPicker.launch(arrayOf("application/zip", "application/octet-stream"))
            return
        }

        AlertDialog.Builder(this)
            .setTitle("MC音声パック")
            .setMessage(
                "${mcVoicePack.name}\n${mcVoicePack.clipCount} clips loaded\n" +
                    if (mcVoicePack.has("score_1") && mcVoicePack.has("score_2")) {
                        "通常得点もMC音声で再生"
                    } else {
                        "通常得点はTTSへフォールバック"
                    }
            )
            .setPositiveButton("試聴") { _, _ ->
                tts?.stop()
                mcVoicePack.previewAll()
            }
            .setNeutralButton("変更") { _, _ ->
                mcPackPicker.launch(arrayOf("application/zip", "application/octet-stream"))
            }
            .setNegativeButton("閉じる", null)
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

    private fun sortedJapaneseVoices(): List<Voice> =
        tts?.voices
            ?.filter { it.locale.language == Locale.JAPANESE.language }
            ?.sortedWith(
                compareByDescending<Voice> { it.quality }
                    .thenBy { it.latency }
                    .thenBy { if (it.isNetworkConnectionRequired) 0 else 1 }
                    .thenBy { it.name }
            )
            .orEmpty()

    private fun bestHumanVoice(voices: List<Voice> = sortedJapaneseVoices()): Voice? {
        val network = voices.filter { it.isNetworkConnectionRequired }
        return (if (network.isNotEmpty()) network else voices)
            .maxWithOrNull(
                compareBy<Voice> { it.quality }
                    .thenByDescending { it.latency }
            )
    }

    private fun applySavedTtsSettings() {
        val engine = tts ?: return
        val prefs = getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        val voiceName = prefs.getString(PREF_VOICE_NAME, null)
        val rate = prefs.getFloat(PREF_SPEECH_RATE, DEFAULT_SPEECH_RATE)
        val pitch = prefs.getFloat(PREF_PITCH, DEFAULT_PITCH)

        engine.language = Locale.JAPAN
        val voices = sortedJapaneseVoices()
        val selectedVoice = voices.firstOrNull { it.name == voiceName } ?: bestHumanVoice(voices)
        if (selectedVoice != null) engine.voice = selectedVoice
        engine.setSpeechRate(rate)
        engine.setPitch(pitch)
    }


    private fun evaluateBallDetectionsForSearch(
        detections: List<AiDetection>,
        roi: RectF,
        reason: String,
        players: PlayerIdentityTracker.Snapshot,
        motionProposals: List<MotionBallProposer.Proposal>,
        hoop: RectF?,
        wasLocked: Boolean,
        nowMs: Long,
        imageWidth: Int,
        imageHeight: Int
    ): BallValidationResult {
        if (detections.isEmpty()) {
            return BallValidationResult(
                geometryAccepted = emptyList(),
                accepted = emptyList(),
                diagnostics = emptyList(),
                dropSummary = "NONE"
            )
        }

        val roiCx = (roi.left + roi.right) / 2f
        val roiCy = (roi.top + roi.bottom) / 2f

        fun centerX(r: RectF) = (r.left + r.right) / 2f
        fun centerY(r: RectF) = (r.top + r.bottom) / 2f

        fun inExpandedPlayerZone(x: Float, y: Float, p: RectF): Boolean {
            val left = p.left - p.width() * 0.80f
            val right = p.right + p.width() * 0.80f
            val top = p.top - p.height() * 0.28f
            val bottom = p.bottom + p.height() * 0.28f
            return x in left..right && y in top..bottom
        }

        fun inPlayerCore(x: Float, y: Float, p: RectF): Boolean {
            val left = p.left + p.width() * 0.18f
            val right = p.right - p.width() * 0.18f
            val top = p.top + p.height() * 0.08f
            val bottom = p.top + p.height() * 0.72f
            return x in left..right && y in top..bottom
        }

        fun nearHoop(x: Float, y: Float): Boolean {
            val h = hoop ?: return false
            val dx = kotlin.math.abs(x - centerX(h))
            val dy = kotlin.math.abs(y - centerY(h))
            return dx <= maxOf(0.10f, h.width() * 3.2f) &&
                dy <= maxOf(0.13f, h.height() * 4.2f)
        }

        fun geometryDropReason(detection: AiDetection): String? {
            val box = detection.box
            val w = box.width().coerceAtLeast(0.0001f)
            val h = box.height().coerceAtLeast(0.0001f)
            val diameter = (w + h) / 2f
            // Boxes are stored in normalized coordinates. Comparing normalized
            // width/height directly makes a true circle look vertically elongated
            // on a 16:9 frame (about 0.5625 instead of 1.0). Convert back to
            // pixel dimensions before applying the roundness gate.
            val pixelW = w * imageWidth.coerceAtLeast(1)
            val pixelH = h * imageHeight.coerceAtLeast(1)
            val pixelAspect = pixelW / pixelH.coerceAtLeast(0.0001f)
            val cx = centerX(box)
            val cy = centerY(box)

            if (pixelAspect !in 0.62f..1.62f) return "SHAPE"
            if (diameter !in 0.004f..0.080f) return "SIZE"

            val playerBoxes = listOfNotNull(players.playerA, players.playerB)
            val nearPlayers = playerBoxes.filter { inExpandedPlayerZone(cx, cy, it) }
            val nearAnyPlayer = nearPlayers.isNotEmpty()
            val deepInsidePlayer = playerBoxes.any { inPlayerCore(cx, cy, it) }
            val closeToHoop = nearHoop(cx, cy)

            if (nearPlayers.isNotEmpty()) {
                val nearest = nearPlayers.minByOrNull { p ->
                    kotlin.math.hypot(
                        (cx - centerX(p)).toDouble(),
                        (cy - centerY(p)).toDouble()
                    )
                }
                val playerHeight = nearest?.height()?.coerceAtLeast(0.02f) ?: 1f
                val ratio = diameter / playerHeight
                if (ratio !in 0.025f..0.26f) return "REL_SIZE"
            }

            if (reason.startsWith("MOTION")) {
                val dx = kotlin.math.abs(cx - roiCx)
                val dy = kotlin.math.abs(cy - roiCy)
                val roiAligned =
                    dx <= roi.width() * 0.28f &&
                        dy <= roi.height() * 0.28f
                val proposalAligned = motionProposals.any { proposal ->
                    kotlin.math.hypot(
                        (cx - proposal.centerX).toDouble(),
                        (cy - proposal.centerY).toDouble()
                    ) <= 0.085
                }
                if (!roiAligned || !proposalAligned) return "MOTION"
            }

            if (!wasLocked && !nearAnyPlayer && !closeToHoop) {
                return "ACQUIRE_ZONE"
            }

            if (!wasLocked && deepInsidePlayer && !closeToHoop) {
                return "PLAYER_CORE"
            }

            return null
        }

        val geometryAccepted = mutableListOf<AiDetection>()
        val accepted = mutableListOf<AiDetection>()
        val diagnostics = mutableListOf<OverlayView.RawBallCandidate>()
        val dropCounts = linkedMapOf<String, Int>()

        detections.take(5).forEach { detection ->
            val geometryDrop = geometryDropReason(detection)
            val finalReason = if (geometryDrop != null) {
                geometryDrop
            } else {
                geometryAccepted += detection
                val contextOk = ballContextTracker.candidateAllowed(
                    box = detection.box,
                    players = players,
                    hoop = hoop,
                    nowMs = nowMs,
                    alreadyLocked = wasLocked
                )
                if (contextOk) {
                    accepted += detection
                    "PASS"
                } else {
                    "CONTEXT"
                }
            }

            diagnostics += OverlayView.RawBallCandidate(
                box = RectF(detection.box),
                score = detection.score,
                reason = finalReason
            )
            dropCounts[finalReason] = (dropCounts[finalReason] ?: 0) + 1
        }

        val dropSummary = dropCounts.entries.joinToString("|") {
            it.key + ":" + it.value
        }.ifEmpty { "NONE" }

        return BallValidationResult(
            geometryAccepted = geometryAccepted,
            accepted = accepted,
            diagnostics = diagnostics,
            dropSummary = dropSummary
        )
    }

    private fun addManualScore(player: Char, points: Int) {
        debugLogger.logEvent(
            eventType = "MANUAL_SCORE_REQUEST",
            scoreA = game.scoreA,
            scoreB = game.scoreB,
            points = points,
            detail = "player=$player"
        )
        game.addScore(player, points)
    }

    private fun shareDebugLog() {
        try {
            debugLogger.logEvent("LOG_SHARE", scoreA = game.scoreA, scoreB = game.scoreB)
            debugLogger.flush()
            val uri = FileProvider.getUriForFile(
                this,
                "$packageName.fileprovider",
                debugLogger.currentFile
            )
            val shareIntent = Intent(Intent.ACTION_SEND).apply {
                type = "text/csv"
                putExtra(Intent.EXTRA_STREAM, uri)
                putExtra(Intent.EXTRA_SUBJECT, "1on1 AI デバッグログ")
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }
            startActivity(Intent.createChooser(shareIntent, "デバッグログを共有"))
        } catch (e: Exception) {
            debugLogger.logEvent("LOG_SHARE_ERROR", detail = e.toString())
            statusText.text = "ログ共有に失敗: ${e.javaClass.simpleName}"
        }
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
                val frameTime = SystemClock.uptimeMillis()
                val rotationDegrees = image.imageInfo.rotationDegrees

                try {
                    val bitmap = analysisBitmap
                        ?.takeIf {
                            it.width == image.width &&
                                it.height == image.height &&
                                !it.isRecycled
                        }
                        ?: Bitmap.createBitmap(
                            image.width,
                            image.height,
                            Bitmap.Config.ARGB_8888
                        ).also { analysisBitmap = it }

                    val motionProposals = motionBallProposer.update(image)

                    val buffer = image.planes[0].buffer
                    buffer.rewind()
                    bitmap.copyPixelsFromBuffer(buffer)

                    pendingYolo.getAndSet(null)?.let { packet ->
                        latestYoloInferenceMs = packet.result.inferenceMs
                        latestYoloDetectionCount = packet.result.detections.size

                        playerIdentityTracker.update(
                            detections = packet.result.detections,
                            imageWidth = packet.result.imageWidth,
                            imageHeight = packet.result.imageHeight,
                            nowMs = packet.captureTimeMs
                        )
                        tracker.updateYoloDetections(
                            detections = packet.result.detections,
                            imageWidth = packet.result.imageWidth,
                            imageHeight = packet.result.imageHeight,
                            nowMs = packet.captureTimeMs
                        )

                        val sceneBalls = packet.result.detections
                            .filter { it.label == "sports ball" }
                            .map {
                                AiDetection(
                                    label = "sports ball",
                                    score = it.score,
                                    box = android.graphics.RectF(
                                        (it.box.left / packet.result.imageWidth)
                                            .coerceIn(0f, 1f),
                                        (it.box.top / packet.result.imageHeight)
                                            .coerceIn(0f, 1f),
                                        (it.box.right / packet.result.imageWidth)
                                            .coerceIn(0f, 1f),
                                        (it.box.bottom / packet.result.imageHeight)
                                            .coerceIn(0f, 1f)
                                    ),
                                    source = "SCENE_YOLO"
                                )
                            }

                        // Scene YOLO is now corroboration only. v0.5 could
                        // seed a completely false track from a full-frame false
                        // positive, so v0.6 never acquires a new ball here.
                        if (ballFusion.isLocked(packet.receivedTimeMs)) {
                            ballFusion.observe(
                                detections = sceneBalls,
                                captureTimeMs = packet.captureTimeMs,
                                receivedTimeMs = packet.receivedTimeMs,
                                source = "SCENE_YOLO"
                            )
                        }
                    }

                    pendingBall.getAndSet(null)?.let { packet ->
                        latestBallInferenceMs = packet.result.inferenceMs
                        val fusionSource =
                            if (packet.reason.startsWith("MOTION")) {
                                "BALL_MOTION_ROI"
                            } else {
                                "BALL_ROI_YOLO"
                            }

                        val rawDetections = packet.result.detections
                        val validation = evaluateBallDetectionsForSearch(
                            detections = rawDetections,
                            roi = packet.result.roi,
                            reason = packet.reason,
                            players = packet.playersAtCapture,
                            motionProposals = packet.motionAtCapture,
                            hoop = packet.hoopAtCapture,
                            wasLocked = packet.wasLockedAtCapture,
                            nowMs = packet.receivedTimeMs,
                            imageWidth = packet.result.imageWidth,
                            imageHeight = packet.result.imageHeight
                        )
                        if (validation.diagnostics.isNotEmpty()) {
                            latestRawBallCandidates = validation.diagnostics
                            latestRawBallCandidatesAt = packet.receivedTimeMs
                        }

                        val accepted = ballFusion.observe(
                            detections = validation.accepted,
                            captureTimeMs = packet.captureTimeMs,
                            receivedTimeMs = packet.receivedTimeMs,
                            source = fusionSource
                        )

                        val rawSummary = validation.diagnostics
                            .take(5)
                            .joinToString(",") {
                                val cx = (it.box.left + it.box.right) / 2f
                                val cy = (it.box.top + it.box.bottom) / 2f
                                "%.3f@%.3f|%.3f|%.3f|%.3f:%s".format(
                                    Locale.US,
                                    it.score,
                                    cx,
                                    cy,
                                    it.box.width(),
                                    it.box.height(),
                                    it.reason
                                )
                            }

                        debugLogger.logEvent(
                            "BALL_ROI_RESULT",
                            detail =
                                "accepted=" + accepted +
                                    "; rawCandidates=" + rawDetections.size +
                                    "; geometryCandidates=" + validation.geometryAccepted.size +
                                    "; candidates=" + validation.accepted.size +
                                    "; context=" + ballContextTracker.currentMode() +
                                    "; inferenceMs=" + packet.result.inferenceMs +
                                    "; reason=" + packet.reason +
                                    "; motionCount=" + packet.motionCount +
                                    "; drops=" + validation.dropSummary +
                                    "; roi=" + packet.result.roi.left + "|" +
                                    packet.result.roi.top + "|" +
                                    packet.result.roi.right + "|" +
                                    packet.result.roi.bottom +
                                    "; raw=" + rawSummary
                        )
                    }

                    val players = playerIdentityTracker.snapshot(frameTime)
                    val fusedBall = ballFusion.predict(frameTime)
                    ballContextTracker.update(
                        players = players,
                        ball = fusedBall,
                        hoop = tracker.hoopRect,
                        nowMs = frameTime
                    )
                    val snapshot = tracker.updateFrame(
                        players = players,
                        ball = fusedBall,
                        nowMs = frameTime
                    )

                    val normalizedRotation =
                        ((rotationDegrees % 360) + 360) % 360
                    val displayWidth =
                        if (normalizedRotation == 90 || normalizedRotation == 270) {
                            image.height
                        } else {
                            image.width
                        }
                    val displayHeight =
                        if (normalizedRotation == 90 || normalizedRotation == 270) {
                            image.width
                        } else {
                            image.height
                        }

                    if (frameTime - lastLogAt >= 100L) {
                        lastLogAt = frameTime
                        debugLogger.logFrame(
                            snapshot = snapshot,
                            inferenceMs = latestBallInferenceMs,
                            detectionCount = latestYoloDetectionCount,
                            ballConfidence = fusedBall?.confidence ?: 0f,
                            ballSource = fusedBall?.source ?: "NONE",
                            roiPasses = 1,
                            scoreA = game.scoreA,
                            scoreB = game.scoreB
                        )
                    }

                    if (frameTime - lastUiAt >= 50L) {
                        lastUiAt = frameTime
                        val rawBallOverlay =
                            if (frameTime - latestRawBallCandidatesAt <= 650L) {
                                latestRawBallCandidates
                            } else {
                                emptyList()
                            }
                        runOnUiThread {
                            overlayView.setCalibration(
                                tracker.hoopRect,
                                tracker.threePointLine
                            )
                            overlayView.update(
                                snapshot,
                                displayWidth,
                                displayHeight,
                                rawBallOverlay
                            )

                            val ballText = if (fusedBall != null) {
                                "BALL %.2f %s".format(
                                    fusedBall.confidence,
                                    fusedBall.source
                                )
                            } else {
                                "BALL SEARCH"
                            }

                            perfText.text =
                                "v0.6.6 SCENE ${latestYoloInferenceMs}ms | " +
                                    "BALL ${latestBallInferenceMs}ms | " +
                                    "MOTION ${motionProposals.size} | " +
                                    "CTX ${ballContextTracker.currentMode()} | " +
                                    ballText

                            if (game.running) {
                                statusText.text = snapshot.status
                            }
                        }
                    }

                    val ballInterval = if (game.running) 80L else 180L
                    val bd = ballDetector
                    if (
                        bd != null &&
                        frameTime - lastBallLaunchAt >= ballInterval &&
                        ballBusy.compareAndSet(false, true)
                    ) {
                        lastBallLaunchAt = frameTime
                        val ballBitmap =
                            bitmap.copy(Bitmap.Config.ARGB_8888, false)
                        val captureRotation = rotationDegrees
                        val captureTime = frameTime
                        val search = ballSearchPlanner.next(
                            players = players,
                            hoop = tracker.hoopRect,
                            motionProposals = motionProposals,
                            active = ballFusion.searchAnchor(frameTime),
                            focusPlayer = snapshot.lastPossessor,
                            rimPriority = tracker.needsFastBallTracking
                        )
                        val roi = search.rect
                        val capturePlayers = PlayerIdentityTracker.Snapshot(
                            playerA = players.playerA?.let { RectF(it) },
                            playerB = players.playerB?.let { RectF(it) }
                        )
                        val captureMotion = motionProposals.map {
                            it.copy(roi = RectF(it.roi))
                        }
                        val captureHoop = tracker.hoopRect?.let { RectF(it) }
                        val captureWasLocked = ballFusion.isLocked(frameTime)

                        ballExecutor.execute {
                            try {
                                val result = bd.detect(
                                    bitmap = ballBitmap,
                                    rotationDegrees = captureRotation,
                                    roiNorm = roi
                                )
                                pendingBall.set(
                                    BallPacket(
                                        result = result,
                                        captureTimeMs = captureTime,
                                        receivedTimeMs = SystemClock.uptimeMillis(),
                                        reason = search.reason,
                                        motionCount = motionProposals.size,
                                        playersAtCapture = capturePlayers,
                                        motionAtCapture = captureMotion,
                                        hoopAtCapture = captureHoop,
                                        wasLockedAtCapture = captureWasLocked
                                    )
                                )
                            } catch (e: Exception) {
                                debugLogger.logEvent(
                                    "BALL_ROI_ERROR",
                                    detail = e.toString()
                                )
                            } finally {
                                if (!ballBitmap.isRecycled) ballBitmap.recycle()
                                ballBusy.set(false)
                            }
                        }
                    }

                    val sceneInterval = if (game.running) 850L else 1200L
                    val d = detector
                    if (
                        d != null &&
                        frameTime - lastYoloLaunchAt >= sceneInterval &&
                        yoloBusy.compareAndSet(false, true)
                    ) {
                        lastYoloLaunchAt = frameTime
                        val yoloBitmap =
                            bitmap.copy(Bitmap.Config.ARGB_8888, false)
                        val captureRotation = rotationDegrees
                        val captureTime = frameTime

                        yoloExecutor.execute {
                            try {
                                val result = d.detect(
                                    bitmap = yoloBitmap,
                                    rotationDegrees = captureRotation,
                                    hoopRect = tracker.hoopRect
                                )

                                pendingYolo.set(
                                    YoloPacket(
                                        result = result,
                                        captureTimeMs = captureTime,
                                        receivedTimeMs = SystemClock.uptimeMillis()
                                    )
                                )
                            } catch (e: Exception) {
                                debugLogger.logEvent(
                                    "SCENE_YOLO_ERROR",
                                    detail = e.toString()
                                )
                            } finally {
                                if (!yoloBitmap.isRecycled) yoloBitmap.recycle()
                                yoloBusy.set(false)
                            }
                        }
                    }
                } catch (e: Exception) {
                    debugLogger.logEvent(
                        "TRACKING_ERROR",
                        detail = e.toString()
                    )
                    runOnUiThread {
                        perfText.text =
                            "TRACK error: ${e.javaClass.simpleName}"
                    }
                } finally {
                    image.close()
                }
            }

            provider.unbindAll()
            provider.bindToLifecycle(
                this,
                CameraSelector.DEFAULT_BACK_CAMERA,
                preview,
                analysis
            )
        }, ContextCompat.getMainExecutor(this))
    }

    private fun updateScoreUi(a: Int, b: Int) {
        playerAText.text = a.toString()
        playerBText.text = b.toString()
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
        debugLogger.logEvent("APP_END", scoreA = game.scoreA, scoreB = game.scoreB)
        debugLogger.close()
        super.onDestroy()
        cameraExecutor.shutdown()
        yoloExecutor.shutdown()
        ballExecutor.shutdown()
        detector?.close()
        detector = null
        ballDetector?.close()
        ballDetector = null
        mcVoicePack.release()
        analysisBitmap?.let { if (!it.isRecycled) it.recycle() }
        analysisBitmap = null
        tts?.stop()
        tts?.shutdown()
    }
}
