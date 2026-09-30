package jp.showchoo.oneononeai

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.PointF
import android.graphics.RectF
import android.util.AttributeSet
import android.view.MotionEvent
import android.view.View
import kotlin.math.min

class OverlayView(context: Context, attrs: AttributeSet?) : View(context, attrs) {
    data class RawBallCandidate(
        val box: RectF,
        val score: Float,
        val reason: String
    )
    private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 5f
    }
    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        textSize = 34f
        typeface = android.graphics.Typeface.DEFAULT_BOLD
    }
    private val hoopPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.YELLOW
        style = Paint.Style.STROKE
        strokeWidth = 6f
    }
    private val linePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.CYAN
        style = Paint.Style.STROKE
        strokeWidth = 6f
    }
    private val pointPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.CYAN
        style = Paint.Style.FILL
    }
    private val rawBallPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.RED
        style = Paint.Style.STROKE
        strokeWidth = 3f
    }
    private val rawBallTextPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.RED
        textSize = 22f
        typeface = android.graphics.Typeface.DEFAULT_BOLD
    }

    private var snapshot: TrackerSnapshot? = null
    private var imageWidth = 16
    private var imageHeight = 9
    private var hoopRectNorm: RectF? = null
    private var threePointNorm: List<PointF> = emptyList()
    private var rawBallCandidates: List<RawBallCandidate> = emptyList()

    private enum class CalibrationMode { NONE, HOOP, THREE }
    private var calibrationMode = CalibrationMode.NONE
    private val pendingThree = mutableListOf<PointF>()
    private var onHoopCalibrated: ((RectF) -> Unit)? = null
    private var onThreeCalibrated: ((List<PointF>) -> Unit)? = null

    fun update(
        snapshot: TrackerSnapshot,
        imageWidth: Int,
        imageHeight: Int,
        rawBallCandidates: List<RawBallCandidate> = emptyList()
    ) {
        this.snapshot = snapshot
        this.imageWidth = imageWidth.coerceAtLeast(1)
        this.imageHeight = imageHeight.coerceAtLeast(1)
        this.rawBallCandidates = rawBallCandidates.map {
            it.copy(box = RectF(it.box))
        }
        invalidate()
    }

    fun setCalibration(hoop: RectF?, three: List<PointF>) {
        hoopRectNorm = hoop
        threePointNorm = three
        invalidate()
    }

    fun calibrateHoop(onDone: (RectF) -> Unit) {
        calibrationMode = CalibrationMode.HOOP
        onHoopCalibrated = onDone
        pendingThree.clear()
        invalidate()
    }

    fun calibrateThreePointLine(onDone: (List<PointF>) -> Unit) {
        calibrationMode = CalibrationMode.THREE
        onThreeCalibrated = onDone
        pendingThree.clear()
        invalidate()
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        if (event.action != MotionEvent.ACTION_DOWN) return true
        val n = viewToNormalized(event.x, event.y) ?: return true
        when (calibrationMode) {
            CalibrationMode.HOOP -> {
                val w = 0.10f
                val h = 0.065f
                val rect = RectF(
                    (n.x - w / 2).coerceIn(0f, 1f),
                    (n.y - h / 2).coerceIn(0f, 1f),
                    (n.x + w / 2).coerceIn(0f, 1f),
                    (n.y + h / 2).coerceIn(0f, 1f)
                )
                hoopRectNorm = rect
                calibrationMode = CalibrationMode.NONE
                onHoopCalibrated?.invoke(rect)
            }
            CalibrationMode.THREE -> {
                pendingThree += n
                if (pendingThree.size >= 5) {
                    val sorted = pendingThree.sortedBy { it.x }
                    threePointNorm = sorted
                    calibrationMode = CalibrationMode.NONE
                    onThreeCalibrated?.invoke(sorted)
                    pendingThree.clear()
                }
            }
            CalibrationMode.NONE -> Unit
        }
        invalidate()
        return true
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        hoopRectNorm?.let { canvas.drawRect(toView(it), hoopPaint) }
        drawThreePoint(canvas, threePointNorm)
        if (pendingThree.isNotEmpty()) drawThreePoint(canvas, pendingThree)

        rawBallCandidates.forEach { candidate ->
            val r = toView(candidate.box)
            canvas.drawOval(r, rawBallPaint)
            val label = "RAW %.2f %s".format(candidate.score, candidate.reason)
            canvas.drawText(
                label,
                r.left,
                (r.top - 5f).coerceAtLeast(24f),
                rawBallTextPaint
            )
        }

        snapshot?.let { s ->
            s.playerA?.let {
                boxPaint.color = Color.GREEN
                textPaint.color = Color.GREEN
                val r = toView(it)
                canvas.drawRect(r, boxPaint)
                canvas.drawText("A", r.left, (r.top - 8f).coerceAtLeast(36f), textPaint)
            }
            s.playerB?.let {
                boxPaint.color = Color.MAGENTA
                textPaint.color = Color.MAGENTA
                val r = toView(it)
                canvas.drawRect(r, boxPaint)
                canvas.drawText("B", r.left, (r.top - 8f).coerceAtLeast(36f), textPaint)
            }
            s.ball?.let {
                boxPaint.color = Color.YELLOW
                canvas.drawOval(toView(it), boxPaint)
            }
        }

        if (calibrationMode == CalibrationMode.HOOP) {
            textPaint.color = Color.YELLOW
            canvas.drawText("リング中央をタップ", 32f, height / 2f, textPaint)
        } else if (calibrationMode == CalibrationMode.THREE) {
            textPaint.color = Color.CYAN
            canvas.drawText("3Pラインを左から5点タップ ${pendingThree.size}/5", 32f, height / 2f, textPaint)
        }
    }

    private fun drawThreePoint(canvas: Canvas, points: List<PointF>) {
        if (points.isEmpty()) return
        val viewPts = points.map { normalizedToView(it) }
        viewPts.forEach { canvas.drawCircle(it.x, it.y, 9f, pointPaint) }
        for (i in 0 until viewPts.lastIndex) {
            canvas.drawLine(viewPts[i].x, viewPts[i].y, viewPts[i + 1].x, viewPts[i + 1].y, linePaint)
        }
    }

    private fun contentRect(): RectF {
        val vw = width.toFloat().coerceAtLeast(1f)
        val vh = height.toFloat().coerceAtLeast(1f)
        val scale = min(vw / imageWidth, vh / imageHeight)
        val cw = imageWidth * scale
        val ch = imageHeight * scale
        val left = (vw - cw) / 2f
        val top = (vh - ch) / 2f
        return RectF(left, top, left + cw, top + ch)
    }

    private fun toView(r: RectF): RectF {
        val c = contentRect()
        return RectF(
            c.left + r.left * c.width(),
            c.top + r.top * c.height(),
            c.left + r.right * c.width(),
            c.top + r.bottom * c.height()
        )
    }

    private fun normalizedToView(p: PointF): PointF {
        val c = contentRect()
        return PointF(c.left + p.x * c.width(), c.top + p.y * c.height())
    }

    private fun viewToNormalized(x: Float, y: Float): PointF? {
        val c = contentRect()
        if (!c.contains(x, y)) return null
        return PointF((x - c.left) / c.width(), (y - c.top) / c.height())
    }
}
