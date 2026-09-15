package com.example.ripcurrentalert

import android.Manifest
import android.annotation.SuppressLint
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Matrix
import android.graphics.PointF
import android.graphics.drawable.GradientDrawable
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.MotionEvent
import android.view.ScaleGestureDetector
import android.view.View
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.doOnLayout
import androidx.core.view.updatePadding
import coil.load
import com.google.firebase.messaging.FirebaseMessaging
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class MainActivity : AppCompatActivity() {

    private lateinit var bandTop: LinearLayout
    private lateinit var tvRiskCode: TextView
    private lateinit var tvCounters: TextView
    private lateinit var tvClock: TextView
    private lateinit var ivCctv: ImageView
    private lateinit var tvCctvHint: TextView
    private lateinit var tvUpdated: TextView
    private lateinit var bandRisk: LinearLayout
    private lateinit var tvRiskLabel: TextView
    private lateinit var tvRiskDesc: TextView
    private lateinit var tvZone: TextView
    private lateinit var tvCamera: TextView
    private lateinit var tvSafety: TextView

    private val timeFmt = SimpleDateFormat("HH:mm:ss", Locale.KOREA)
    private val clockHandler = Handler(Looper.getMainLooper())

    // 푸시가 실제 탐지 시각을 실어 보내면 그 값으로 고정된다. 없으면 수신 시각.
    private var lastUpdate: String? = null

    private val zoomMatrix = Matrix()
    private val matrixValues = FloatArray(9)
    private val lastTouch = PointF()
    private var fitScale = 1f
    private var maxScale = 4f
    private var dragging = false
    private lateinit var scaleDetector: ScaleGestureDetector

    private val clockTick = object : Runnable {
        override fun run() {
            val now = timeFmt.format(Date())
            tvClock.text = now
            tvUpdated.text = "최근 갱신 ${lastUpdate ?: now} · ${REFRESH_SEC}초 주기"
            clockHandler.postDelayed(this, TICK_MS)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        bandTop = findViewById(R.id.bandTop)
        tvRiskCode = findViewById(R.id.tvRiskCode)
        tvCounters = findViewById(R.id.tvCounters)
        tvClock = findViewById(R.id.tvClock)
        ivCctv = findViewById(R.id.ivCctv)
        tvCctvHint = findViewById(R.id.tvCctvHint)
        tvUpdated = findViewById(R.id.tvUpdated)
        bandRisk = findViewById(R.id.bandRisk)
        tvRiskLabel = findViewById(R.id.tvRiskLabel)
        tvRiskDesc = findViewById(R.id.tvRiskDesc)
        tvZone = findViewById(R.id.tvZone)
        tvCamera = findViewById(R.id.tvCamera)
        tvSafety = findViewById(R.id.tvSafety)

        applyWindowInsets()
        setupZoom()

        // Android 13 이상 알림 런타임 권한 요청
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS)
                != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(
                    this,
                    arrayOf(Manifest.permission.POST_NOTIFICATIONS),
                    101
                )
            }
        }

        // 경보 토픽 구독. 기기마다 다른 토큰을 사람이 옮길 필요가 없어진다.
        // 멱등이라 앱을 켤 때마다 호출해도 안전하다.
        FirebaseMessaging.getInstance().subscribeToTopic(ALERT_TOPIC)
            .addOnCompleteListener { task ->
                if (task.isSuccessful) {
                    Log.d("FCM_CHECK", "✅ 토픽 구독 완료: $ALERT_TOPIC")
                } else {
                    Log.e("FCM_CHECK", "❌ 토픽 구독 실패: ${task.exception?.message}")
                }
            }

        // 앱이 꺼진 상태에서 알림 탭으로 진입했을 때
        applyIntentData(intent)
    }

    // 앱이 백그라운드에 살아있는 상태에서 알림을 탭했을 때 필수 동작
    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent) // 새로운 인텐트로 교체
        applyIntentData(intent)
    }

    override fun onResume() {
        super.onResume()
        clockHandler.post(clockTick)
    }

    override fun onPause() {
        super.onPause()
        clockHandler.removeCallbacks(clockTick)
    }

    /**
     * targetSdk 37 은 edge-to-edge 가 강제라 상태바/네비게이션바가 화면 위에 겹친다.
     * 색 띠는 화면 끝까지 닿아야 하므로 루트가 아니라 위아래 끝 뷰의 패딩으로 밀어낸다.
     */
    private fun applyWindowInsets() {
        val root = findViewById<LinearLayout>(R.id.rootLayout)
        val topBase = bandTop.paddingTop
        val bottomBase = tvSafety.paddingBottom

        ViewCompat.setOnApplyWindowInsetsListener(root) { _, insets ->
            val bars = insets.getInsets(
                WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout()
            )
            bandTop.updatePadding(top = topBase + bars.top)
            tvSafety.updatePadding(bottom = bottomBase + bars.bottom)
            root.updatePadding(left = bars.left, right = bars.right)
            insets
        }
    }

    // ---------------------------------------------------------------- 데이터

    private fun applyIntentData(targetIntent: Intent?) {
        val location = targetIntent?.getStringExtra("location_name")
        if (targetIntent == null || location.isNullOrEmpty()) {
            // 푸시 없이 런처로 열린 상태. 레이아웃의 기본 표시값을 그대로 둔다.
            paintRisk(DEFAULT_RISK)
            return
        }

        // 파이프라인은 긴급 판정일 때만 알림을 보낸다. 등급 키가 없으면 긴급으로 본다.
        val sent = targetIntent.getStringExtra("risk_level")?.lowercase()
        val risk = if (sent != null && RISK_LABEL.containsKey(sent)) sent else "emergency"

        val personCount = targetIntent.getStringExtra("person_count")
        val inZone = targetIntent.getStringExtra("persons_in_rip") ?: personCount ?: DEFAULT_IN_ZONE
        val total = targetIntent.getStringExtra("total_persons") ?: personCount ?: DEFAULT_TOTAL
        val zones = targetIntent.getStringExtra("rip_count") ?: DEFAULT_ZONES

        tvCounters.text = "in-zone $inZone │ total $total │ zones $zones"
        tvZone.text = targetIntent.getStringExtra("zone") ?: location
        tvCamera.text = targetIntent.getStringExtra("camera_id") ?: DEFAULT_CAMERA

        lastUpdate = targetIntent.getStringExtra("detected_at") ?: timeFmt.format(Date())
        paintRisk(risk)

        val imageUrl = targetIntent.getStringExtra("image_url")
        if (!imageUrl.isNullOrEmpty()) loadCctv(imageUrl)
    }

    private fun paintRisk(risk: String) {
        val color = RISK_COLOR[risk] ?: RISK_COLOR.getValue(DEFAULT_RISK)
        tvRiskCode.text = risk.uppercase()
        tvRiskLabel.text = RISK_LABEL[risk]
        tvRiskDesc.text = RISK_DESC[risk]
        bandTop.setBackgroundColor(color)
        bandRisk.background = GradientDrawable().apply {
            setColor(color)
            cornerRadius = 14f * resources.displayMetrics.density
        }
    }

    private fun loadCctv(url: String) {
        ivCctv.load(url) {
            // 크로스페이드를 켜면 전환 중 drawable 이 CrossfadeDrawable 이 되어
            // intrinsic 크기가 원본과 달라지고 줌 기준 행렬이 틀어진다.
            crossfade(false)
            listener(onSuccess = { _, _ ->
                tvCctvHint.visibility = View.GONE
                ivCctv.post { resetZoom() }
            })
        }
    }

    // ------------------------------------------------------------------ 줌

    @SuppressLint("ClickableViewAccessibility")
    private fun setupZoom() {
        scaleDetector = ScaleGestureDetector(
            this,
            object : ScaleGestureDetector.SimpleOnScaleGestureListener() {
                override fun onScale(detector: ScaleGestureDetector): Boolean {
                    val current = currentScale()
                    if (current <= 0f) return true
                    val target = (current * detector.scaleFactor).coerceIn(fitScale, maxScale)
                    val factor = target / current
                    zoomMatrix.postScale(factor, factor, detector.focusX, detector.focusY)
                    applyMatrix()
                    return true
                }
            }
        )

        ivCctv.setOnTouchListener { _, event -> handleTouch(event) }
        ivCctv.doOnLayout { resetZoom() }
    }

    private fun handleTouch(event: MotionEvent): Boolean {
        scaleDetector.onTouchEvent(event)

        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                lastTouch.set(event.x, event.y)
                dragging = true
            }

            MotionEvent.ACTION_POINTER_DOWN -> dragging = false

            MotionEvent.ACTION_MOVE -> {
                if (dragging && !scaleDetector.isInProgress) {
                    zoomMatrix.postTranslate(event.x - lastTouch.x, event.y - lastTouch.y)
                    applyMatrix()
                }
                lastTouch.set(event.x, event.y)
            }

            MotionEvent.ACTION_POINTER_UP -> {
                // 떨어진 손가락 말고 남아 있는 손가락을 새 기준점으로 삼아야 화면이 튀지 않는다.
                val remaining = if (event.actionIndex == 0) 1 else 0
                if (remaining < event.pointerCount) {
                    lastTouch.set(event.getX(remaining), event.getY(remaining))
                    dragging = true
                }
            }

            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> dragging = false
        }
        return true
    }

    /** 처음에는 영역을 꽉 채우고(cover), 최대 축소 시 원본 전체가 보이게(fit) 기준을 잡는다. */
    private fun resetZoom() {
        val drawable = ivCctv.drawable ?: return
        val viewW = ivCctv.width.toFloat()
        val viewH = ivCctv.height.toFloat()
        val srcW = drawable.intrinsicWidth.toFloat()
        val srcH = drawable.intrinsicHeight.toFloat()
        if (viewW <= 0f || viewH <= 0f || srcW <= 0f || srcH <= 0f) return

        fitScale = minOf(viewW / srcW, viewH / srcH)
        val coverScale = maxOf(viewW / srcW, viewH / srcH)
        maxScale = coverScale * 4f

        zoomMatrix.reset()
        zoomMatrix.postScale(coverScale, coverScale)
        zoomMatrix.postTranslate(
            (viewW - srcW * coverScale) / 2f,
            (viewH - srcH * coverScale) / 2f
        )
        ivCctv.imageMatrix = zoomMatrix
    }

    private fun currentScale(): Float {
        zoomMatrix.getValues(matrixValues)
        return matrixValues[Matrix.MSCALE_X]
    }

    /** 이미지가 영역보다 작으면 가운데로, 크면 가장자리를 넘지 않게 붙인다. */
    private fun applyMatrix() {
        val drawable = ivCctv.drawable ?: return
        val viewW = ivCctv.width.toFloat()
        val viewH = ivCctv.height.toFloat()
        if (viewW <= 0f || viewH <= 0f) return

        zoomMatrix.getValues(matrixValues)
        val scale = matrixValues[Matrix.MSCALE_X]
        val transX = matrixValues[Matrix.MTRANS_X]
        val transY = matrixValues[Matrix.MTRANS_Y]
        val scaledW = drawable.intrinsicWidth * scale
        val scaledH = drawable.intrinsicHeight * scale

        val fixedX = if (scaledW <= viewW) (viewW - scaledW) / 2f
        else transX.coerceIn(viewW - scaledW, 0f)
        val fixedY = if (scaledH <= viewH) (viewH - scaledH) / 2f
        else transY.coerceIn(viewH - scaledH, 0f)

        zoomMatrix.postTranslate(fixedX - transX, fixedY - transY)
        ivCctv.imageMatrix = zoomMatrix
    }

    companion object {
        /** send_alert.py 의 TOPIC 과 반드시 같아야 한다. 다르면 아무도 못 받는다. */
        const val ALERT_TOPIC = "rip_current_alert"

        private const val REFRESH_SEC = "0.7"
        private const val TICK_MS = 700L

        // 레이아웃의 초기 표시값과 같아야 한다. send_alert.py 가 실제로 보내는 경보 기준.
        private const val DEFAULT_RISK = "emergency"
        private const val DEFAULT_IN_ZONE = "4"
        private const val DEFAULT_TOTAL = "4"
        private const val DEFAULT_ZONES = "1"
        private const val DEFAULT_CAMERA = "CCTV-E01"

        // core/schemas.py 의 RISK_LABEL, core/render.py 의 RISK_DESC / RISK_HEX 와 동일하게 유지한다.
        val RISK_LABEL = mapOf(
            "watch" to "관찰",
            "warn" to "경고",
            "emergency" to "🚨 긴급"
        )
        val RISK_DESC = mapOf(
            "watch" to "이안류 의심 구역 없음",
            "warn" to "구역은 있으나 인원 없음",
            "emergency" to "구역 안에 인원 감지"
        )
        val RISK_COLOR = mapOf(
            "watch" to 0xFF2E9E5B.toInt(),
            "warn" to 0xFFE08A00.toInt(),
            "emergency" to 0xFFD6202A.toInt()
        )
    }
}
