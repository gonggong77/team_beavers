package com.example.ripcurrentalert // 프로젝트 실제 패키지명 확인

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.os.Build
import android.util.Log
import androidx.core.app.NotificationCompat
import com.google.firebase.messaging.FirebaseMessaging
import com.google.firebase.messaging.FirebaseMessagingService
import com.google.firebase.messaging.RemoteMessage

class MyFirebaseMessagingService : FirebaseMessagingService() {

    override fun onNewToken(token: String) {
        super.onNewToken(token)
        Log.d("FCM_CHECK", "🔑 토큰 갱신 → 토픽 재구독")
        // 재설치나 데이터 삭제로 토큰이 바뀌면 구독도 끊길 수 있어 다시 구독한다.
        FirebaseMessaging.getInstance().subscribeToTopic(MainActivity.ALERT_TOPIC)
    }

    override fun onMessageReceived(remoteMessage: RemoteMessage) {
        super.onMessageReceived(remoteMessage)
        Log.d("FCM_CHECK", "🚀 [수신 성공] 데이터: ${remoteMessage.data}")

        val data = remoteMessage.data
        val title = data["title"] ?: "⚠️ 긴급 경보"
        val body = data["body"] ?: "위험이 감지되었습니다."

        // 앱이 떠 있으면 알림을 탭하지 않아도 화면이 바로 갱신된다.
        AlertBus.publish(data)

        sendNotification(title, body, data)
    }

    /** 알림 배너 색을 화면의 등급색과 맞춘다. 등급 키가 없으면 긴급으로 본다. */
    private fun riskColor(data: Map<String, String>): Int {
        val risk = data["risk_level"]?.lowercase()
        return MainActivity.RISK_COLOR[risk] ?: MainActivity.RISK_COLOR.getValue("emergency")
    }

    private fun sendNotification(title: String, body: String, data: Map<String, String>) {
        val channelId = "rip_current_channel"
        val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                channelId,
                "이안류 긴급 경보",
                NotificationManager.IMPORTANCE_HIGH
            ).apply {
                description = "이안류 감지 시 긴급 알림"
                enableLights(true)
                enableVibration(true)
                lockscreenVisibility = Notification.VISIBILITY_PUBLIC
            }
            notificationManager.createNotificationChannel(channel)
        }

        val intent = Intent(this, MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP
            for ((key, value) in data) {
                putExtra(key, value)
            }
        }
        val pendingIntent = PendingIntent.getActivity(
            this,
            0,
            intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        val builder = NotificationCompat.Builder(this, channelId)
            .setSmallIcon(R.drawable.ic_notification_alert)
            .setContentTitle(title)
            .setContentText(body)
            .setAutoCancel(true)
            .setPriority(NotificationCompat.PRIORITY_MAX)
            .setDefaults(NotificationCompat.DEFAULT_ALL)
            .setContentIntent(pendingIntent)
            .setColor(riskColor(data))
            .setColorized(true)

        // 고정 ID 로 알림을 덮어써서 최신 경보 하나만 남긴다.
        // 다만 같은 ID 로 그냥 다시 notify 하면 안드로이드가 '업데이트'로 처리해
        // 헤즈업 배너도 소리도 다시 울리지 않는다. 경보는 매번 새로 알려야 하므로
        // 먼저 지우고 새 알림으로 올린다.
        notificationManager.cancel(NOTIFY_ID)
        notificationManager.notify(NOTIFY_ID, builder.build())
        Log.d("FCM_CHECK", "🔔 알림 배너 생성 완료")
    }

    companion object {
        private const val NOTIFY_ID = 1001
    }
}