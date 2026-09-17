package com.example.ripcurrentalert

import android.os.Handler
import android.os.Looper

/**
 * 서비스가 받은 경보 데이터를 같은 프로세스의 액티비티로 넘기는 통로.
 *
 * 이게 없으면 화면 갱신 경로가 "알림 탭 → PendingIntent extras" 하나뿐이라,
 * 앱이 떠 있는 상태로 경보를 받으면 배너만 뜨고 화면은 그대로 멈춰 있다.
 *
 * 서비스와 액티비티가 같은 프로세스에 있으므로(매니페스트에 android:process 없음)
 * LocalBroadcastManager(deprecated)나 registerReceiver 를 쓸 이유가 없다.
 */
object AlertBus {

    private val main = Handler(Looper.getMainLooper())

    /**
     * 마지막으로 받은 경보. 액티비티가 백그라운드일 때 온 경보를 나중에
     * 앱으로 돌아왔을 때 그려주기 위해 들고 있는다.
     */
    @Volatile
    var latest: Map<String, String>? = null
        private set

    // 액티비티가 하나뿐이라 리스너도 하나면 충분하다.
    private var listener: ((Map<String, String>) -> Unit)? = null

    /** 서비스(백그라운드 스레드)에서 호출한다. 리스너는 메인 스레드로 넘긴다. */
    fun publish(data: Map<String, String>) {
        latest = data
        val current = listener ?: return
        main.post { current(data) }
    }

    fun subscribe(listener: (Map<String, String>) -> Unit) {
        this.listener = listener
    }

    /** onPause 에서 반드시 호출한다. 안 하면 액티비티가 통째로 샌다. */
    fun unsubscribe() {
        listener = null
    }
}
