package com.gemx.liveclient

object GemxConfig {
    /** Change this to the machine running gemx-demo (same Wi-Fi or adb reverse). */
    const val DEFAULT_SERVER = "http://192.168.1.42:8098"

    /** Long edge limit – matches the browser demo scaling. */
    const val MAX_LONG_EDGE = 960

    /** JPEG quality 1–100. */
    const val JPEG_QUALITY = 85

    /** Detection interval passed to the live session. */
    const val DETECT_INTERVAL = 5
}
