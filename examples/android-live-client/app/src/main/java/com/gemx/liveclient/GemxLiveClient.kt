package com.gemx.liveclient

import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import androidx.camera.core.ImageProxy
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger

/**
 * Thin HTTP client that talks the gem-x.cpp live protocol.
 *
 * POST /api/live?pipeline=2&detect_interval=N  → session id
 * PUT  /api/live/<id>/frame  (X-GEMX-Frame + JPEG body)
 * DELETE /api/live/<id>
 */
class GemxLiveClient(
    private var baseUrl: String,
    private val detectInterval: Int = GemxConfig.DETECT_INTERVAL
) {
    private val client = OkHttpClient.Builder()
        .connectTimeout(5, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .writeTimeout(15, TimeUnit.SECONDS)
        .build()

    private var sessionId: String? = null
    private val frameCounter = AtomicInteger(1)

    val isActive: Boolean get() = sessionId != null

    fun setBaseUrl(url: String) {
        baseUrl = url.trimEnd('/')
    }

    /** Start a live session. Throws on failure. */
    suspend fun start(): String = withContext(Dispatchers.IO) {
        val url = "$baseUrl/api/live?pipeline=2&detect_interval=$detectInterval"
        val req = Request.Builder().url(url).post(ByteArray(0).toRequestBody(null)).build()
        client.newCall(req).execute().use { resp ->
            if (!resp.isSuccessful) {
                throw RuntimeException("start failed: HTTP ${resp.code} ${resp.body?.string()}")
            }
            val body = resp.body?.string() ?: throw RuntimeException("empty start response")
            val id = JSONObject(body).getString("id")
            sessionId = id
            frameCounter.set(1)
            id
        }
    }

    /** Stop the current session (best-effort). */
    suspend fun stop() = withContext(Dispatchers.IO) {
        val id = sessionId ?: return@withContext
        sessionId = null
        val req = Request.Builder()
            .url("$baseUrl/api/live/$id")
            .delete()
            .build()
        try {
            client.newCall(req).execute().close()
        } catch (_: Exception) {
            // ignore
        }
    }

    /**
     * Encode the CameraX ImageProxy to JPEG and upload it.
     * Returns true when the server accepted the frame (including warmup 204).
     */
    suspend fun sendFrame(image: ImageProxy): Boolean = withContext(Dispatchers.IO) {
        val id = sessionId ?: return@withContext false
        val jpeg = imageProxyToJpeg(image, GemxConfig.MAX_LONG_EDGE, GemxConfig.JPEG_QUALITY)
            ?: return@withContext false

        val frameNum = frameCounter.getAndIncrement()
        val body = jpeg.toRequestBody("image/jpeg".toMediaType())
        val req = Request.Builder()
            .url("$baseUrl/api/live/$id/frame")
            .put(body)
            .header("X-GEMX-Frame", frameNum.toString())
            .build()

        client.newCall(req).execute().use { resp ->
            resp.code == 200 || resp.code == 204
        }
    }

    companion object {
        fun imageProxyToJpeg(
            image: ImageProxy,
            maxLongEdge: Int = GemxConfig.MAX_LONG_EDGE,
            quality: Int = GemxConfig.JPEG_QUALITY
        ): ByteArray? {
            if (image.format != ImageFormat.YUV_420_888) return null

            val yBuffer = image.planes[0].buffer
            val uBuffer = image.planes[1].buffer
            val vBuffer = image.planes[2].buffer
            val ySize = yBuffer.remaining()
            val uSize = uBuffer.remaining()
            val vSize = vBuffer.remaining()

            val nv21 = ByteArray(ySize + uSize + vSize)
            yBuffer.get(nv21, 0, ySize)
            vBuffer.get(nv21, ySize, vSize)
            uBuffer.get(nv21, ySize + vSize, uSize)

            val yuv = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
            val out = ByteArrayOutputStream()
            val rect = Rect(0, 0, image.width, image.height)
            if (!yuv.compressToJpeg(rect, quality, out)) return null

            var jpeg = out.toByteArray()

            val longEdge = maxOf(image.width, image.height)
            if (longEdge > maxLongEdge) {
                val scale = maxLongEdge.toFloat() / longEdge
                val w = (image.width * scale).toInt().coerceAtLeast(1)
                val h = (image.height * scale).toInt().coerceAtLeast(1)
                val bmp = android.graphics.BitmapFactory.decodeByteArray(jpeg, 0, jpeg.size)
                    ?: return jpeg
                val scaled = Bitmap.createScaledBitmap(bmp, w, h, true)
                bmp.recycle()
                val scaledOut = ByteArrayOutputStream()
                scaled.compress(Bitmap.CompressFormat.JPEG, quality, scaledOut)
                scaled.recycle()
                jpeg = scaledOut.toByteArray()
            }
            return jpeg
        }
    }
}
