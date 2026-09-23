package com.gemx.liveclient

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.gemx.liveclient.databinding.ActivityMainBinding
import kotlinx.coroutines.launch
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraExecutor: ExecutorService
    private val client = GemxLiveClient(GemxConfig.DEFAULT_SERVER)
    private val streaming = AtomicBoolean(false)

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) startCameraAndSession()
        else Toast.makeText(this, "Camera permission required", Toast.LENGTH_LONG).show()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        cameraExecutor = Executors.newSingleThreadExecutor()

        binding.serverUrl.setText(GemxConfig.DEFAULT_SERVER)
        binding.btnStart.setOnClickListener { onStartClicked() }
        binding.btnStop.setOnClickListener { onStopClicked() }
    }

    private fun onStartClicked() {
        val url = binding.serverUrl.text.toString().trim()
        if (url.isEmpty()) {
            Toast.makeText(this, "Enter server URL", Toast.LENGTH_SHORT).show()
            return
        }
        client.setBaseUrl(url)

        when {
            ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                    == PackageManager.PERMISSION_GRANTED -> startCameraAndSession()
            else -> permissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun startCameraAndSession() {
        binding.btnStart.isEnabled = false
        binding.statusText.text = "Starting session…"

        lifecycleScope.launch {
            try {
                val id = client.start()
                binding.statusText.text = "Session $id – starting camera"
                bindCamera()
                streaming.set(true)
                binding.btnStop.isEnabled = true
            } catch (e: Exception) {
                Log.e(TAG, "start failed", e)
                binding.statusText.text = "Error: ${e.message}"
                binding.btnStart.isEnabled = true
                Toast.makeText(this@MainActivity, e.message, Toast.LENGTH_LONG).show()
            }
        }
    }

    private fun bindCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            val provider = providerFuture.get()

            val preview = Preview.Builder().build().also {
                it.surfaceProvider = binding.previewView.surfaceProvider
            }

            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                .build()

            analysis.setAnalyzer(cameraExecutor) { imageProxy ->
                if (!streaming.get()) {
                    imageProxy.close()
                    return@setAnalyzer
                }
                lifecycleScope.launch {
                    try {
                        val ok = client.sendFrame(imageProxy)
                        if (!ok) Log.w(TAG, "frame rejected by server")
                    } catch (e: Exception) {
                        Log.e(TAG, "sendFrame failed", e)
                        runOnUiThread {
                            binding.statusText.text = "Upload error: ${e.message}"
                        }
                    } finally {
                        imageProxy.close()
                    }
                }
            }

            try {
                provider.unbindAll()
                provider.bindToLifecycle(
                    this,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    preview,
                    analysis
                )
                binding.statusText.text = "Streaming…"
            } catch (e: Exception) {
                Log.e(TAG, "bind failed", e)
                binding.statusText.text = "Camera error: ${e.message}"
                onStopClicked()
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun onStopClicked() {
        streaming.set(false)
        binding.btnStop.isEnabled = false
        lifecycleScope.launch {
            client.stop()
            runOnUiThread {
                binding.statusText.text = getString(R.string.status_idle)
                binding.btnStart.isEnabled = true
            }
        }
        ProcessCameraProvider.getInstance(this).get().unbindAll()
    }

    override fun onDestroy() {
        super.onDestroy()
        streaming.set(false)
        cameraExecutor.shutdown()
        lifecycleScope.launch { client.stop() }
    }

    companion object {
        private const val TAG = "GemxLive"
    }
}
