package com.example.openvocabdetector

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.example.openvocabdetector.detection.DetectorInput
import com.example.openvocabdetector.detection.FakeDetector
import com.example.openvocabdetector.detection.FrameResult
import com.example.openvocabdetector.detection.Labels
import com.example.openvocabdetector.detection.YuvPlanes
import com.example.openvocabdetector.overlay.DetectionOverlay
import com.example.openvocabdetector.ui.theme.OpenvocabDetectorTheme
import java.util.concurrent.Executors

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            OpenvocabDetectorTheme {
                CameraScreen()
            }
        }
    }
}

@Composable
fun CameraScreen() {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    
    var hasCameraPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(
                context,
                Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED
        )
    }
    
    val launcher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission(),
        onResult = { granted -> hasCameraPermission = granted }
    )
    
    LaunchedEffect(Unit) {
        if (!hasCameraPermission) {
            launcher.launch(Manifest.permission.CAMERA)
        }
    }

    if (hasCameraPermission) {
        Box(modifier = Modifier.fillMaxSize()) {
            var frameResult by remember { mutableStateOf<FrameResult?>(null) }
            val detector = remember { FakeDetector(Labels.P0) }
            val analysisExecutor = remember { Executors.newSingleThreadExecutor() }
            val previewView = remember { PreviewView(context) }

            LaunchedEffect(hasCameraPermission) {
                val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
                val cameraProvider = cameraProviderFuture.get()
                
                val preview = Preview.Builder().build().also {
                    it.surfaceProvider = previewView.surfaceProvider
                }

                val imageAnalysis = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                    .build()
                    .also { analysis ->
                        analysis.setAnalyzer(analysisExecutor) { imageProxy ->
                            try {
                                val input = DetectorInput.Yuv(
                                    planes = YuvPlanes(
                                        y = imageProxy.planes[0].buffer,
                                        u = imageProxy.planes[1].buffer,
                                        v = imageProxy.planes[2].buffer,
                                        yStride = imageProxy.planes[0].rowStride,
                                        uvStride = imageProxy.planes[1].rowStride,
                                        uvPixelStride = imageProxy.planes[1].pixelStride
                                    ),
                                    rotationDegrees = imageProxy.imageInfo.rotationDegrees,
                                    width = imageProxy.width,
                                    height = imageProxy.height
                                )
                                frameResult = detector.detect(input)
                            } catch (e: Exception) {
                                Log.e("CameraScreen", "Detection failed", e)
                            } finally {
                                imageProxy.close()
                            }
                        }
                    }

                try {
                    cameraProvider.unbindAll()
                    cameraProvider.bindToLifecycle(
                        lifecycleOwner,
                        CameraSelector.DEFAULT_BACK_CAMERA,
                        preview,
                        imageAnalysis
                    )
                } catch (e: Exception) {
                    Log.e("CameraScreen", "Use case binding failed", e)
                }
            }

            AndroidView(
                modifier = Modifier.fillMaxSize(),
                factory = { previewView }
            )

            DetectionOverlay(
                frameResult = frameResult,
                labels = Labels.P0,
                modifier = Modifier.fillMaxSize()
            )
        }
    } else {
        Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            Text(text = "Camera permission required")
        }
    }
}
