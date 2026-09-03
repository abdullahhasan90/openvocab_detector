package com.example.openvocabdetector

import android.Manifest
import android.content.ContentValues
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.OptIn
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.video.*
import androidx.camera.view.PreviewView
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.media3.common.MediaItem
import androidx.media3.common.util.UnstableApi
import androidx.media3.exoplayer.ExoPlayer
import androidx.media3.ui.AspectRatioFrameLayout
import androidx.media3.ui.PlayerView
import com.example.openvocabdetector.detection.*
import com.example.openvocabdetector.media.ProcessingStatus
import com.example.openvocabdetector.media.VideoProcessor
import com.example.openvocabdetector.overlay.DetectionOverlay
import com.example.openvocabdetector.overlay.ViewportTransform
import com.example.openvocabdetector.ui.theme.OpenvocabDetectorTheme
import kotlinx.coroutines.delay
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
    
    var hasCameraPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
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
        MainContent()
    } else {
        Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            Text(text = "Camera permission required")
        }
    }
}

@Composable
fun MainContent() {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    
    // State
    var frameResult by remember { mutableStateOf<FrameResult?>(null) }
    var recording by remember { mutableStateOf<Recording?>(null) }
    var lastVideoUri by remember { mutableStateOf<Uri?>(null) }
    var processingResults by remember { mutableStateOf<Map<Int, FrameResult>?>(null) }
    var processingProgress by remember { mutableStateOf(0f) }
    var isProcessing by remember { mutableStateOf(false) }
    var isPlaying by remember { mutableStateOf(false) }

    // ML & Media
    val detector = remember { EfficientDetLiteDetector(context) }
    val videoProcessor = remember { VideoProcessor(detector) }
    val analysisExecutor = remember { Executors.newSingleThreadExecutor() }
    
    // Safety flag to stop live inference when processing/playing
    val isAnalysisActive = remember(isProcessing, isPlaying) {
        !isProcessing && !isPlaying
    }

    DisposableEffect(Unit) {
        onDispose {
            detector.close()
            analysisExecutor.shutdown()
        }
    }
    
    // CameraX setup
    val previewView = remember { PreviewView(context) }
    val videoCapture = remember {
        val recorder = Recorder.Builder()
            .setQualitySelector(QualitySelector.from(Quality.SD))
            .build()
        VideoCapture.withOutput(recorder)
    }

    LaunchedEffect(Unit) {
        val cameraProvider = ProcessCameraProvider.getInstance(context).get()
        val preview = Preview.Builder().build().also {
            it.surfaceProvider = previewView.surfaceProvider
        }
        val imageAnalysis = ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()
            .also { analysis ->
                analysis.setAnalyzer(analysisExecutor) { imageProxy ->
                    if (!isAnalysisActive) {
                        imageProxy.close()
                        return@setAnalyzer
                    }

                    try {
                        // Use official ImageProxy.toBitmap() - Handles basic YUV conversion
                        val bitmap = imageProxy.toBitmap()
                        
                        // Rotate bitmap correctly for the model
                        val rotation = imageProxy.imageInfo.rotationDegrees
                        val rotatedBitmap = if (rotation != 0) {
                            val matrix = Matrix().apply { postRotate(rotation.toFloat()) }
                            Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
                        } else {
                            bitmap
                        }

                        frameResult = detector.detect(DetectorInput.Bmp(rotatedBitmap))
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
                imageAnalysis,
                videoCapture
            )
        } catch (e: Exception) {
            Log.e("CameraScreen", "Use case binding failed", e)
        }
    }

    Scaffold(
        modifier = Modifier.fillMaxSize()
    ) { padding ->
        Box(modifier = Modifier.padding(padding).fillMaxSize()) {
            if (isPlaying && lastVideoUri != null && processingResults != null) {
                PlaybackView(
                    videoUri = lastVideoUri!!,
                    results = processingResults!!,
                    onClose = { isPlaying = false }
                )
            } else {
                AndroidView(factory = { previewView }, modifier = Modifier.fillMaxSize())
                
                DetectionOverlay(
                    frameResult = frameResult,
                    labels = detector.labels,
                    modifier = Modifier.fillMaxSize()
                )

                ControlBar(
                    isRecording = recording != null,
                    onRecordToggle = {
                        if (recording != null) {
                            recording?.stop()
                            recording = null
                        } else {
                            val fileName = "capture_${System.currentTimeMillis()}"
                            val contentValues = ContentValues().apply {
                                put(MediaStore.MediaColumns.DISPLAY_NAME, fileName)
                                put(MediaStore.MediaColumns.MIME_TYPE, "video/mp4")
                                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                                    put(MediaStore.Video.Media.RELATIVE_PATH, "Movies/OpenvocabDetector")
                                }
                            }

                            val mediaStoreOutputOptions = MediaStoreOutputOptions
                                .Builder(context.contentResolver, MediaStore.Video.Media.EXTERNAL_CONTENT_URI)
                                .setContentValues(contentValues)
                                .build()

                            recording = videoCapture.output
                                .prepareRecording(context, mediaStoreOutputOptions)
                                .start(ContextCompat.getMainExecutor(context)) { event ->
                                    if (event is VideoRecordEvent.Finalize) {
                                        if (!event.hasError()) {
                                            lastVideoUri = event.outputResults.outputUri
                                            Log.d("MainContent", "Video saved to Gallery: $lastVideoUri")
                                        }
                                    }
                                }
                        }
                    },
                    hasVideo = lastVideoUri != null,
                    onProcess = {
                        if (lastVideoUri != null) {
                            isProcessing = true
                        }
                    },
                    modifier = Modifier.align(Alignment.BottomCenter).padding(bottom = 32.dp)
                )

                // Processing UI
                if (isProcessing) {
                    Box(
                        modifier = Modifier.fillMaxSize().background(Color.Black.copy(alpha = 0.5f)),
                        contentAlignment = Alignment.Center
                    ) {
                        Column(horizontalAlignment = Alignment.CenterHorizontally) {
                            CircularProgressIndicator(progress = { processingProgress })
                            Spacer(modifier = Modifier.height(16.dp))
                            Text("Processing Video...", color = Color.White)
                        }
                    }
                }
            }
        }
    }
    
    // Processing trigger
    LaunchedEffect(isProcessing) {
        if (isProcessing && lastVideoUri != null) {
            videoProcessor.processVideo(context, lastVideoUri!!).collect { status ->
                when (status) {
                    is ProcessingStatus.Progress -> processingProgress = status.progress
                    is ProcessingStatus.Complete -> {
                        processingResults = status.results
                        isProcessing = false
                        isPlaying = true
                    }
                    is ProcessingStatus.Error -> {
                        isProcessing = false
                        Log.e("MainContent", "Processing failed", status.throwable)
                    }
                }
            }
        }
    }
}

@Composable
fun ControlBar(
    isRecording: Boolean,
    onRecordToggle: () -> Unit,
    hasVideo: Boolean,
    onProcess: () -> Unit,
    modifier: Modifier = Modifier
) {
    Row(
        modifier = modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceEvenly,
        verticalAlignment = Alignment.CenterVertically
    ) {
        if (hasVideo) {
            Button(onClick = onProcess) {
                Text("Process Last")
            }
        }
        
        IconButton(
            onClick = onRecordToggle,
            modifier = Modifier.size(72.dp).background(if (isRecording) Color.Red else Color.White, CircleShape)
        ) {
            // Record circle
        }
    }
}

@OptIn(UnstableApi::class)
@Composable
fun PlaybackView(
    videoUri: Uri,
    results: Map<Int, FrameResult>,
    onClose: () -> Unit
) {
    val context = LocalContext.current
    val exoPlayer = remember {
        ExoPlayer.Builder(context).build().apply {
            setMediaItem(MediaItem.fromUri(videoUri))
            prepare()
            playWhenReady = true
        }
    }
    
    var currentFrameResult by remember { mutableStateOf<FrameResult?>(null) }

    DisposableEffect(Unit) {
        onDispose { exoPlayer.release() }
    }

    LaunchedEffect(exoPlayer) {
        while (true) {
            val pos = exoPlayer.currentPosition
            val frameIndex = (pos / 1000f * 30).toInt()
            currentFrameResult = results[frameIndex]
            delay(16)
        }
    }

    Box(modifier = Modifier.fillMaxSize()) {
        AndroidView(
            factory = {
                PlayerView(it).apply {
                    player = exoPlayer
                    useController = true
                    resizeMode = AspectRatioFrameLayout.RESIZE_MODE_FIT
                }
            },
            modifier = Modifier.fillMaxSize()
        )
        
        DetectionOverlay(
            frameResult = currentFrameResult,
            labels = Labels.P0,
            modifier = Modifier.fillMaxSize(),
            scaleType = ViewportTransform.ScaleType.FIT_CENTER
        )
        
        Button(
            onClick = onClose,
            modifier = Modifier.align(Alignment.TopStart).padding(16.dp)
        ) {
            Text("Close")
        }
    }
}
