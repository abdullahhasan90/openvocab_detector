package com.example.openvocabdetector

import android.Manifest
import android.content.ContentValues
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.SystemClock
import android.provider.MediaStore
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.OptIn
import androidx.camera.core.CameraControl
import androidx.camera.core.CameraInfo
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.video.*
import androidx.camera.view.PreviewView
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.pointer.pointerInput
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
import kotlinx.coroutines.launch
import java.io.File
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
    val scope = rememberCoroutineScope()
    
    // State
    var frameResult by remember { mutableStateOf<FrameResult?>(null) }
    var recording by remember { mutableStateOf<Recording?>(null) }
    var recordingStartTime by remember { mutableLongStateOf(0L) }
    val liveMetadata = remember { mutableStateMapOf<Int, FrameResult>() }
    
    var lastVideoUri by remember { mutableStateOf<Uri?>(null) }
    var processingResults by remember { mutableStateOf<Map<Int, FrameResult>?>(null) }
    var processingProgress by remember { mutableStateOf(0f) }
    var isProcessing by remember { mutableStateOf(false) }
    var isPlaying by remember { mutableStateOf(false) }
    var exportProgress by remember { mutableStateOf<Float?>(null) }

    // Camera control for zoom
    var cameraControl by remember { mutableStateOf<CameraControl?>(null) }
    var zoomRatio by remember { mutableFloatStateOf(0f) } // 0..1 linear zoom

    // ML & Media
    var detector by remember { mutableStateOf<Detector?>(null) }
    val videoProcessor = remember(detector) { detector?.let { VideoProcessor(it) } }
    val analysisExecutor = remember { Executors.newSingleThreadExecutor() }
    
    // Safety flag to stop live inference when processing/playing
    val isAnalysisActive = remember { 
        derivedStateOf { !isProcessing && !isPlaying && detector != null } 
    }

    DisposableEffect(Unit) {
        onDispose {
            detector?.close()
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
        analysisExecutor.execute {
            try {
                // SMARTER INT8 PATH: Calibrated model for high sensitivity + XNNPACK speed
                Log.d("MainContent", "Initializing Calibrated Int8 Detector...")
                detector = YoloWorldDetector(context, modelPath = "yoloworld_s_int8_calibrated.tflite", useGpu = false)
                Log.d("MainContent", "Calibrated Detector ready.")
            } catch (t: Throwable) {
                Log.e("MainContent", "Init failed, falling back to basic Int8", t)
                try {
                    detector = YoloWorldDetector(context, modelPath = "yoloworld_s_int8.tflite", useGpu = false)
                } catch (t2: Throwable) {
                    Log.e("MainContent", "Total Init Failure", t2)
                }
            }
        }

        val cameraProvider = ProcessCameraProvider.getInstance(context).get()
        val preview = Preview.Builder().build().also {
            it.surfaceProvider = previewView.surfaceProvider
        }
        val imageAnalysis = ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()
            .also { analysis ->
                analysis.setAnalyzer(analysisExecutor) { imageProxy ->
                    val currentDetector = detector
                    if (!isAnalysisActive.value || currentDetector == null) {
                        imageProxy.close()
                        return@setAnalyzer
                    }

                    try {
                        val bitmap = imageProxy.toBitmap()
                        val rotation = imageProxy.imageInfo.rotationDegrees
                        val rotatedBitmap = if (rotation != 0) {
                            val matrix = Matrix().apply { postRotate(rotation.toFloat()) }
                            Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
                        } else {
                            bitmap
                        }

                        frameResult = currentDetector.detect(DetectorInput.Bmp(rotatedBitmap))
                        
                        // Collect live metadata if recording
                        if (recording != null) {
                            val elapsedMs = SystemClock.elapsedRealtime() - recordingStartTime
                            val frameIndex = (elapsedMs / 1000f * 30).toInt()
                            liveMetadata[frameIndex] = frameResult!!
                        }
                    } catch (e: Exception) {
                        Log.e("CameraScreen", "Detection failed", e)
                    } finally {
                        imageProxy.close()
                    }
                }
            }

        try {
            cameraProvider.unbindAll()
            val camera = cameraProvider.bindToLifecycle(
                lifecycleOwner,
                CameraSelector.DEFAULT_BACK_CAMERA,
                preview,
                imageAnalysis,
                videoCapture
            )
            cameraControl = camera.cameraControl
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
                    labels = detector?.labels ?: emptyList(),
                    onClose = { isPlaying = false },
                    exportProgress = exportProgress,
                    onStartExport = {
                        if (exportProgress == null) {
                            scope.launch {
                                val fileName = "burned_${System.currentTimeMillis()}.mp4"
                                val outputFile = File(context.cacheDir, fileName)
                                val success = videoProcessor?.exportBurnedVideo(
                                    context, lastVideoUri!!, outputFile, 
                                    processingResults!!, detector?.labels ?: emptyList()
                                ) {
                                    exportProgress = it
                                }
                                
                                if (success == true) {
                                    // Move to gallery
                                    val values = ContentValues().apply {
                                        put(MediaStore.Video.Media.DISPLAY_NAME, fileName)
                                        put(MediaStore.Video.Media.MIME_TYPE, "video/mp4")
                                        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                                            put(MediaStore.Video.Media.RELATIVE_PATH, "Movies/OpenvocabDetector")
                                        }
                                    }
                                    context.contentResolver.insert(MediaStore.Video.Media.EXTERNAL_CONTENT_URI, values)?.let { uri ->
                                        context.contentResolver.openOutputStream(uri)?.use { out ->
                                            outputFile.inputStream().use { it.copyTo(out) }
                                        }
                                        Toast.makeText(context, "Burned video saved to gallery", Toast.LENGTH_LONG).show()
                                    }
                                }
                                exportProgress = 1.0f // Mark as done
                            }
                        }
                    }
                )
            } else {
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .pointerInput(Unit) {
                            detectTransformGestures { _, _, zoom, _ ->
                                if (zoom != 1f) {
                                    val newZoom = (zoomRatio + (zoom - 1f) * 1.5f).coerceIn(0f, 1f)
                                    zoomRatio = newZoom
                                    cameraControl?.setLinearZoom(newZoom)
                                }
                            }
                        }
                ) {
                    AndroidView(factory = { previewView }, modifier = Modifier.fillMaxSize())
                }
                
                DetectionOverlay(
                    frameResult = frameResult,
                    labels = detector?.labels ?: emptyList(),
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
                            // Reset live metadata and export progress
                            liveMetadata.clear()
                            exportProgress = null
                            
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
                                    if (event is VideoRecordEvent.Start) {
                                        recordingStartTime = SystemClock.elapsedRealtime()
                                    } else if (event is VideoRecordEvent.Finalize) {
                                        if (!event.hasError()) {
                                            val uri = event.outputResults.outputUri
                                            lastVideoUri = uri
                                            Log.d("MainContent", "Video saved to Gallery: $uri")
                                            
                                            // Save metadata as a sidecar (Internal storage for simplicity first)
                                            // We use the URI hash or similar to link them
                                            val metadataFile = File(context.filesDir, "last_metadata.json")
                                            videoProcessor?.saveMetadata(liveMetadata.toMap(), metadataFile)
                                        }
                                    }
                                }
                        }
                    },
                    hasVideo = lastVideoUri != null,
                    isProcessing = isProcessing,
                    onProcess = {
                        if (lastVideoUri != null) {
                            frameResult = null // Clear live boxes
                            val metadataFile = File(context.filesDir, "last_metadata.json")
                            val loadedResults = videoProcessor?.loadMetadata(metadataFile)
                            
                            if (loadedResults != null) {
                                processingResults = loadedResults
                                isPlaying = true
                            } else {
                                // Fallback to slow processing if no sidecar
                                isProcessing = true
                            }
                        }
                    },
                    modifier = Modifier.align(Alignment.BottomCenter).padding(bottom = 32.dp)
                )

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
    
    LaunchedEffect(isProcessing) {
        if (isProcessing && lastVideoUri != null) {
            val processor = videoProcessor ?: return@LaunchedEffect
            delay(500)
            processor.processVideo(context, lastVideoUri!!).collect { status ->
                when (status) {
                    is ProcessingStatus.Progress -> {
                        processingProgress = status.progress
                    }
                    is ProcessingStatus.Complete -> {
                        Toast.makeText(context, "Processing complete!", Toast.LENGTH_SHORT).show()
                        processingResults = status.results
                        isProcessing = false
                        isPlaying = true
                    }
                    is ProcessingStatus.Error -> {
                        isProcessing = false
                        Toast.makeText(context, "Processing failed: ${status.throwable.message}", Toast.LENGTH_LONG).show()
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
    isProcessing: Boolean,
    onProcess: () -> Unit,
    modifier: Modifier = Modifier
) {
    Row(
        modifier = modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceEvenly,
        verticalAlignment = Alignment.CenterVertically
    ) {
        if (hasVideo) {
            Button(
                onClick = onProcess,
                enabled = !isProcessing
            ) {
                Text("Review Last")
            }
        }
        
        IconButton(
            onClick = onRecordToggle,
            modifier = Modifier.size(72.dp).background(if (isRecording) Color.Red else Color.White, CircleShape)
        ) {
        }
    }
}

@OptIn(UnstableApi::class)
@Composable
fun PlaybackView(
    videoUri: Uri,
    results: Map<Int, FrameResult>,
    labels: List<String>,
    onClose: () -> Unit,
    exportProgress: Float?,
    onStartExport: () -> Unit
) {
    val context = LocalContext.current
    
    // Auto-start export on first view
    LaunchedEffect(Unit) {
        onStartExport()
    }
    
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
            
            // Search back up to 5 frames (~166ms) to find the most recent detection
            // This provides "stickiness" and eliminates flicker
            var foundFrame: FrameResult? = null
            for (offset in 0..5) {
                val targetIndex = frameIndex - offset
                if (results.containsKey(targetIndex)) {
                    foundFrame = results[targetIndex]
                    break
                }
            }
            
            currentFrameResult = foundFrame
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
            labels = labels,
            modifier = Modifier.fillMaxSize(),
            scaleType = ViewportTransform.ScaleType.FIT_CENTER
        )
        
        if (exportProgress != null && exportProgress < 1.0f) {
            Box(modifier = Modifier.align(Alignment.TopEnd).padding(16.dp)) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    CircularProgressIndicator(
                        progress = { exportProgress },
                        modifier = Modifier.size(32.dp),
                        strokeWidth = 3.dp,
                        color = Color.Cyan
                    )
                    Text("Burning HUD...", color = Color.Cyan, style = MaterialTheme.typography.labelSmall)
                }
            }
        }
        
        Button(
            onClick = onClose,
            modifier = Modifier.align(Alignment.TopStart).padding(16.dp)
        ) {
            Text("Close")
        }
    }
}
