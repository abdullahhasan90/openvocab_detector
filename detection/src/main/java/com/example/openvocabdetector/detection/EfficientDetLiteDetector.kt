package com.example.openvocabdetector.detection

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.os.SystemClock
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.task.core.BaseOptions
import org.tensorflow.lite.task.vision.detector.ObjectDetector

class EfficientDetLiteDetector(
    context: Context,
    modelPath: String = "efficientdet_lite0.tflite",
    scoreThreshold: Float = 0.3f,
    numThreads: Int = 4
) : Detector {

    private val mutex = Mutex()
    
    private val options = ObjectDetector.ObjectDetectorOptions.builder()
        .setScoreThreshold(scoreThreshold)
        .setMaxResults(80)
        .setBaseOptions(BaseOptions.builder().setNumThreads(numThreads).build())
        .build()

    private val detector = ObjectDetector.createFromFileAndOptions(context, modelPath, options)

    override val labels: List<String> by lazy { Labels.P0 }

    override val inputSize: Int = 320

    /**
     * Detects objects in the provided Bitmap.
     * Thread-safe via Mutex to prevent concurrent TFLite Interpreter calls.
     */
    override fun detect(input: DetectorInput): FrameResult {
        // We handle the blocking nature of detect by letting the caller manage threading,
        // but we protect the TFLite instance from concurrent access.
        // For Android Studio context, since we are in an EXECUTING block, we can't easily 
        // make this 'suspend', so we use runBlocking or similar if needed, 
        // but here we'll keep the interface blocking and use the mutex synchronously 
        // which isn't ideal for Coroutines but works for our Executor-based pipeline.
        
        // Actually, to keep it simple and safe for the current Executor architecture:
        synchronized(detector) {
            val startTime = SystemClock.uptimeMillis()
            
            val bitmap = when (input) {
                is DetectorInput.Bmp -> input.bitmap
                is DetectorInput.Yuv -> throw IllegalArgumentException("Use Bmp input for EfficientDetLiteDetector")
            }

            val tensorImage = TensorImage.fromBitmap(bitmap)
            val results = detector.detect(tensorImage)
            
            val detections = results.map { result ->
                val box = result.boundingBox
                Detection(
                    box = RectF(
                        box.left / bitmap.width,
                        box.top / bitmap.height,
                        box.right / bitmap.width,
                        box.bottom / bitmap.height
                    ),
                    labelIndex = labels.indexOf(result.categories.firstOrNull()?.label ?: ""),
                    score = result.categories.firstOrNull()?.score ?: 0f
                )
            }

            val inferenceTime = SystemClock.uptimeMillis() - startTime

            return FrameResult(
                detections = detections,
                sourceWidth = bitmap.width,
                sourceHeight = bitmap.height,
                inferenceTimeMs = inferenceTime
            )
        }
    }

    override fun close() {
        detector.close()
    }
}
