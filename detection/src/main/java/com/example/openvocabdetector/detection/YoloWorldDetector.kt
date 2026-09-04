package com.example.openvocabdetector.detection

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.os.SystemClock
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.io.FileInputStream
import java.nio.channels.FileChannel
import kotlin.math.exp

class YoloWorldDetector(
    context: Context,
    modelPath: String = "yoloworld_s_int8.tflite",
    private val scoreThreshold: Float = 0.15f, // Lowered for higher sensitivity
    private val nmsThreshold: Float = 0.5f,
) : Detector {

    private val interpreter: Interpreter
    override val labels: List<String>
    override val inputSize: Int = 640

    // Actual number of classes the model was exported with (may be > labels.size)
    private val numModelClasses: Int

    // Precomputed anchor grid
    private val anchors: List<Anchor>

    // Reusable buffers
    private val inputBuffer: ByteBuffer
    private val scoreOutput: ByteBuffer
    private val bboxOutput: ByteBuffer
    private val intValues: IntArray

    init {
        // Load Model
        val assetFileDescriptor = context.assets.openFd(modelPath)
        val fileInputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = fileInputStream.channel
        val mappedByteBuffer = fileChannel.map(
            FileChannel.MapMode.READ_ONLY, 
            assetFileDescriptor.startOffset, 
            assetFileDescriptor.declaredLength
        )
        
        val options = Interpreter.Options().apply {
            setNumThreads(4)
        }
        interpreter = Interpreter(mappedByteBuffer, options)

        // Dynamically get the model's output width (channels)
        // Score tensor is index 0: [1, 8400, 280]
        numModelClasses = interpreter.getOutputTensor(0).shape()[2]

        // Load Labels
        labels = context.assets.open("labels.txt").bufferedReader().readLines()

        // Precompute Anchors for 640x640
        val mutableAnchors = mutableListOf<Anchor>()
        val strides = listOf(8, 16, 32)
        for (stride in strides) {
            val gridSide = inputSize / stride
            for (y in 0 until gridSide) {
                for (x in 0 until gridSide) {
                    mutableAnchors.add(Anchor(x.toFloat() + 0.5f, y.toFloat() + 0.5f, stride.toFloat()))
                }
            }
        }
        anchors = mutableAnchors

        // Allocate reusable buffers
        inputBuffer = ByteBuffer.allocateDirect(1 * inputSize * inputSize * 3)
        inputBuffer.order(ByteOrder.nativeOrder())
        
        scoreOutput = ByteBuffer.allocateDirect(1 * 8400 * numModelClasses)
        scoreOutput.order(ByteOrder.nativeOrder())
        
        bboxOutput = ByteBuffer.allocateDirect(1 * 8400 * 64)
        bboxOutput.order(ByteOrder.nativeOrder())
        
        intValues = IntArray(inputSize * inputSize)
    }

    /**
     * Detects objects in the provided Bitmap.
     * Thread-safe via synchronized(this) to prevent concurrent TFLite Interpreter calls.
     */
    override fun detect(input: DetectorInput): FrameResult = synchronized(this) {
        val startTime = SystemClock.uptimeMillis()
        
        val bitmap = when (input) {
            is DetectorInput.Bmp -> input.bitmap
            else -> throw IllegalArgumentException("YoloWorldDetector requires Bitmap input")
        }

        // 1. Preprocess
        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
        
        inputBuffer.rewind()
        scaledBitmap.getPixels(intValues, 0, scaledBitmap.width, 0, 0, scaledBitmap.width, scaledBitmap.height)
        
        for (pixelValue in intValues) {
            inputBuffer.put(((pixelValue shr 16 and 0xFF) - 128).toByte())
            inputBuffer.put(((pixelValue shr 8 and 0xFF) - 128).toByte())
            inputBuffer.put(((pixelValue and 0xFF) - 128).toByte())
        }
        
        if (scaledBitmap != bitmap) scaledBitmap.recycle()

        // 2. Run Inference
        scoreOutput.rewind()
        bboxOutput.rewind()

        val outputs = mutableMapOf<Int, Any>(
            0 to scoreOutput,
            1 to bboxOutput
        )
        
        interpreter.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputs)

        // 3. Decode
        val scoreTensor = interpreter.getOutputTensor(0)
        val scoreScale = scoreTensor.quantizationParams().scale
        val scoreZeroPoint = scoreTensor.quantizationParams().zeroPoint
        
        val bboxTensor = interpreter.getOutputTensor(1)
        val bboxScale = bboxTensor.quantizationParams().scale
        val bboxZeroPoint = bboxTensor.quantizationParams().zeroPoint

        val rawDetections = mutableListOf<Detection>()
        
        scoreOutput.rewind()

        for (i in 0 until 8400) {
            var maxScoreRaw: Byte = -128
            var classIndex = -1
            
            // Iterate over ALL model classes (280) to maintain buffer sync
            for (c in 0 until numModelClasses) {
                val scoreRaw = scoreOutput.get()
                // Only track if it's within our label set (270)
                if (c < labels.size && scoreRaw > maxScoreRaw) {
                    maxScoreRaw = scoreRaw
                    classIndex = c
                }
            }
            
            // Apply Dequantization + Sigmoid to get real confidence
            val logit = (maxScoreRaw.toInt() - scoreZeroPoint) * scoreScale
            val score = 1.0f / (1.0f + exp(-logit))
            
            if (score >= scoreThreshold) {
                val dists = FloatArray(4) { side ->
                    var sum = 0.0
                    val sideDists = DoubleArray(16) { bin ->
                        val raw = bboxOutput.get((i * 64) + (side * 16) + bin).toInt()
                        val dequant = (raw - bboxZeroPoint) * bboxScale
                        val expVal = exp(dequant.toDouble())
                        sum += expVal
                        expVal
                    }
                    
                    var dot = 0.0
                    for (bin in 0 until 16) {
                        dot += (sideDists[bin] / sum) * bin
                    }
                    dot.toFloat()
                }
                
                val anchor = anchors[i]
                val l = (anchor.cx - dists[0]) * anchor.stride
                val t = (anchor.cy - dists[1]) * anchor.stride
                val r = (anchor.cx + dists[2]) * anchor.stride
                val b = (anchor.cy + dists[3]) * anchor.stride
                
                // NORMALIZE & CLIP (Critical for screen boundaries)
                val left = (l / inputSize).coerceIn(0f, 1f)
                val top = (t / inputSize).coerceIn(0f, 1f)
                val right = (r / inputSize).coerceIn(0f, 1f)
                val bottom = (b / inputSize).coerceIn(0f, 1f)

                if (left < right && top < bottom) {
                    rawDetections.add(
                        Detection(
                            box = RectF(left, top, right, bottom),
                            labelIndex = classIndex,
                            score = score
                        )
                    )
                }
            }
        }

        val finalDetections = applyNms(rawDetections).take(80)
        val inferenceTime = SystemClock.uptimeMillis() - startTime

        return FrameResult(
            detections = finalDetections,
            sourceWidth = bitmap.width,
            sourceHeight = bitmap.height,
            inferenceTimeMs = inferenceTime
        )
    }

    private fun applyNms(detections: List<Detection>): List<Detection> {
        val sorted = detections.sortedByDescending { it.score }
        val results = mutableListOf<Detection>()
        val active = BooleanArray(sorted.size) { true }
        
        for (i in sorted.indices) {
            if (active[i]) {
                val d1 = sorted[i]
                results.add(d1)
                for (j in i + 1 until sorted.size) {
                    if (active[j]) {
                        val d2 = sorted[j]
                        if (calculateIoU(d1.box, d2.box) > nmsThreshold) {
                            active[j] = false
                        }
                    }
                }
            }
        }
        return results
    }

    private fun calculateIoU(box1: RectF, box2: RectF): Float {
        val interLeft = maxOf(box1.left, box2.left)
        val interTop = maxOf(box1.top, box2.top)
        val interRight = minOf(box1.right, box2.right)
        val interBottom = minOf(box1.bottom, box2.bottom)
        
        if (interLeft >= interRight || interTop >= interBottom) return 0f
        
        val interArea = (interRight - interLeft) * (interBottom - interTop)
        val box1Area = (box1.right - box1.left) * (box1.bottom - box1.top)
        val box2Area = (box2.right - box2.left) * (box2.bottom - box2.top)
        
        return interArea / (box1Area + box2Area - interArea)
    }

    override fun close() {
        interpreter.close()
    }

    private data class Anchor(val cx: Float, val cy: Float, val stride: Float)
}
