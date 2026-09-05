package com.example.openvocabdetector.detection

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.os.SystemClock
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.io.FileInputStream
import java.nio.channels.FileChannel
import kotlin.math.exp
import kotlin.math.ln

class YoloWorldDetector(
    context: Context,
    modelPath: String = "yoloworld_s_int8.tflite",
    private val scoreThreshold: Float = 0.18f,
    private val nmsThreshold: Float = 0.5f,
    useGpu: Boolean = false
) : Detector {

    private val interpreter: Interpreter
    private var gpuDelegate: GpuDelegate? = null
    override val labels: List<String>
    override val inputSize: Int = 640

    private val numModelClasses: Int
    private val isFloatModel: Boolean
    private val scoreTensorIndex: Int
    private val bboxTensorIndex: Int

    private val anchors: List<Anchor>

    // Reusable buffers and arrays for BULK tensor access
    private val inputBuffer: ByteBuffer
    private val scoreOutput: ByteBuffer
    private val bboxOutput: ByteBuffer
    private val intValues: IntArray
    
    private val scoreArrayFloat: FloatArray?
    private val scoreArrayByte: ByteArray?
    private val bboxArrayFloat: FloatArray?
    private val bboxArrayByte: ByteArray?
    
    private val inputFloats: FloatArray?
    private val inputBytes: ByteArray?

    // Optimization constants
    private val logitThreshold: Float = -ln(1.0f / scoreThreshold - 1.0f)

    init {
        val assetFileDescriptor = context.assets.openFd(modelPath)
        val mappedByteBuffer = FileInputStream(assetFileDescriptor.fileDescriptor).channel.map(
            FileChannel.MapMode.READ_ONLY, 
            assetFileDescriptor.startOffset, 
            assetFileDescriptor.declaredLength
        )
        
        val options = Interpreter.Options().apply {
            setUseXNNPACK(true)
            setNumThreads(4)
            
            if (useGpu) { 
                try {
                    val gpuOptions = GpuDelegate.Options().apply {
                        this.isPrecisionLossAllowed = true
                        this.inferencePreference = GpuDelegate.Options.INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER
                    }
                    gpuDelegate = GpuDelegate(gpuOptions)
                    addDelegate(gpuDelegate)
                    Log.d("YoloWorld", "GPU delegate enabled")
                } catch (t: Throwable) {
                    Log.e("YoloWorld", "GPU Init failed fallback to CPU", t)
                }
            }
        }
        
        Log.d("YoloWorld", "Initializing Hybrid Path (Model: $modelPath, Engine: XNNPACK)")
        
        try {
            interpreter = Interpreter(mappedByteBuffer, options)
        } catch (t: Throwable) {
            gpuDelegate?.close()
            throw t
        }

        val inputTensor = interpreter.getInputTensor(0)
        isFloatModel = inputTensor.dataType() == DataType.FLOAT32
        
        val out0 = interpreter.getOutputTensor(0)
        if (out0.shape()[2] == 64) {
            bboxTensorIndex = 0
            scoreTensorIndex = 1
        } else {
            bboxTensorIndex = 1
            scoreTensorIndex = 0
        }
        
        numModelClasses = interpreter.getOutputTensor(scoreTensorIndex).shape()[2]
        labels = context.assets.open("labels.txt").bufferedReader().readLines()

        val mutableAnchors = mutableListOf<Anchor>()
        for (stride in listOf(8, 16, 32)) {
            val gridSide = inputSize / stride
            for (y in 0 until gridSide) {
                for (x in 0 until gridSide) {
                    mutableAnchors.add(Anchor(x.toFloat() + 0.5f, y.toFloat() + 0.5f, stride.toFloat()))
                }
            }
        }
        anchors = mutableAnchors

        val bytesPerValue = if (isFloatModel) 4 else 1
        inputBuffer = ByteBuffer.allocateDirect(inputSize * inputSize * 3 * bytesPerValue).apply { 
            order(ByteOrder.nativeOrder()) 
        }
        scoreOutput = ByteBuffer.allocateDirect(8400 * numModelClasses * bytesPerValue).apply { 
            order(ByteOrder.nativeOrder()) 
        }
        bboxOutput = ByteBuffer.allocateDirect(8400 * 64 * bytesPerValue).apply { 
            order(ByteOrder.nativeOrder()) 
        }
        intValues = IntArray(inputSize * inputSize)

        if (isFloatModel) {
            scoreArrayFloat = FloatArray(8400 * numModelClasses)
            bboxArrayFloat = FloatArray(8400 * 64)
            scoreArrayByte = null
            bboxArrayByte = null
            inputFloats = FloatArray(inputSize * inputSize * 3)
            inputBytes = null
        } else {
            scoreArrayFloat = null
            bboxArrayFloat = null
            scoreArrayByte = ByteArray(8400 * numModelClasses)
            bboxArrayByte = ByteArray(8400 * 64)
            inputFloats = null
            inputBytes = ByteArray(inputSize * inputSize * 3)
        }
    }

    override fun detect(input: DetectorInput): FrameResult = synchronized(this) {
        val overallStartTime = SystemClock.uptimeMillis()
        val bitmap = (input as? DetectorInput.Bmp)?.bitmap ?: throw IllegalArgumentException("Requires Bmp")

        // 1. Preprocess
        val preprocessStart = SystemClock.uptimeMillis()
        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
        scaledBitmap.getPixels(intValues, 0, scaledBitmap.width, 0, 0, scaledBitmap.width, scaledBitmap.height)
        
        if (isFloatModel) {
            val floats = inputFloats!!
            val inv255 = 1.0f / 255.0f
            for (i in intValues.indices) {
                val pixel = intValues[i]
                floats[i * 3] = ((pixel shr 16) and 0xFF) * inv255
                floats[i * 3 + 1] = ((pixel shr 8) and 0xFF) * inv255
                floats[i * 3 + 2] = (pixel and 0xFF) * inv255
            }
            inputBuffer.rewind()
            inputBuffer.asFloatBuffer().put(floats)
        } else {
            val bytes = inputBytes!!
            for (i in intValues.indices) {
                val pixel = intValues[i]
                bytes[i * 3] = (((pixel shr 16) and 0xFF) - 128).toByte()
                bytes[i * 3 + 1] = (((pixel shr 8) and 0xFF) - 128).toByte()
                bytes[i * 3 + 2] = ((pixel and 0xFF) - 128).toByte()
            }
            inputBuffer.rewind()
            inputBuffer.put(bytes)
        }
        if (scaledBitmap != bitmap) scaledBitmap.recycle()
        val preprocessTime = SystemClock.uptimeMillis() - preprocessStart

        // 2. Inference
        val inferenceStart = SystemClock.uptimeMillis()
        scoreOutput.rewind()
        bboxOutput.rewind()
        interpreter.runForMultipleInputsOutputs(
            arrayOf(inputBuffer), 
            mapOf(scoreTensorIndex to scoreOutput, bboxTensorIndex to bboxOutput)
        )
        val inferenceTime = SystemClock.uptimeMillis() - inferenceStart

        // 3. Post-process (Decoding)
        val decodeStart = SystemClock.uptimeMillis()
        scoreOutput.rewind()
        bboxOutput.rewind()
        if (isFloatModel) {
            scoreOutput.asFloatBuffer().get(scoreArrayFloat!!)
            bboxOutput.asFloatBuffer().get(bboxArrayFloat!!)
        } else {
            scoreOutput.get(scoreArrayByte!!)
            bboxOutput.get(bboxArrayByte!!)
        }

        val scoreQP = interpreter.getOutputTensor(scoreTensorIndex).quantizationParams()
        val bboxQP = interpreter.getOutputTensor(bboxTensorIndex).quantizationParams()
        val labelsCount = labels.size

        val rawDetections = mutableListOf<Detection>()
        
        for (i in 0 until 8400) {
            var maxRaw = if (isFloatModel) -100f else -128f
            var classIdx = -1
            val offset = i * numModelClasses
            
            // Loop optimized: Move labelsCount check outside inner loop logic if possible
            // Actually, we must check labelsCount to avoid out of bounds in our label list
            for (c in 0 until labelsCount) {
                val s = if (isFloatModel) {
                    scoreArrayFloat!![offset + c]
                } else {
                    (scoreArrayByte!![offset + c].toInt() - scoreQP.zeroPoint) * scoreQP.scale
                }
                
                if (s > maxRaw) {
                    maxRaw = s
                    classIdx = c
                }
            }
            
            // Check against logit threshold to skip sigmoid/bbox math for 99% of locations
            if (maxRaw >= logitThreshold) {
                val finalScore = 1.0f / (1.0f + exp(-maxRaw))
                
                val dists = FloatArray(4) { side ->
                    var sum = 0.0
                    val sideOffset = i * 64 + side * 16
                    val bins = DoubleArray(16) { bin ->
                        val rawVal = if (isFloatModel) {
                            bboxArrayFloat!![sideOffset + bin]
                        } else {
                            (bboxArrayByte!![sideOffset + bin].toInt() - bboxQP.zeroPoint) * bboxQP.scale
                        }
                        val e = exp(rawVal.toDouble())
                        sum += e
                        e
                    }
                    var dot = 0.0
                    for (bin in 0 until 16) dot += (bins[bin] / sum) * bin
                    dot.toFloat()
                }
                
                val anchor = anchors[i]
                val l = (anchor.cx - dists[0]) * anchor.stride / inputSize
                val t = (anchor.cy - dists[1]) * anchor.stride / inputSize
                val r = (anchor.cx + dists[2]) * anchor.stride / inputSize
                val b = (anchor.cy + dists[3]) * anchor.stride / inputSize

                rawDetections.add(Detection(
                    RectF(l.coerceIn(0f, 1f), t.coerceIn(0f, 1f), r.coerceIn(0f, 1f), b.coerceIn(0f, 1f)),
                    classIdx, finalScore
                ))
            }
        }

        // Cap raw detections before NMS to avoid O(N^2) explosion
        val cappedRaw = if (rawDetections.size > 300) {
            rawDetections.sortedByDescending { it.score }.take(300)
        } else {
            rawDetections
        }
        
        val finalDetections = applyNms(cappedRaw).take(80)
        val decodeTime = SystemClock.uptimeMillis() - decodeStart
        val totalTime = SystemClock.uptimeMillis() - overallStartTime

        Log.d("YoloWorld", "Perf: Total=${totalTime}ms (Pre=${preprocessTime}ms, Inf=${inferenceTime}ms, Dec=${decodeTime}ms) RawCount=${rawDetections.size}")

        return FrameResult(finalDetections, bitmap.width, bitmap.height, totalTime)
    }

    private fun applyNms(detections: List<Detection>): List<Detection> {
        val sorted = detections.sortedByDescending { it.score }
        val results = mutableListOf<Detection>()
        val active = BooleanArray(sorted.size) { true }
        for (i in sorted.indices) {
            if (!active[i]) continue
            val d1 = sorted[i]
            results.add(d1)
            for (j in i + 1 until sorted.size) {
                if (active[j] && calculateIoU(d1.box, sorted[j].box) > nmsThreshold) {
                    active[j] = false
                }
            }
        }
        return results
    }

    private fun calculateIoU(b1: RectF, b2: RectF): Float {
        val inter = RectF(maxOf(b1.left, b2.left), maxOf(b1.top, b2.top), minOf(b1.right, b2.right), minOf(b1.bottom, b2.bottom))
        if (inter.left >= inter.right || inter.top >= inter.bottom) return 0f
        val areaInter = (inter.right - inter.left) * (inter.bottom - inter.top)
        return areaInter / ((b1.right - b1.left) * (b1.bottom - b1.top) + (b2.right - b2.left) * (b2.bottom - b2.top) - areaInter)
    }

    override fun close() {
        interpreter.close()
        gpuDelegate?.close()
    }

    private data class Anchor(val cx: Float, val cy: Float, val stride: Float)
}
