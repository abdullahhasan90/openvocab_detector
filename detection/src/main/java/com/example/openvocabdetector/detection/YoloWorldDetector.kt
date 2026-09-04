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

class YoloWorldDetector(
    context: Context,
    modelPath: String = "yoloworld_s_int8.tflite",
    private val scoreThreshold: Float = 0.15f,
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

    // Reusable buffers
    private val inputBuffer: ByteBuffer
    private val scoreOutput: ByteBuffer
    private val bboxOutput: ByteBuffer
    private val intValues: IntArray

    init {
        // Load Model
        val assetFileDescriptor = context.assets.openFd(modelPath)
        val mappedByteBuffer = FileInputStream(assetFileDescriptor.fileDescriptor).channel.map(
            FileChannel.MapMode.READ_ONLY, 
            assetFileDescriptor.startOffset, 
            assetFileDescriptor.declaredLength
        )
        
        val options = Interpreter.Options().apply {
            if (useGpu) {
                try {
                    gpuDelegate = GpuDelegate()
                    addDelegate(gpuDelegate)
                    Log.d("YoloWorld", "GPU delegate enabled")
                } catch (t: Throwable) {
                    Log.e("YoloWorld", "GPU Init failed", t)
                    setNumThreads(4)
                }
            } else {
                setNumThreads(4)
            }
        }
        
        try {
            interpreter = Interpreter(mappedByteBuffer, options)
        } catch (t: Throwable) {
            gpuDelegate?.close()
            throw t
        }

        // Model inspection
        val inputTensor = interpreter.getInputTensor(0)
        isFloatModel = inputTensor.dataType() == DataType.FLOAT32
        
        // Find output indices by shape (Score: [1, 8400, 280], BBox: [1, 8400, 64])
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

        // Precompute Anchors
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

        // Allocate buffers based on data type
        val bytesPerValue = if (isFloatModel) 4 else 1
        inputBuffer = ByteBuffer.allocateDirect(1 * inputSize * inputSize * 3 * bytesPerValue).apply { 
            order(ByteOrder.nativeOrder()) 
        }
        scoreOutput = ByteBuffer.allocateDirect(1 * 8400 * numModelClasses * bytesPerValue).apply { 
            order(ByteOrder.nativeOrder()) 
        }
        bboxOutput = ByteBuffer.allocateDirect(1 * 8400 * 64 * bytesPerValue).apply { 
            order(ByteOrder.nativeOrder()) 
        }
        intValues = IntArray(inputSize * inputSize)
    }

    override fun detect(input: DetectorInput): FrameResult = synchronized(this) {
        val startTime = SystemClock.uptimeMillis()
        val bitmap = (input as? DetectorInput.Bmp)?.bitmap ?: throw IllegalArgumentException("Requires Bmp")

        // 1. Preprocess
        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
        inputBuffer.rewind()
        scaledBitmap.getPixels(intValues, 0, scaledBitmap.width, 0, 0, scaledBitmap.width, scaledBitmap.height)
        
        for (pixel in intValues) {
            val r = (pixel shr 16) and 0xFF
            val g = (pixel shr 8) and 0xFF
            val b = pixel and 0xFF
            if (isFloatModel) {
                inputBuffer.putFloat(r / 255.0f)
                inputBuffer.putFloat(g / 255.0f)
                inputBuffer.putFloat(b / 255.0f)
            } else {
                inputBuffer.put((r - 128).toByte())
                inputBuffer.put((g - 128).toByte())
                inputBuffer.put((b - 128).toByte())
            }
        }
        if (scaledBitmap != bitmap) scaledBitmap.recycle()

        // 2. Inference
        scoreOutput.rewind()
        bboxOutput.rewind()
        interpreter.runForMultipleInputsOutputs(
            arrayOf(inputBuffer), 
            mapOf(scoreTensorIndex to scoreOutput, bboxTensorIndex to bboxOutput)
        )

        // 3. Decode
        val scoreQP = interpreter.getOutputTensor(scoreTensorIndex).quantizationParams()
        val bboxQP = interpreter.getOutputTensor(bboxTensorIndex).quantizationParams()

        val rawDetections = mutableListOf<Detection>()
        scoreOutput.rewind()

        for (i in 0 until 8400) {
            var maxScore = -100f
            var classIdx = -1
            
            for (c in 0 until numModelClasses) {
                val score = if (isFloatModel) {
                    scoreOutput.getFloat()
                } else {
                    (scoreOutput.get().toInt() - scoreQP.zeroPoint) * scoreQP.scale
                }
                
                if (c < labels.size && score > maxScore) {
                    maxScore = score
                    classIdx = c
                }
            }
            
            val finalScore = 1.0f / (1.0f + exp(-maxScore))
            if (finalScore >= scoreThreshold) {
                val dists = FloatArray(4) { side ->
                    var sum = 0.0
                    val bins = DoubleArray(16) { bin ->
                        val raw = if (isFloatModel) {
                            bboxOutput.getFloat((i * 64 + side * 16 + bin) * 4)
                        } else {
                            (bboxOutput.get(i * 64 + side * 16 + bin).toInt() - bboxQP.zeroPoint) * bboxQP.scale
                        }
                        val e = exp(raw.toDouble())
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

        val finalDetections = applyNms(rawDetections).take(80)
        FrameResult(finalDetections, bitmap.width, bitmap.height, SystemClock.uptimeMillis() - startTime)
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
