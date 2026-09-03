package com.example.openvocabdetector.detection

import android.graphics.RectF
import kotlin.random.Random

class FakeDetector(override val labels: List<String>) : Detector {
    override val inputSize: Int = 640
    private val random = Random(42)

    override fun detect(input: DetectorInput): FrameResult {
        // Generate a busy set of boxes
        val detections = List(25) {
            val w = random.nextFloat() * 0.3f + 0.05f
            val h = random.nextFloat() * 0.3f + 0.05f
            val x = random.nextFloat() * (1f - w)
            val y = random.nextFloat() * (1f - h)
            
            Detection(
                box = RectF(x, y, x + w, y + h),
                labelIndex = random.nextInt(labels.size),
                score = random.nextFloat() * 0.6f + 0.3f
            )
        }
        
        // Simulate inference time
        Thread.sleep(15)

        // Use input size if available, otherwise fallback
        val (sw, sh) = when (input) {
            is DetectorInput.Bmp -> input.bitmap.width to input.bitmap.height
            is DetectorInput.Yuv -> input.width to input.height
        }

        return FrameResult(
            detections = detections,
            sourceWidth = sw,
            sourceHeight = sh,
            inferenceTimeMs = 15
        )
    }

    override fun close() {}
}
