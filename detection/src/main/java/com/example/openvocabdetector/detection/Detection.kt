package com.example.openvocabdetector.detection

import android.graphics.RectF

data class Detection(
    /** Normalized 0..1 in SOURCE IMAGE space. */
    val box: RectF,
    val labelIndex: Int,
    val score: Float,
)

data class FrameResult(
    val detections: List<Detection>,
    val sourceWidth: Int,
    val sourceHeight: Int,
    val inferenceTimeMs: Long,
)
