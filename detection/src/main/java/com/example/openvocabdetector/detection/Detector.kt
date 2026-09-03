package com.example.openvocabdetector.detection

import android.graphics.Bitmap
import java.io.Closeable
import java.nio.ByteBuffer

interface Detector : Closeable {
    val labels: List<String>
    val inputSize: Int

    /** Blocking. Caller controls threading. */
    fun detect(input: DetectorInput): FrameResult
}

sealed interface DetectorInput {
    @JvmInline value class Bmp(val bitmap: Bitmap) : DetectorInput
    data class Yuv(
        val planes: YuvPlanes,
        val rotationDegrees: Int,
        val width: Int,
        val height: Int
    ) : DetectorInput
}

data class YuvPlanes(
    val y: ByteBuffer,
    val u: ByteBuffer,
    val v: ByteBuffer,
    val yStride: Int,
    val uvStride: Int,
    val uvPixelStride: Int
)
