package com.example.openvocabdetector.overlay

import android.graphics.RectF
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size

/**
 * Handles coordinate transformation from normalized source image space (0..1)
 * to Compose View space, accounting for aspect ratio mismatch and cropping.
 */
class ViewportTransform(
    sourceWidth: Int,
    sourceHeight: Int,
    viewWidth: Float,
    viewHeight: Float,
    scaleType: ScaleType = ScaleType.FILL_CENTER
) {
    enum class ScaleType { FILL_CENTER, FIT_CENTER }

    private val scale: Float
    private val offsetX: Float
    private val offsetY: Float

    init {
        val sourceRatio = sourceWidth.toFloat() / sourceHeight.toFloat()
        val viewRatio = viewWidth / viewHeight

        when (scaleType) {
            ScaleType.FILL_CENTER -> {
                // Crop mode: The image fills the view, some parts are hidden.
                if (sourceRatio > viewRatio) {
                    // Source is wider than view (height is the limiting factor)
                    scale = viewHeight / sourceHeight
                    offsetX = (viewWidth - sourceWidth * scale) / 2f
                    offsetY = 0f
                } else {
                    // Source is taller than view (width is the limiting factor)
                    scale = viewWidth / sourceWidth
                    offsetX = 0f
                    offsetY = (viewHeight - sourceHeight * scale) / 2f
                }
            }
            ScaleType.FIT_CENTER -> {
                // Letterbox mode: The entire image is visible, some view space is empty.
                if (sourceRatio > viewRatio) {
                    scale = viewWidth / sourceWidth
                    offsetX = 0f
                    offsetY = (viewHeight - sourceHeight * scale) / 2f
                } else {
                    scale = viewHeight / sourceHeight
                    offsetX = (viewWidth - sourceWidth * scale) / 2f
                    offsetY = 0f
                }
            }
        }
    }

    /**
     * Transforms a normalized RectF (0..1) from source space to view space coordinates.
     */
    fun transform(rect: RectF, sourceW: Int, sourceH: Int): RectF {
        val left = (rect.left * sourceW * scale) + offsetX
        val top = (rect.top * sourceH * scale) + offsetY
        val right = (rect.right * sourceW * scale) + offsetX
        val bottom = (rect.bottom * sourceH * scale) + offsetY
        return RectF(left, top, right, bottom)
    }
}
