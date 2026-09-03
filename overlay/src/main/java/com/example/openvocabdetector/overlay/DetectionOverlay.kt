package com.example.openvocabdetector.overlay

import android.graphics.Paint
import android.graphics.Rect
import android.graphics.Typeface
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.openvocabdetector.detection.FrameResult

@Composable
fun DetectionOverlay(
    frameResult: FrameResult?,
    labels: List<String>,
    modifier: Modifier = Modifier
) {
    if (frameResult == null) return

    val density = LocalDensity.current
    val strokeWidth = with(density) { 1.5.dp.toPx() }
    val textSize = with(density) { 9.sp.toPx() }
    
    // Fixed palette of 12 hues
    val palette = remember {
        List(12) { i ->
            Color.hsv(i * 30f, 0.8f, 0.9f)
        }
    }

    val textPaint = remember(textSize) {
        Paint().apply {
            color = android.graphics.Color.WHITE
            this.textSize = textSize
            typeface = Typeface.MONOSPACE
            isAntiAlias = true
        }
    }

    val textBounds = remember { Rect() }

    Canvas(modifier = modifier.fillMaxSize()) {
        val viewWidth = size.width
        val viewHeight = size.height

        frameResult.detections.forEach { detection ->
            val color = palette[detection.labelIndex % palette.size]
            val rect = detection.box
            
            // source (normalized 0..1) -> view coordinates
            val left = rect.left * viewWidth
            val top = rect.top * viewHeight
            val right = rect.right * viewWidth
            val bottom = rect.bottom * viewHeight
            
            // Draw Box
            drawRect(
                color = color,
                topLeft = Offset(left, top),
                size = Size(right - left, bottom - top),
                style = Stroke(width = strokeWidth)
            )

            // Draw Label
            val label = labels.getOrNull(detection.labelIndex) ?: "unknown"
            val scoreText = "${(detection.score * 100).toInt()}"
            val fullLabel = "$label $scoreText"

            textPaint.getTextBounds(fullLabel, 0, fullLabel.length, textBounds)
            
            val labelWidth = textBounds.width().toFloat() + 8f
            val labelHeight = textBounds.height().toFloat() + 4f
            
            // Label Background
            drawRect(
                color = color.copy(alpha = 0.7f),
                topLeft = Offset(left, top - labelHeight),
                size = Size(labelWidth, labelHeight)
            )
            
            // Label Text
            drawContext.canvas.nativeCanvas.drawText(
                fullLabel,
                left + 4f,
                top - 4f,
                textPaint
            )
        }
    }
}
