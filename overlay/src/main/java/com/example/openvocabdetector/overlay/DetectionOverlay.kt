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
    modifier: Modifier = Modifier,
    scaleType: ViewportTransform.ScaleType = ViewportTransform.ScaleType.FILL_CENTER
) {
    if (frameResult == null) return

    val density = LocalDensity.current
    val strokeWidth = with(density) { 1.5.dp.toPx() }
    val textSize = with(density) { 9.sp.toPx() }
    
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
        
        // Hard clip to view boundaries
        drawContext.canvas.clipRect(0f, 0f, viewWidth, viewHeight)

        // Calculate transform once per frame
        val transform = ViewportTransform(
            sourceWidth = frameResult.sourceWidth,
            sourceHeight = frameResult.sourceHeight,
            viewWidth = viewWidth,
            viewHeight = viewHeight,
            scaleType = scaleType
        )

        frameResult.detections.forEach { detection ->
            val color = palette[detection.labelIndex % palette.size]
            
            // Transform normalized source coordinates to View coordinates
            val transformedRect = transform.transform(
                detection.box, 
                frameResult.sourceWidth, 
                frameResult.sourceHeight
            )
            
            // STRICT VIEW CLIPPING
            val left = transformedRect.left.coerceIn(0f, viewWidth)
            val top = transformedRect.top.coerceIn(0f, viewHeight)
            val right = transformedRect.right.coerceIn(0f, viewWidth)
            val bottom = transformedRect.bottom.coerceIn(0f, viewHeight)
            
            // Only draw if the clipped box is still valid
            if (right <= left || bottom <= top) return@forEach

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
            
            // Position label: above the box, but if too high, push inside the box
            val labelTop = if (top - labelHeight < 0) top else top - labelHeight
            
            // Label Background
            drawRect(
                color = color.copy(alpha = 0.7f),
                topLeft = Offset(left, labelTop),
                size = Size(labelWidth, labelHeight)
            )
            
            // Label Text
            drawContext.canvas.nativeCanvas.drawText(
                fullLabel,
                left + 4f,
                labelTop + labelHeight - 4f, // Adjusted baseline for monospace
                textPaint
            )
        }
    }
}
