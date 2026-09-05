package com.example.openvocabdetector.media

import android.content.Context
import android.graphics.*
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.util.Log
import androidx.annotation.OptIn
import androidx.media3.common.Effect
import androidx.media3.common.MediaItem
import androidx.media3.common.MimeTypes
import androidx.media3.common.util.UnstableApi
import androidx.media3.effect.OverlayEffect
import androidx.media3.effect.TextureOverlay
import androidx.media3.effect.BitmapOverlay
import androidx.media3.common.util.Size
import androidx.media3.transformer.EditedMediaItem
import androidx.media3.transformer.Effects
import androidx.media3.transformer.ProgressHolder
import androidx.media3.transformer.Transformer
import com.example.openvocabdetector.detection.Detector
import com.example.openvocabdetector.detection.DetectorInput
import com.example.openvocabdetector.detection.FrameResult
import com.google.common.collect.ImmutableList
import com.google.gson.*
import com.google.gson.reflect.TypeToken
import com.google.gson.stream.JsonReader
import com.google.gson.stream.JsonWriter
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.withContext
import java.io.File

@OptIn(UnstableApi::class)
class VideoProcessor(private val detector: Detector) {

    private val gson: Gson = GsonBuilder()
        .registerTypeAdapter(RectF::class.java, RectFAdapter())
        .create()

    private val resultsType = object : TypeToken<Map<Int, FrameResult>>() {}.type

    fun saveMetadata(results: Map<Int, FrameResult>, outputFile: File) {
        try {
            val json = gson.toJson(results, resultsType)
            outputFile.writeText(json)
        } catch (e: Exception) {
            Log.e("VideoProcessor", "Failed to save metadata", e)
        }
    }

    fun loadMetadata(inputFile: File): Map<Int, FrameResult>? {
        return try {
            if (!inputFile.exists()) return null
            gson.fromJson(inputFile.readText(), resultsType)
        } catch (e: Exception) {
            null
        }
    }

    suspend fun exportBurnedVideo(
        context: Context,
        inputUri: Uri,
        outputFile: File,
        results: Map<Int, FrameResult>,
        labels: List<String>,
        onProgress: (Float) -> Unit
    ): Boolean = withContext(Dispatchers.Main) {
        try {
            val transformer = Transformer.Builder(context)
                .setVideoMimeType(MimeTypes.VIDEO_H264)
                .build()

            val hudOverlay = HudOverlay(results, labels)
            val overlayEffect = OverlayEffect(ImmutableList.of(hudOverlay as TextureOverlay))

            val editedMediaItem = EditedMediaItem.Builder(MediaItem.fromUri(inputUri))
                .setEffects(
                    Effects(
                        ImmutableList.of(),
                        ImmutableList.of(overlayEffect as Effect)
                    )
                )
                .build()

            transformer.start(editedMediaItem, outputFile.absolutePath)

            while (true) {
                val progressHolder = ProgressHolder()
                val state = transformer.getProgress(progressHolder)
                if (state == Transformer.PROGRESS_STATE_NOT_STARTED) break
                if (state == Transformer.PROGRESS_STATE_AVAILABLE) {
                    onProgress(progressHolder.progress / 100f)
                }
                delay(200)
            }
            true
        } catch (e: Exception) {
            Log.e("VideoProcessor", "Export failed", e)
            false
        }
    }

    fun processVideo(context: Context, videoUri: Uri): Flow<ProcessingStatus> = flow {
        val retriever = MediaMetadataRetriever()
        try {
            retriever.setDataSource(context, videoUri)
            val durationMs = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLong() ?: 0L
            val rotation = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION)?.toInt() ?: 0
            val nativeFps = 30
            val totalFrames = (durationMs / 1000f * nativeFps).toInt()
            val results = mutableMapOf<Int, FrameResult>()
            val matrix = if (rotation != 0) Matrix().apply { postRotate(rotation.toFloat()) } else null

            for (i in 0 until totalFrames) {
                try {
                    val timeUs = (i * 1000000L / nativeFps)
                    val rawBitmap = retriever.getScaledFrameAtTime(timeUs, MediaMetadataRetriever.OPTION_CLOSEST_SYNC, 640, 640)
                    if (rawBitmap != null) {
                        val rotatedBitmap = if (matrix != null) {
                            Bitmap.createBitmap(rawBitmap, 0, 0, rawBitmap.width, rawBitmap.height, matrix, true).also { rawBitmap.recycle() }
                        } else rawBitmap
                        results[i] = detector.detect(DetectorInput.Bmp(rotatedBitmap))
                        rotatedBitmap.recycle()
                    }
                } catch (_: Exception) {}
                emit(ProcessingStatus.Progress(i.toFloat() / totalFrames))
            }
            emit(ProcessingStatus.Complete(results))
        } catch (e: Exception) {
            emit(ProcessingStatus.Error(e))
        } finally {
            retriever.release()
        }
    }.flowOn(Dispatchers.Default)
}

sealed interface ProcessingStatus {
    data class Progress(val progress: Float) : ProcessingStatus
    data class Complete(val results: Map<Int, FrameResult>) : ProcessingStatus
    data class Error(val throwable: Throwable) : ProcessingStatus
}

private class RectFAdapter : TypeAdapter<RectF>() {
    override fun write(out: JsonWriter, value: RectF?) {
        if (value == null) { out.nullValue(); return }
        out.beginObject().name("l").value(value.left.toDouble()).name("t").value(value.top.toDouble())
            .name("r").value(value.right.toDouble()).name("b").value(value.bottom.toDouble()).endObject()
    }
    override fun read(`in`: JsonReader): RectF {
        `in`.beginObject()
        var l = 0f; var t = 0f; var r = 0f; var b = 0f
        while (`in`.hasNext()) {
            when (`in`.nextName()) {
                "l" -> l = `in`.nextDouble().toFloat()
                "t" -> t = `in`.nextDouble().toFloat()
                "r" -> r = `in`.nextDouble().toFloat()
                "b" -> b = `in`.nextDouble().toFloat()
            }
        }
        `in`.endObject()
        return RectF(l, t, r, b)
    }
}

@OptIn(UnstableApi::class)
private class HudOverlay(
    private val results: Map<Int, FrameResult>,
    private val labels: List<String>
) : BitmapOverlay() {
    private val paint = Paint().apply { style = Paint.Style.STROKE; strokeWidth = 4f; isAntiAlias = true }
    private val textPaint = Paint().apply { color = Color.WHITE; textSize = 24f; typeface = Typeface.MONOSPACE; isAntiAlias = true }
    private val palette = List(12) { Color.HSVToColor(floatArrayOf(it * 30f, 0.8f, 0.9f)) }

    override fun getBitmap(presentationTimeUs: Long): Bitmap {
        // Assume 30fps
        val frameIndex = (presentationTimeUs / 1000000f * 30).toInt()
        
        // Find best frame with 166ms lookback for stickiness
        var foundFrame: FrameResult? = null
        for (offset in 0..5) {
            val target = frameIndex - offset
            if (results.containsKey(target)) {
                foundFrame = results[target]
                break
            }
        }

        // Create a transparent bitmap to draw the HUD
        // Size should match video, but for now we use model size 640x640
        // and Transformer will scale it.
        val bitmap = Bitmap.createBitmap(640, 640, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bitmap)

        foundFrame?.let { frame ->
            frame.detections.forEach { detection ->
                val color = palette[detection.labelIndex % palette.size]
                paint.color = color
                
                val l = detection.box.left * 640
                val t = detection.box.top * 640
                val r = detection.box.right * 640
                val b = detection.box.bottom * 640
                
                canvas.drawRect(l, t, r, b, paint)
                
                val label = labels.getOrNull(detection.labelIndex) ?: "unknown"
                val text = "$label ${(detection.score * 100).toInt()}"
                
                val textBgPaint = Paint().apply { this.color = color; alpha = 180 }
                canvas.drawRect(l, t - 30, l + text.length * 15, t, textBgPaint)
                canvas.drawText(text, l, t - 5, textPaint)
            }
        }
        return bitmap
    }
}
