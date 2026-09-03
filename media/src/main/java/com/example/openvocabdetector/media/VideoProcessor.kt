package com.example.openvocabdetector.media

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.media.MediaMetadataRetriever
import android.net.Uri
import com.example.openvocabdetector.detection.Detector
import com.example.openvocabdetector.detection.DetectorInput
import com.example.openvocabdetector.detection.FrameResult
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn

class VideoProcessor(private val detector: Detector) {

    /** 
     * Processes a video file and emits a Map of frame results.
     */
    fun processVideo(context: Context, videoUri: Uri): Flow<ProcessingStatus> = flow {
        val retriever = MediaMetadataRetriever()
        try {
            retriever.setDataSource(context, videoUri)
            val durationMs = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLong() ?: 0L
            val rotation = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION)?.toInt() ?: 0
            
            val fps = 30 // Assumption for P0
            val totalFrames = (durationMs / 1000f * fps).toInt()
            
            val results = mutableMapOf<Int, FrameResult>()
            
            val matrix = if (rotation != 0) {
                Matrix().apply { postRotate(rotation.toFloat()) }
            } else null

            for (i in 0 until totalFrames) {
                val timeUs = (i * 1000000L / fps)
                val rawBitmap = retriever.getFrameAtTime(timeUs, MediaMetadataRetriever.OPTION_CLOSEST)
                
                if (rawBitmap != null) {
                    val rotatedBitmap = if (matrix != null) {
                        Bitmap.createBitmap(rawBitmap, 0, 0, rawBitmap.width, rawBitmap.height, matrix, true)
                    } else {
                        rawBitmap
                    }
                    
                    val frameResult = detector.detect(DetectorInput.Bmp(rotatedBitmap))
                    results[i] = frameResult
                }
                
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
