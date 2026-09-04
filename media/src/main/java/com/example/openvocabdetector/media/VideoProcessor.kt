package com.example.openvocabdetector.media

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.Build
import android.util.Log
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
            
            val nativeFps = 30 // Assumption
            val targetFps = 30 // Restored to full 30fps for smoothness
            val step = 1 // Process every frame
            
            val totalFrames = (durationMs / 1000f * nativeFps).toInt()
            val results = mutableMapOf<Int, FrameResult>()
            
            val matrix = if (rotation != 0) {
                Matrix().apply { postRotate(rotation.toFloat()) }
            } else null

            for (i in 0 until totalFrames step step) {
                try {
                    val timeUs = (i * 1000000L / nativeFps)
                    // Use getScaledFrameAtTime (API 27+) to reduce memory footprint by ~95%
                    val rawBitmap = retriever.getScaledFrameAtTime(
                        timeUs, 
                        MediaMetadataRetriever.OPTION_CLOSEST_SYNC, 
                        640, 
                        640
                    )
                    
                    if (rawBitmap != null) {
                        Log.d("VideoProcessor", "Processing frame $i / $totalFrames")
                        val rotatedBitmap = if (matrix != null) {
                            val rotated = Bitmap.createBitmap(rawBitmap, 0, 0, rawBitmap.width, rawBitmap.height, matrix, true)
                            rawBitmap.recycle()
                            rotated
                        } else {
                            rawBitmap
                        }
                        
                        val frameResult = detector.detect(DetectorInput.Bmp(rotatedBitmap))
                        results[i] = frameResult
                        
                        rotatedBitmap.recycle()
                    }
                } catch (_: Exception) {
                    // Log frame error but don't kill the whole process
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
