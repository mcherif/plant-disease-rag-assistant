package com.example.plantdiseasemobile

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor
import java.io.File

class ViTClassifier(
    context: Context,
    modelAsset: String = "vit_fp32_executorch.pte",
    private val labels: List<String>
) {
    private val tag = "ViTClassifier"
    private val appContext = context.applicationContext
    private val module: Module
    private val assetName: String

    init {
        // Prefer caller-specified asset; no silent fallbacks to avoid stale models.
        assetName = listOf(modelAsset).firstOrNull { assetExists(it) }
            ?: error("Model asset not found: $modelAsset")

        val modelFile = File(appContext.filesDir, assetName)
        // Always refresh the model from assets to avoid stale files
        appContext.assets.open(assetName).use { input ->
            modelFile.outputStream().use { output -> input.copyTo(output) }
        }
        Log.i(
            tag,
            "Loaded asset=$assetName, size=${modelFile.length()} bytes, labels=${labels.size}"
        )
        module = Module.load(modelFile.absolutePath)
    }

    suspend fun predict(bitmap: Bitmap): Pair<String, Float> = withContext(Dispatchers.Default) {
        Log.i(tag, "predict start (asset=$assetName, size=${bitmap.width}x${bitmap.height})")
        val top3 = try {
            val input = loadInputTensor(bitmap)
            Log.i(tag, "forward start")
            val outputs = withTimeout(30_000) { module.forward(EValue.from(input)) }
            Log.i(tag, "forward end")
            val logits = outputs.first().toTensor().getDataAsFloatArray()
            Log.d(tag, "Logits: ${logits.joinToString()}")

            logits.withIndex()
                .sortedByDescending { it.value }
                .take(3)
                .map { (idx, score) ->
                    labels.getOrElse(idx) { "unknown" } to score
                }
        } catch (e: Exception) {
            Log.e(tag, "predict failed: ${e.message}", e)
            throw e
        }

        Log.i(tag, "Top 3 predictions: $top3")
        top3.first()
    }

    private fun preprocess(src: Bitmap): Tensor {
        // Mirror the training/export pipeline (ViTImageProcessor): resize to 224x224,
        // rescale to [0,1], then normalize with mean/std of 0.5 per channel.
        val target = Bitmap.createScaledBitmap(src, 224, 224, true)
        val mean = 0.5f
        val std = 0.5f
        val data = FloatArray(3 * 224 * 224)
        val planeSize = 224 * 224
        for (y in 0 until 224) {
            for (x in 0 until 224) {
                val c = target.getPixel(x, y)
                val base = y * 224 + x
                data[base] = (Color.red(c) / 255f - mean) / std           // R plane
                data[planeSize + base] = (Color.green(c) / 255f - mean) / std // G plane
                data[2 * planeSize + base] = (Color.blue(c) / 255f - mean) / std // B plane
            }
        }
        return Tensor.fromBlob(data, longArrayOf(1, 3, 224, 224))
    }

    private fun loadInputTensor(bitmap: Bitmap): Tensor {
        val forced = File(appContext.filesDir, "olive.bin")
        if (forced.exists()) {
            val bytes = forced.readBytes()
            // expect 1x3x224x224 float32 => 602112 bytes
            val buf = java.nio.ByteBuffer.wrap(bytes).order(java.nio.ByteOrder.LITTLE_ENDIAN)
            val floats = FloatArray(bytes.size / 4)
            buf.asFloatBuffer().get(floats)
            Log.i(tag, "Using forced tensor from ${forced.absolutePath}")
            return Tensor.fromBlob(floats, longArrayOf(1, 3, 224, 224))
        }
        // existing path: preprocess image to tensor
        return preprocess(bitmap)
    }

    private fun assetExists(name: String): Boolean {
        return runCatching {
            appContext.assets.open(name).close()
            true
        }.getOrDefault(false)
    }
}
