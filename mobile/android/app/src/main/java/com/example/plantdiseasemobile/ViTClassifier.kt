package com.example.plantdiseasemobile

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import org.pytorch.executorch.Module
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Tensor
import java.io.File

class ViTClassifier(
    context: Context,
    modelAsset: String = "vit_int8_executorch_qnnpack.pte",
    private val labels: List<String>
) {
    private val module: Module

    init {
        val modelFile = File(context.filesDir, modelAsset)
        if (!modelFile.exists()) {
            // Copy once from assets to an internal file for Module.load
            context.assets.open(modelAsset).use { input ->
                modelFile.outputStream().use { output -> input.copyTo(output) }
            }
        }
        module = Module.load(modelFile.absolutePath)
    }

    fun predict(bitmap: Bitmap): Pair<String, Float> {
        val input = preprocess(bitmap)
        val outputs = module.forward(EValue.from(input))
        val logits = outputs.first().toTensor().getDataAsFloatArray()
        val (idx, score) = logits.withIndex().maxBy { it.value }
        val label = labels.getOrElse(idx) { "unknown" }
        return label to score
    }

    private fun preprocess(src: Bitmap): Tensor {
        // Match the HF ViTImageProcessor: resize shortest side to 256, then center-crop 224.
        val (w, h) = src.width to src.height
        val scale = 256f / minOf(w, h)
        val newW = (w * scale).toInt()
        val newH = (h * scale).toInt()
        val resized = Bitmap.createScaledBitmap(src, newW, newH, true)
        val x0 = (newW - 224) / 2
        val y0 = (newH - 224) / 2
        val target = Bitmap.createBitmap(resized, x0, y0, 224, 224)

        val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
        val std = floatArrayOf(0.229f, 0.224f, 0.225f)
        val data = FloatArray(3 * 224 * 224)
        val planeSize = 224 * 224
        for (y in 0 until 224) {
            for (x in 0 until 224) {
                val c = target.getPixel(x, y)
                val base = y * 224 + x
                data[base] = (Color.red(c) / 255f - mean[0]) / std[0]           // R plane
                data[planeSize + base] = (Color.green(c) / 255f - mean[1]) / std[1] // G plane
                data[2 * planeSize + base] = (Color.blue(c) / 255f - mean[2]) / std[2] // B plane
            }
        }
        return Tensor.fromBlob(data, longArrayOf(1, 3, 224, 224))
    }
}
