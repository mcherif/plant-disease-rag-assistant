package com.example.plantdiseasemobile

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import org.pytorch.executorch.ExecutionSession
import org.pytorch.executorch.Tensor
import java.nio.FloatBuffer

class ViTClassifier(
    context: Context,
    modelAsset: String = "vit_int8_executorch_qnnpack.pte",
    private val labels: List<String>
) {
    private val session: ExecutionSession

    init {
        val buffer = context.assets.open(modelAsset).readBytes()
        session = ExecutionSession(buffer)
    }

    fun predict(bitmap: Bitmap): Pair<String, Float> {
        val input = preprocess(bitmap)
        val outputs = session.run(mapOf("pixel_values" to input))
        val logits = outputs.values.first().data() as FloatArray
        val (idx, score) = logits.withIndex().maxBy { it.value }
        val label = labels.getOrElse(idx) { "unknown" }
        return label to score
    }

    private fun preprocess(src: Bitmap): Tensor {
        val target = Bitmap.createScaledBitmap(src, 224, 224, true)
        val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
        val std = floatArrayOf(0.229f, 0.224f, 0.225f)
        val buf = FloatBuffer.allocate(1 * 3 * 224 * 224)
        for (y in 0 until 224) {
            for (x in 0 until 224) {
                val c = target.getPixel(x, y)
                val r = (Color.red(c) / 255f - mean[0]) / std[0]
                val g = (Color.green(c) / 255f - mean[1]) / std[1]
                val b = (Color.blue(c) / 255f - mean[2]) / std[2]
                buf.put(r); buf.put(g); buf.put(b)
            }
        }
        buf.rewind()
        return Tensor.fromBlob(buf, longArrayOf(1, 3, 224, 224))
    }
}
