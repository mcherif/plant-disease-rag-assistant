# Mobile Demo App (Android, CPU/XNNPACK)

A minimal, good-looking Android app to demo the quantized ViT `.pte` on-device (no RAG, no camera pipeline yet). It runs fully offline using ExecuTorch + XNNPACK on CPU.

## Assets
- Quantized model: `mobile/assets/vit_int8_executorch_qnnpack.pte`
- Class map (41 classes): `mobile/assets/class_map.json`

Copy both into your Android module’s `app/src/main/assets/`.

## Tech stack
- Kotlin
- Jetpack Compose for UI
- ExecuTorch Android (CPU/XNNPACK)

## Gradle (app/build.gradle)
```gradle
plugins {
    id 'com.android.application'
    id 'org.jetbrains.kotlin.android'
}

android {
    namespace "com.example.plantdisease"
    compileSdk 34

    defaultConfig {
        applicationId "com.example.plantdisease"
        minSdk 24
        targetSdk 34
        versionCode 1
        versionName "0.1"
    }

    buildFeatures { compose true }
    composeOptions { kotlinCompilerExtensionVersion = "1.5.4" }
}

dependencies {
    implementation "androidx.core:core-ktx:1.12.0"
    implementation "androidx.compose.ui:ui:1.5.4"
    implementation "androidx.compose.material3:material3:1.2.1"
    implementation "androidx.activity:activity-compose:1.8.1"

    // ExecuTorch (pick a published version; replace if using a local AAR)
    implementation "org.pytorch:executorch-android:0.3.0"
}
```

> If you built ExecuTorch from source, drop the generated AAR into `app/libs/` and use `implementation files("libs/executorch-android.aar")` instead of the Maven coordinate.

## Inference helper (Kotlin)
```kotlin
package com.example.plantdisease

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
        // Resize + center-crop to 224x224
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
```

## Compose UI (landing page)
```kotlin
@Composable
fun LandingScreen(
    state: UiState,
    onPickImage: () -> Unit
) {
    val gradient = Brush.verticalGradient(
        colors = listOf(Color(0xFF0F9B8E), Color(0xFF0B5345))
    )
    Surface(
        modifier = Modifier.fillMaxSize(),
        color = Color.Transparent
    ) {
        Box(
            modifier = Modifier
                .background(gradient)
                .fillMaxSize()
                .padding(24.dp)
        ) {
            Column(
                verticalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                Text("Plant Care, Fast", style = MaterialTheme.typography.headlineLarge, color = Color.White)
                Text(
                    "Run the quantized ViT on-device. Pick a leaf photo and get an instant diagnosis.",
                    style = MaterialTheme.typography.bodyLarge,
                    color = Color(0xFFE6F2EF)
                )
                Button(
                    onClick = onPickImage,
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(54.dp),
                    colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF16A085))
                ) {
                    Text("Pick an image", color = Color.White)
                }
                state.preview?.let { bmp ->
                    Image(bitmap = bmp.asImageBitmap(), contentDescription = null,
                          modifier = Modifier
                              .fillMaxWidth()
                              .height(200.dp)
                              .clip(RoundedCornerShape(16.dp)),
                          contentScale = ContentScale.Crop)
                }
                state.prediction?.let { (label, score) ->
                    Text(
                        "Prediction: $label",
                        style = MaterialTheme.typography.titleLarge,
                        color = Color.White
                    )
                    LinearProgressIndicator(
                        progress = { score.coerceIn(0f, 1f) },
                        modifier = Modifier.fillMaxWidth(),
                        color = Color(0xFF1ABC9C)
                    )
                }
            }
        }
    }
}
```

`UiState` is a simple data class holding the selected bitmap and prediction; wire it up with `rememberLauncherForActivityResult` (gallery picker), then call `classifier.predict()` on the selected image.

## Run it on your Oppo Reno8T
1. Enable Developer Options + USB debugging.
2. Build & install from Android Studio (or `./gradlew installDebug`).
3. Open the app, tap “Pick an image,” select a leaf photo, and view the prediction.

## Next steps (optional)
- Add a camera capture flow.
- Show top-3 predictions with confidence.
- Add a “learn more” link to open a browser for the predicted disease.
- Bundle a few sample images in assets for instant offline demo.
