package com.example.plantdiseasemobile

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.compose.setContent
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.getValue
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import com.example.plantdiseasemobile.ui.theme.PlantDiseaseTheme
import java.io.InputStream

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            PlantDiseaseTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    DemoScreen()
                }
            }
        }
    }
}

@Composable
fun DemoScreen() {
    val context = LocalContext.current
    val labels = remember { loadLabels(context) }
    val classifier = remember { ViTClassifier(context, labels = labels) }
    var state by remember { mutableStateOf(UiState()) }

    val pickImageLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let {
            val bmp = uriToBitmap(context, it)
            if (bmp != null) {
                val (label, score) = classifier.predict(bmp)
                state = state.copy(preview = bmp, prediction = label to score)
            }
        }
    }

    val gradient = Brush.verticalGradient(
        listOf(Color(0xFF0F9B8E), Color(0xFF0B5345))
    )

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(gradient)
            .padding(24.dp)
    ) {
        Column(
            verticalArrangement = Arrangement.spacedBy(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(
                "Plant Care, Fast",
                style = MaterialTheme.typography.headlineLarge,
                color = Color.White
            )
            Text(
                "Run the quantized ViT fully on-device. Pick a leaf photo and get an instant diagnosis.",
                style = MaterialTheme.typography.bodyLarge,
                color = Color(0xFFE6F2EF)
            )
            Button(
                onClick = { pickImageLauncher.launch("image/*") },
                modifier = Modifier
                    .fillMaxWidth()
                    .height(54.dp),
                colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF16A085))
            ) {
                Text("Pick an image", color = Color.White)
            }
            state.preview?.let { bmp ->
                Image(
                    bitmap = bmp.asImageBitmap(),
                    contentDescription = null,
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(220.dp)
                        .clip(RoundedCornerShape(20.dp)),
                    contentScale = ContentScale.Crop
                )
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
                Text(
                    "Confidence: ${(score * 100).coerceIn(0f, 100f).toInt()}%",
                    style = MaterialTheme.typography.bodyMedium,
                    color = Color(0xFFCFF4EE)
                )
            }
            Spacer(modifier = Modifier.height(12.dp))
        }
    }
}

private fun loadLabels(context: android.content.Context): List<String> {
    return runCatching {
        val text = context.assets.open("class_map.json").bufferedReader().use { it.readText() }
        val json = org.json.JSONArray(text)
        (0 until json.length()).map { idx -> json.getString(idx) }
    }.getOrDefault(emptyList())
}

private fun uriToBitmap(context: android.content.Context, uri: Uri) =
    runCatching {
        val stream: InputStream? = context.contentResolver.openInputStream(uri)
        stream.use { BitmapFactory.decodeStream(it) }
    }.getOrNull()
