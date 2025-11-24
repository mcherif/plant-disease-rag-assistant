package com.example.plantdiseasemobile

import android.graphics.Bitmap

data class UiState(
    val preview: Bitmap? = null,
    val prediction: Pair<String, Float>? = null
)
