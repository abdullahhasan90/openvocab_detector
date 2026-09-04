import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    id("com.android.library")
}

android {
    namespace = "com.example.openvocabdetector.detection"
    compileSdk = 37

    defaultConfig {
        minSdk = 28
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

kotlin {
    compilerOptions {
        jvmTarget.set(JvmTarget.JVM_17)
    }
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.tflite.task.vision) {
        exclude(group = "org.tensorflow", module = "tensorflow-lite-api")
        exclude(group = "com.google.ai.edge.litert", module = "litert-api")
    }
    implementation(libs.tflite.gpu) {
        exclude(group = "org.tensorflow", module = "tensorflow-lite-api")
        exclude(group = "com.google.ai.edge.litert", module = "litert-api")
    }
    implementation("org.tensorflow:tensorflow-lite:2.16.1")
}
