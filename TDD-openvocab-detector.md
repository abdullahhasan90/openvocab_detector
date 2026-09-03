# Technical Design Document
## Open-Vocabulary Real-Time Object Detection — Android

**Status:** Draft
**Platform:** Android 9+ (API 28), arm64-v8a only
**Language:** Kotlin
**Author:** —

---

## 1. What this is

A camera app that draws bounding boxes and labels over everything it can name, at high
density and low confidence, in a deliberately over-saturated HUD style. The aesthetic goal
is "machine perception made visible" — not a clean, correct detector, but a busy one.

Two operating modes:

- **Capture mode (v0):** record ~10s of video, process it after the fact, play it back with
  the overlay. No latency constraint, so a heavier model at higher accuracy is affordable.
- **Live mode (v1):** real-time overlay on the camera preview at 20–30fps.

### 1.1 Explicit non-goals

- Accuracy. Mislabels are a feature. No effort will be spent on precision tuning.
- Temporal smoothing. Boxes flickering frame-to-frame is on-brand and will not be fixed.
- Tracking / persistent object identity. Not needed for the visual effect.
- iOS.
- Any server component. Everything runs on device.

### 1.2 Design principles

1. **Density over correctness.** An empty screen is the failure mode, not a wrong label.
2. **The vocabulary is the product.** Model choice is swappable; the class list is the
   design decision that distinguishes this app from every other YOLO demo.
3. **Ship v0 before optimizing v1.** The offline path exists to find the real bottleneck,
   which is probably rendering, not inference.

---

## 2. Phasing

| Phase | Model | Mode | Purpose |
|---|---|---|---|
| **P0** | EfficientDet-Lite0 (COCO, 80 cls) | Capture | End-to-end pipeline. Prove camera → frames → inference → overlay → playback. |
| **P1** | YOLO-World-S v2, int8, ~270 cls | Capture | Swap in real vocabulary. Validate the offline-vocabulary export path. |
| **P2** | Same | Live | Real-time preview overlay. Where the perf work happens. |
| **P3** | Same | Capture | Export burned-in video to gallery. |
| **P4** | Same | Both | "Terminator lines" — vectors from box vertices to a screen anchor. Cosmetic. |

P0 exists to be thrown away. Do not invest in it beyond making the interfaces right.

---

## 3. Architecture

### 3.1 Module boundaries

```
:app            — UI, navigation, camera screens, permissions
:detection      — Detector interface + implementations, decode, NMS
:overlay        — Rendering of boxes/labels/lines. Knows nothing about ML.
:media          — Recording, frame extraction, video export
```

`:overlay` depending on `:detection` is allowed (for the `Detection` type only).
`:detection` depending on `:overlay` is a build failure.

### 3.2 The core seam

Everything hangs off one interface. This is the single most important decision in the
document, because the model will be replaced at least three times.

```kotlin
data class Detection(
    val box: RectF,        // normalized 0..1 in SOURCE IMAGE space, not view space
    val labelIndex: Int,
    val score: Float,
)

data class FrameResult(
    val detections: List<Detection>,
    val sourceWidth: Int,
    val sourceHeight: Int,
    val inferenceTimeMs: Long,
)

interface Detector : Closeable {
    val labels: List<String>
    val inputSize: Int

    /** Blocking. Caller controls threading. */
    fun detect(image: DetectorInput): FrameResult
}

/** Wrapper so callers can pass either a Bitmap or a CameraX ImageProxy
 *  without :detection needing to know which. */
sealed interface DetectorInput {
    @JvmInline value class Bmp(val bitmap: Bitmap) : DetectorInput
    class Yuv(val planes: YuvPlanes, val rotationDegrees: Int) : DetectorInput
}
```

**Normalized coordinates in source space** is non-negotiable. If detections come back in
pixel coordinates of the letterboxed 640×640 tensor, every consumer has to know about
letterbox padding, and you will get the aspect-ratio bug in three separate places. Undo the
letterbox inside the detector, before it returns.

### 3.3 Implementations

- `EfficientDetLiteDetector` — P0. Uses LiteRT Task Library `ObjectDetector`, which does
  decode and NMS internally. ~20 lines of real code.
- `YoloWorldDetector` — P1+. Raw `Interpreter`, manual decode and NMS. See §5.
- `FakeDetector` — returns deterministic garbage boxes. Used for overlay development and
  instrumented tests. Build this early; it lets `:overlay` be developed without a model.

---

## 4. Model preparation (offline, Python)

This runs on a workstation and produces two artifacts. It is checked into the repo under
`tools/model/` and is part of the build's source of truth.

### 4.1 Single source for vocabulary

```python
# tools/model/vocabulary.py
CLASSES = [ ... 270 entries ... ]
```

Both artifacts are generated from this one list, in one script run:

```python
from ultralytics import YOLO
from vocabulary import CLASSES

model = YOLO("yolov8s-worldv2.pt")
model.set_classes(CLASSES)          # reparameterizes CLIP embeddings into weights
model.save("yoloworld_s_custom.pt")

with open("out/labels.txt", "w") as f:
    f.write("\n".join(CLASSES))
```

**Failure mode this prevents:** `labels.txt` order drifting from the baked-in embedding
order. There is no runtime validation of this. A mismatch produces a working app that is
confidently, consistently wrong — every dog labeled "microwave" — with no error, no crash,
and no obvious cause. Never hand-edit either artifact.

Additionally emit `out/model_meta.json` with `{ sha256, num_classes, input_size,
quantization, generated_at }`. The app asserts `labels.size == meta.num_classes` on startup
and refuses to launch on mismatch. Cheap insurance.

### 4.2 Export chain

```
yoloworld_s_custom.pt
  → export_onnx.py --custom-text vocabulary.txt --opset 11 --without-bbox-decoder
  → onnx-simplifier
  → onnx2tf -oiqt -cind ... (int8, ~100-image calibration set)
  → yoloworld_s_int8.tflite
```

Two constraints from upstream that are easy to miss:

1. **Must use the reparameterized model.** Non-reparameterized YOLO-World keeps a live text
   encoder in the graph and will not quantize sensibly.
2. **`--without-bbox-decoder` is mandatory.** The bbox decoder does not survive int8
   quantization. Consequence: the app receives raw DFL distribution outputs and must decode
   them itself. This is the largest single chunk of novel work in the project. See §5.

Calibration set: ~100 images, but **not** COCO. Use images from the actual target domain —
handheld phone photos of streets, rooms, clutter, at the camera's real resolution and noise
profile. Quantization ranges derived from clean COCO photos will be wrong for a phone
sensor in low light.

### 4.3 Validation gate

The export script is not done until `tools/model/validate.py` passes: run the fp32 `.pt`
and the int8 `.tflite` over the same 50 held-out images and compare. Gate on **recall at
conf ≥ 0.25**, not mAP — this app cares whether boxes appear at all, not whether they're
tight. Accept up to ~15% recall loss. Beyond that, fall back to fp16 and eat the size.

Also emit the raw output tensor shape and dtype into `model_meta.json`. The Kotlin decoder
reads it rather than hardcoding.

---

## 5. Inference path (Kotlin)

### 5.1 Preprocessing

CameraX `ImageAnalysis` delivers `YUV_420_888`. Path:

```
ImageProxy (YUV_420_888)
  → RGB conversion
  → rotate by imageInfo.rotationDegrees
  → letterbox to 640×640, gray fill (114,114,114)
  → int8 quantized input buffer
```

Do the YUV→RGB conversion in **RenderScript-replacement territory**: use
`android.graphics.YuvImage` only as a fallback. Preferred is a C++/NEON path or, simplest,
`ImageProxy.toBitmap()` (available in CameraX 1.3+) and accept the cost for P0/P1. Revisit
in P2 if profiling says preprocessing exceeds ~5ms.

Reuse buffers. Allocate `ByteBuffer`s once at detector construction, never per frame.
Per-frame allocation of a 640×640×3 buffer at 30fps is ~35MB/s of garbage and will produce
visible GC stutter.

### 5.2 Output decode — the hard part

Because of `--without-bbox-decoder`, the model outputs two tensors:

- `bboxes`: `[1, 8400, 64]` — DFL distributions, 4 sides × 16 bins
- `scores`: `[1, 8400, 270]`

Steps, per frame:

1. **Dequantize** using the tensor's `QuantizationParams` (scale, zeroPoint). Do not
   assume symmetric quantization; read the params from the interpreter.
2. **Score threshold first, before decoding boxes.** Filter to anchors with any class score
   ≥ 0.3. Typically leaves 50–300 of 8400. Decoding all 8400 boxes and then filtering wastes
   ~95% of the work — this is the difference between 8ms and 45ms of post-processing.
3. **DFL decode** on survivors only: softmax over each 16-bin distribution, dot with
   `[0..15]`, gives distance from anchor centre to each box side in stride units. Multiply
   by the anchor's stride (8/16/32 for the three feature levels).
4. **Anchor grid** is deterministic — precompute the `(cx, cy, stride)` table once at
   construction for 640×640. Do not rebuild per frame.
5. **NMS.** Class-agnostic, IoU 0.5. Class-*wise* NMS would let "man", "person", and "shirt"
   all box the same torso — which is arguably the desired aesthetic. **Make this a runtime
   toggle and decide by looking at it.** Cap output at ~80 boxes for render sanity.

Write the decode against a golden fixture: dump one frame's raw output tensors from the
Python side, check them into test resources, and unit-test the Kotlin decoder against the
Python decoder's output. This is the one piece of logic where a subtle error produces
plausible-looking-but-wrong boxes rather than an obvious failure, and debugging it through
the camera is miserable.

### 5.3 Delegates

Benchmark on real hardware before committing. Three configurations:

| Config | Expected | Notes |
|---|---|---|
| int8 + XNNPACK (CPU) | baseline | 4 threads. Default-on. Handles int8 natively. |
| fp16 + GPU delegate | maybe faster | GPU delegate prefers fp16; an int8 model may partially fall back to CPU with per-op conversion overhead, ending up *slower* than pure CPU. |
| int8 + GPU delegate | probably worst | Included only to confirm the above. |

NNAPI is deprecated as of Android 15 and vendor implementations were never reliable. Skip.

Ship the winner as default, keep a developer-menu toggle. Fall back to XNNPACK if delegate
creation throws — some devices fail GPU delegate init at runtime with no warning.

### 5.4 Threading

```
CameraX ImageAnalysis (STRATEGY_KEEP_ONLY_LATEST)
  → single-thread Executor (inference)
    → StateFlow<FrameResult>
      → Compose overlay recomposition on main
```

`KEEP_ONLY_LATEST` is essential: it drops frames rather than queueing them. Without it,
inference falling behind the camera builds an unbounded backlog and the overlay drifts
seconds behind reality.

One inference thread. Not a pool — two concurrent interpreter calls on one `Interpreter`
instance is undefined behaviour, and two interpreters doubles the memory.

**Always `close()` the ImageProxy in a `finally`.** A leaked ImageProxy stalls the entire
camera pipeline after ~3 frames with no exception thrown. Classic, silent, and the symptom
(preview freezes) points nowhere near the cause.

---

## 6. Packaging

### 6.1 Assets

```
app/src/main/assets/
  yoloworld_s_int8.tflite     ~28MB
  labels.txt                  ~2KB
  model_meta.json
```

```kotlin
android {
    androidResources {
        noCompress += listOf("tflite")
    }
    defaultConfig {
        ndk { abiFilters += listOf("arm64-v8a") }
    }
}
```

`noCompress` is not optional. Without it, AAPT deflates the model and the interpreter
cannot memory-map it — it must inflate ~28MB onto the heap at load.

### 6.2 Loading

```kotlin
private fun loadModel(context: Context, name: String): MappedByteBuffer =
    context.assets.openFd(name).use { fd ->
        FileInputStream(fd.fileDescriptor).use { stream ->
            stream.channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
        }
    }
```

Weights then live in mmapped, file-backed pages. The OS can evict them under memory
pressure and page them back, instead of the process being OOM-killed. This is the single
largest RAM lever available.

### 6.3 Distribution

Bundle the model. At ~28MB the total APK lands around 40MB, well under Play's 150MB AAB
base limit. On-demand model download buys nothing here except a network dependency and a
worse first-run experience.

`arm64-v8a` only. Nothing worth running this on is 32-bit, and the ABI filter roughly halves
the packaged native libs.

### 6.4 Budgets

| Resource | Target | Hard ceiling |
|---|---|---|
| APK size | 45MB | 100MB |
| Peak RSS (live mode) | 220MB | 350MB |
| Java heap | 60MB | 100MB |
| Inference latency | 30ms | 50ms |
| End-to-end frame latency | 60ms | 120ms |

Ceilings are where a bug report gets filed, not where the app breaks.

---

## 7. Rendering

Compose `Canvas` in a `Box` over the `PreviewView`. At ≤80 boxes this is comfortably fast
enough; if profiling in P2 disagrees, drop to a dedicated `SurfaceView` with its own render
thread. Do not start there — the SurfaceView path costs you Compose interop and is only
worth it if measurements demand it.

### 7.1 Coordinate transform

The one that always breaks. Detections are normalized to *source image* space. The preview
is typically letterboxed or centre-cropped into the view. Compute the transform once per
size change, not per box:

```kotlin
class ViewportTransform(
    sourceW: Int, sourceH: Int,
    viewW: Int, viewH: Int,
    scaleType: ScaleType,   // FILL_CENTER (crop) vs FIT_CENTER (letterbox)
)
```

CameraX `PreviewView` defaults to `FILL_CENTER`, meaning the preview is **cropped** — parts
of the analysed frame are off-screen. Boxes on cropped-out regions must be clipped, not
squashed. Test this explicitly in landscape, where the mismatch is most visible.

### 7.2 Visual spec

- Box stroke: 1.5dp, hairline. High density means heavy strokes turn the screen into mud.
- Label: 9sp monospace, below-left of box, with a solid backing rect at ~70% alpha.
- Colour by class-index hash → fixed palette of ~12 hues. Stable per class across frames.
- Score rendered to 2dp. It's part of the aesthetic.
- Label collision: none. Overlap is fine, arguably desirable. Do not implement label
  de-confliction.

### 7.3 P4 — connector lines

From each box's nearest vertex to a fixed screen anchor (default: bottom-left). Drawn in a
single `Path` per frame, not per-box draw calls. Alpha scaled by score. Gate behind a
settings toggle; it will look terrible with 80 boxes and great with 20.

---

## 8. Capture mode

1. `Recorder` writes MP4 to app cache. Cap at 15s.
2. Extract frames via `MediaMetadataRetriever.getFramesAtIndex()` in batches — do **not**
   pull all 300 frames into memory at once. Batch of 15, recycle bitmaps.
3. Run detector per frame on a background dispatcher, emit progress.
4. Store `Map<frameIndex, FrameResult>` — for 300 frames × 80 boxes that's ~2MB. Fine in
   memory; serialize to JSON in cache for the session.
5. Playback: `ExoPlayer` with the Compose overlay driven by
   `player.currentPosition → frameIndex`. Nearest-frame lookup, no interpolation.

Capture mode may use a **larger model or larger input size** than live mode, since there's
no latency budget. Worth trying 960px input here; the detection density difference is
substantial and it costs nothing but processing time.

**P3 export** (burning overlay into a video file) is genuinely more work than it sounds —
it needs `MediaCodec` with an input `Surface`, rendering the overlay via GL rather than
Compose, plus muxing. Treat it as its own mini-project. Do not scope it into P1.

---

## 9. Testing

Unit (JVM):
- **DFL decode against golden fixtures.** Highest-value test in the project. Dump raw output
  tensors from Python for 3 frames, assert Kotlin decode matches Python decode within 1e-3.
- NMS: known-input IoU cases, including the class-agnostic vs class-wise divergence.
- Letterbox coordinate round-trip: source px → letterboxed → normalized → back. Property test
  across random aspect ratios.
- `ViewportTransform` for FILL_CENTER and FIT_CENTER, portrait and landscape.
- Label/embedding count assertion.

Instrumented:
- Model loads, produces non-empty output on a fixture bitmap, on a real device.
- Delegate fallback: force GPU delegate failure, assert XNNPACK takes over.
- ImageProxy lifecycle: run 200 frames, assert no stall.

Perf (macrobenchmark):
- Inference latency p50/p95 per delegate config, per device tier.
- Peak RSS during a 60s live session.
- Frame drop rate.

Manual / subjective — no automation, but do it every build:
- Point at a blank wall. Does anything fire? (Empty screen = failure.)
- Point at a cluttered desk. Is it too dense to read?
- Two minutes continuous live mode. Does the device throttle noticeably?

---

## 10. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| int8 quantization collapses YOLO-World recall | Medium | High | Validation gate in §4.3. Fall back to fp16 (+~25MB, still shippable). |
| Manual DFL decode subtly wrong | High | High | Golden-fixture tests before any camera integration. |
| GPU delegate slower than CPU for int8 | Medium | Medium | Benchmark all three configs. XNNPACK is the safe default. |
| Thermal throttling in live mode | High | Medium | Accept degradation. Optionally drop to 15fps inference above a temp threshold. |
| onnx2tf export chain breaks on version drift | High | Medium | Pin every version in `tools/model/requirements.txt`. Dockerize if it recurs. |
| Vocabulary produces empty screens outdoors | Medium | High | The abstract classes (shadow, reflection, crack, fog) exist for this. Test outdoors early. |
| Compose Canvas too slow at 80 boxes | Low | Medium | SurfaceView fallback, but only if measured. |

**The one to worry about** is int8 recall collapse. Open-vocabulary models depend on
cosine similarity between image features and text embeddings; int8 quantization compresses
the feature space and similarity margins are exactly the thing that degrades. If it fails,
it will fail as "everything is 0.31 confidence and the labels are uniformly random" — which
is superficially hard to distinguish from the app working as intended. Test against fp32
output, not against vibes.

---

## 11. Open questions

- Class-agnostic vs class-wise NMS. Genuinely unresolved; decide visually in P1.
- Whether plurals ("leaf" and "leaves") earn their place or just double the boxes.
- Input resolution for live mode: 640 vs 480. 480 is ~2× faster; unclear whether detection
  density drops enough to matter at phone-screen scale.
- Whether capture mode should use a different, larger vocabulary than live mode.
- Score threshold as a user-facing slider vs a fixed 0.3. A slider is more fun and less
  designed.

---

## 12. Milestones

| # | Deliverable | Done when |
|---|---|---|
| M1 | `FakeDetector` + overlay | Boxes render correctly over live preview, both orientations, both scale types |
| M2 | P0 EfficientDet capture mode | Record → process → play back with overlay |
| M3 | Export chain | `.tflite` + `labels.txt` produced by one script, validation gate passes |
| M4 | `YoloWorldDetector` decode | Golden-fixture tests green |
| M5 | P1 complete | Capture mode running 270-class YOLO-World |
| M6 | Delegate benchmark | Numbers for all three configs on ≥2 devices |
| M7 | P2 live mode | ≥20fps sustained on target midrange device |
| M8 | Polish + ship | Settings, permissions, APK under budget |

M1 before M2 matters. Building the overlay against a fake detector means the rendering and
coordinate work is finished and tested before any model-related uncertainty enters the
picture — and rendering is where more of your time will actually go.
