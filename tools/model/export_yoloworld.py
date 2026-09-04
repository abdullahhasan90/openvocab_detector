import os
import json
import hashlib
import torch
from ultralytics import YOLO
from vocabulary import CLASSES

def strip_post_processing(model):
    """
    Monkey-patches the YOLO-World head to return raw tensors (DFL distributions and Scores)
    instead of decoded bounding boxes. This is required for stable int8 quantization.
    """
    head = model.model.model[-1] # The WorldDetect head

    def raw_forward(x, embed=None):
        # x is a list of features from 3 scales: [1, 256, 80, 80], [1, 512, 40, 40], [1, 1024, 20, 20]
        # (Channels vary by model scale, e.g., 's')

        bboxes = []
        scores = []

        # If embed is not provided, use the pre-computed text features in the head
        txt_feats = embed if embed is not None else head.txt_feats

        for i in range(head.nl):
            # bboxes branch: [batch, 64, h, w]
            b = head.cv2[i](x[i])
            # scores branch: [batch, num_classes, h, w]
            s = head.cv3[i](x[i])

            # Contrastive head processing for scores
            # head.cv4[i] is BNContrastiveHead
            s = head.cv4[i](s, txt_feats)

            # Flatten to [batch, channels, h*w]
            bboxes.append(b.view(b.shape[0], b.shape[1], -1))
            scores.append(s.view(s.shape[0], s.shape[1], -1))

        # Concatenate across all scales
        # Result shapes: [batch, 64, 8400] and [batch, 270, 8400]
        return torch.cat(bboxes, 2).transpose(1, 2), torch.cat(scores, 2).transpose(1, 2)

    head.forward = raw_forward
    print("Monkey-patched WorldDetect head for raw output.")

def main():
    print(f"Loading base YOLO-World-S model...")
    model = YOLO("yolov8s-worldv2.pt")

    print(f"Reparameterizing with {len(CLASSES)} classes...")
    model.set_classes(CLASSES)

    # Save the reparameterized PyTorch model
    custom_model_pt = "yoloworld_s_custom.pt"
    model.save(custom_model_pt)
    print(f"Saved custom model to {custom_model_pt}")

    # Strip post-processing for the export version
    strip_post_processing(model)

    # Create output directory
    os.makedirs("out", exist_ok=True)

    # 1. Save Labels
    with open("out/labels.txt", "w") as f:
        f.write("\n".join(CLASSES))
    print("Saved out/labels.txt")

    # 2. Export to ONNX
    print("Exporting to ONNX...")
    # imgsz=640, opset=12 to support einsum
    onnx_path = model.export(format="onnx", imgsz=640, opset=12)

    # Rename to our convention if needed (Ultralytics might use default name)
    if os.path.exists("yolov8s-worldv2.onnx"):
        if os.path.exists("yoloworld_s_custom.onnx"):
            os.remove("yoloworld_s_custom.onnx")
        os.rename("yolov8s-worldv2.onnx", "yoloworld_s_custom.onnx")
        onnx_path = "yoloworld_s_custom.onnx"

    print(f"Exported to {onnx_path}")

    # 3. Export to TFLite (Float16 for GPU)
    print("Exporting to TFLite (Float16)...")
    # Ultralytics can export directly to TFLite with FP16
    tflite_path = model.export(format="tflite", imgsz=640, half=True)
    print(f"Exported to {tflite_path}")

    # 4. Model Meta JSON
    with open(custom_model_pt, "rb") as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()

    meta = {
        "sha256": sha256,
        "num_classes": len(CLASSES),
        "input_size": 640,
        "generated_at": "2024-09-04",
        "notes": "Exported WITH raw DFL and Scores. Manual decode required in Kotlin."
    }

    with open("out/model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("Saved out/model_meta.json")

    print("\n--- NEXT STEPS ---")
    print(f"1. Simplify ONNX: onnxsim {onnx_path} yoloworld_s_sim.onnx")
    print("2. Convert to int8 TFLite: onnx2tf -i yoloworld_s_sim.onnx -o out/yoloworld_s_int8.tflite -oiqt")

if __name__ == "__main__":
    main()
