import os
import json
import hashlib
import torch
import numpy as np
import shutil
from ultralytics import YOLO
from vocabulary import CLASSES

def strip_post_processing(model):
    """
    Monkey-patches the YOLO-World head to return raw tensors (DFL distributions and Scores)
    instead of decoded bounding boxes. This is required for stable int8 quantization.
    """
    head = model.model.model[-1] # The WorldDetect head

    def raw_forward(x, embed=None):
        bboxes = []
        scores = []
        txt_feats = embed if embed is not None else head.txt_feats

        for i in range(head.nl):
            b = head.cv2[i](x[i])
            s = head.cv3[i](x[i])
            s = head.cv4[i](s, txt_feats)

            bboxes.append(b.view(b.shape[0], b.shape[1], -1))
            scores.append(s.view(s.shape[0], s.shape[1], -1))

        return torch.cat(bboxes, 2).transpose(1, 2), torch.cat(scores, 2).transpose(1, 2)

    head.forward = raw_forward
    print("Monkey-patched WorldDetect head for raw output.")

def generate_calibration_data():
    """Generates synthetic calibration data for the representative dataset."""
    print(f"Generating 100 calibration samples...")
    for _ in range(100):
        # YOLO-World expects 0-1 range for the input images
        img = np.random.uniform(0, 1, (1, 640, 640, 3)).astype(np.float32)
        yield [img]

def export_enhanced_int8(saved_model_path, output_path):
    """Converts SavedModel to calibrated Int8 TFLite."""
    import tensorflow as tf

    print(f"Loading SavedModel from {saved_model_path}...")
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = generate_calibration_data

    # Force full integer quantization
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    print("Running conversion (this may take a few minutes)...")
    tflite_model = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_model)
    print(f"Enhanced Int8 model saved to {output_path}")

def main():
    print(f"Loading base YOLO-World-S model...")
    model = YOLO("yolov8s-worldv2.pt")

    print(f"Reparameterizing with {len(CLASSES)} classes...")
    model.set_classes(CLASSES)

    custom_model_pt = "yoloworld_s_custom.pt"
    model.save(custom_model_pt)

    strip_post_processing(model)
    os.makedirs("out", exist_ok=True)

    with open("out/labels.txt", "w") as f:
        f.write("\n".join(CLASSES))

    print("Exporting to SavedModel via Ultralytics...")
    # This avoids the onnx2tf dependency issues for now
    # Ultralytics will export to a folder containing the SavedModel
    saved_model_path = model.export(format="saved_model", imgsz=640)
    print(f"SavedModel exported to {saved_model_path}")

    # Ultralytics usually creates a folder like 'yolov8s-worldv2_saved_model'
    # We need the actual path to the folder containing 'saved_model.pb'
    if not os.path.isdir(saved_model_path):
        # Handle cases where it might return the zip or different path
        print("Checking for SavedModel directory...")
        potential_dir = "yolov8s-worldv2_saved_model"
        if os.path.exists(potential_dir):
            saved_model_path = potential_dir

    tflite_output = "out/yoloworld_s_int8_calibrated.tflite"
    export_enhanced_int8(saved_model_path, tflite_output)

    meta = {
        "num_classes": len(CLASSES),
        "input_size": 640,
        "type": "int8_calibrated",
        "notes": "Exported with 100 random calibration samples for enhanced precision."
    }
    with open("out/model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

if __name__ == "__main__":
    main()
