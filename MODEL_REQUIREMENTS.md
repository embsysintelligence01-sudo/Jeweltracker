# Jewel Tracker — AI Model Requirements

Model weights are stored locally and excluded from GitHub.

| Model file | Purpose / identification |
|---|---|
| weights/siglip2-base-patch32-256_vision_encoder.sim.onnx | SigLIP2 Base Patch32 256 vision encoder |
| weights/resnet18-f37072fd.pth | ResNet18 pretrained weights |
| weights/resnet18_features.onnx | ResNet18 feature extractor |
| newstone1.onnx | Custom ONNX model |
| rubbing_gold.onnx | Custom ONNX model |
| acid.onnx | Custom ONNX model |
| rojeweltrack.onnx | Custom ONNX model |
| nivagar_gold_v3.onnx | Custom ONNX model |
| nivagar_gold_v3.pt | Custom PyTorch model |
| acid_test/model_training/artifacts/temporal_mobilenet_v3_20260912_170149.pt | Temporal MobileNetV3 |
| acid_test/model_training/artifacts/temporal_mobilenet_v3_20260912_170149.onnx | Temporal MobileNetV3 ONNX |

## Setup

Obtain the compatible model weights separately and place them
at their documented relative paths before running the project.

Check that model versions and preprocessing match the application.
