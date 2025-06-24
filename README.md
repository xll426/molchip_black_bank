# CornerYOLO Black Bank

This repository contains a lightweight corner detection network used in the so called "black bank" workflow.  The code has been reorganised into a small package located under `blackbank/` and several runnable scripts.

## Package Overview

```
blackbank/
├── __init__.py          # package exports
├── dataset.py           # `TVCornerDataset` and data transforms
├── losses.py            # focal loss and detection loss utilities
└── model.py             # `CornerYOLOOptimized` network definition
```

### Dataset
`TVCornerDataset` loads images and the corresponding JSON annotations describing four corner points.  During `__getitem__` the image is resized using letterbox, optional Albumentations transforms are applied and the corner targets are encoded into a 25‑dimensional vector (24 dims of corner/angle features plus an object confidence flag).  A segmentation mask of the quadrilateral is also generated.

Useful helpers for letterbox, corner feature computation and building training/validation transforms are provided in the same module.

### Model
`CornerYOLOOptimized` is a lightweight convolutional network producing a `40×40` feature map with 10 output channels:

1. background score
2. center heatmap
3-10. offset regressions for the four corners

The model returns both the fused feature map and the raw detection head output.  A helper wrapper `DetectOnlyWrapper` exposes only the detection output when exporting to ONNX.

### Loss
`blackbank.losses` implements the detection loss used during training.  It combines a focal BCE loss for background/center heatmaps with Smooth‑L1 regression on corner offsets.  Target generation uses an adaptive gaussian centred on the object size.

## Scripts

- **train.py** – training entry point.  It constructs `TVCornerDataset` loaders, instantiates `CornerYOLOOptimized` and optimises it using Adam with a cosine annealing scheduler.  Checkpoints are written to `checkpoints/` and a loss curve can be saved.
- **inference.py** – run inference on a single image.  The script handles letterbox preprocessing, forwards the network and decodes the predicted quadrilateral back to image coordinates.
- **export_onnx.py** – export the trained model to ONNX (optionally simplified via `onnxsim`).

All scripts now import from the `blackbank` package which keeps the codebase tidy.

## Usage

Training example:
```bash
python train.py --img_dir /path/to/images \
                --label_dir /path/to/labels \
                --epochs 300
```

Inference example:
```bash
python inference.py --model checkpoints/best_checkpoint.pth \
                    --image example.jpg
```

Export ONNX:
```bash
python export_onnx.py --model checkpoints/best_checkpoint.pth \
                      --output black_bank.onnx
```

## Notes

Example images and checkpoints are kept in this repository for demonstration.  They can be removed if a clean workspace is required.
