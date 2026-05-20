
# HW4: Semantic Component Segmentation of Sofa

**Course:** Computer Vision  
National Yang Ming Chiao Tung University  

## Overview
This repository contains the implementation of four semantic segmentation architectures designed to extract six structural components from a specialized sofa dataset. The models include a baseline U-Net (from scratch), a custom attention-enhanced ResNet34-UNet (from scratch), a fine-tuned ResNet34-UNet, and a lightweight DeepLabV3+ utilizing a MobileNetV3 backbone.

> **Note on Model Checkpoints:** Due to size constraints, the `.pth` weight files are not included in this zip archive. 
> *[Optional: You can download the pre-trained weights from this Google Drive Link: ]*

---

## 1. Environment Requirements
Ensure your environment has the following primary dependencies installed:
* `torch` and `torchvision` (with CUDA support recommended)
* `albumentations` (for data augmentation)
* `opencv-python` and `Pillow` (for image processing)
* `matplotlib` and `pandas` (for plotting and metrics tracking)

---

## 2. Directory Structure
Please ensure the dataset is placed in the root directory before execution:
```text
computer_vision_hw4/
├── data/
│   ├── train/                  # Contains input/ and GT/ subdirectories
│   └── test/                   # Contains input/, GT/, and predict/ subdirectories
├── models/                     # Custom architecture definitions
├── outputs/                    # Training logs, validation images, and saved .pth weights
├── utils.py                    # Metrics (mIoU/PSNR), transforms, and loss functions
├── dataset.py                  # Custom PyTorch Dataset loader
├── inference.py                # Universal testing script
├── plot_history.py             # Script to generate training curves
├── models_inference_results.txt # Console output logs from the final test set inference
├── finetune_resnet34_unet.py   # Fine-tuning script for pre-trained ResNet34
└── train_*.py                  # Training scripts for the other architectures

```

---

## 3. Training the Models

To train the models from scratch or fine-tune the pre-trained variants, execute the following scripts. All models will automatically train for 80 epochs, utilizing `AdamW`, a `CosineAnnealingLR` scheduler, and Mixed Precision (AMP).

```bash
# 1. Baseline U-Net (From Scratch)
python train_unet_scratch.py

# 2. Custom ResNet34-UNet with CBAM (From Scratch)
python train_resnet34_unet_scratch.py

# 3. Pre-trained ResNet34-UNet (Fine-tuned)
python finetune_resnet34_unet.py

# 4. Lightweight DeepLabV3+ MobileNetV3 (Fine-tuned)
python train_deeplabv3_mobilenet.py

```

---

## 4. Running Inference

The `inference.py` script utilizes `argparse` to dynamically load the selected model and its corresponding checkpoint. It performs bilinear upsampling on the logits and maps the predictions back to the required RGB color space.

*(Note: Ensure you have placed the trained `.pth` files into their respective `outputs/` subdirectories before running these commands).*

```bash
# Evaluate U-Net
python inference.py --model unet_scratch --checkpoint outputs/unet_scratch/best_unet_scratch.pth

# Evaluate Custom ResNet34-UNet (Scratch)
python inference.py --model resnet34_unet_scratch --checkpoint outputs/resnet34_unet_scratch/best_resnet34_unet_scratch.pth

# Evaluate Pre-trained ResNet34-UNet
python inference.py --model resnet34_unet_finetune --checkpoint outputs/resnet34_unet_finetune/best_resnet34_unet_finetune.pth

# Evaluate DeepLabV3+
python inference.py --model deeplabv3_mobilenet --checkpoint outputs/deeplabv3_mobilenet/best_deeplabv3_mobilenet.pth

```

*The final colored RGB masks will be saved to `data/test/predict/<model_name>/`.*

---

## 5. Generating Training Curves

To visualize the training and validation mIoU progression, use the `plot_history.py` script. This reads the `.csv` logs generated during training and outputs a graph with the peak validation score highlighted.

```bash
python plot_history.py --csv outputs/unet_scratch/training_log.csv --out figures/ --name unet_scratch
python plot_history.py --csv outputs/resnet34_unet_scratch/training_log.csv --out figures/ --name resnet34_unet_scratch
python plot_history.py --csv outputs/resnet34_unet_finetune/training_log.csv --out figures/ --name resnet34_unet_finetune
python plot_history.py --csv outputs/deeplabv3_mobilenet/training_log.csv --out figures/ --name deeplabv3_mobilenet

```