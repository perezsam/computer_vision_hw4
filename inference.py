import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import cv2

# --- CUSTOM ARCHITECTURE IMPORTS ---
from models.unet import UNet
from models.resnet34_unet import ResNet34UNet
from models.pretrained_resnet34_unet import PretrainedResNet34UNet
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

from utils import (
    get_unet_valid_transforms, 
    calculate_multiclass_iou, 
    calculate_psnr_rgb,
    rgb_to_index,
    index_to_rgb,
    CLASS_NAMES
)

def get_args():
    parser = argparse.ArgumentParser(description="Run Inference on Test Set")
    parser.add_argument('--model', type=str, required=True, 
                        choices=[
                            'unet_scratch', 
                            'resnet34_unet_scratch', 
                            'resnet34_unet_finetune', 
                            'deeplabv3_mobilenet'
                        ],
                        help="Model architecture to use.")
    parser.add_argument('--checkpoint', type=str, required=True, 
                        help="Path to the saved .pth weights file.")
    return parser.parse_args()

def run_inference():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Initialize custom architectures
    if args.model == 'unet_scratch':
        model = UNet(in_channels=3, out_channels=6)
    
    elif args.model in ['resnet34_unet_scratch', 'resnet34_unet_finetune']:
        model = ResNet34UNet(in_channels=3, out_channels=6) if 'scratch' in args.model else PretrainedResNet34UNet(out_classes=6)
            
    elif args.model == 'deeplabv3_mobilenet':
        model = deeplabv3_mobilenet_v3_large(weights=None, aux_loss=True)
        model.classifier[4] = torch.nn.Conv2d(256, 6, kernel_size=(1, 1))
        if model.aux_classifier is not None:
            model.aux_classifier[4] = torch.nn.Conv2d(10, 6, kernel_size=(1, 1))

    # 2. Load Weights
    print(f"Loading {args.model} weights from: {args.checkpoint}")
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)
    model.eval()

    test_dir = "data/test"
    test_input_dir = os.path.join(test_dir, "input")
    test_gt_dir = os.path.join(test_dir, "GT")
    
    predict_dir = os.path.join(test_dir, "predict", args.model)
    os.makedirs(predict_dir, exist_ok=True)
    
    test_files = sorted([f for f in os.listdir(test_input_dir) if f.endswith('.png')])
    transform = get_unet_valid_transforms()
    
    total_miou = 0
    total_psnr = 0

    print(f"\n--- Starting Inference on {len(test_files)} Test Images ---")
    
    with torch.no_grad():
        for file_name in test_files:
            img_path = os.path.join(test_input_dir, file_name)
            gt_name = file_name.replace('.png', '_pix.png')
            gt_path = os.path.join(test_gt_dir, gt_name)
            
            # --- Load Original Image ---
            orig_img = cv2.imread(img_path)
            orig_h, orig_w, _ = orig_img.shape
            img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
            
            # --- Load Ground Truth as RGB and convert to Indices ---
            gt_bgr = cv2.imread(gt_path)
            gt_rgb = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB)
            gt_idx = rgb_to_index(gt_rgb)
            gt_tensor = torch.from_numpy(gt_idx).unsqueeze(0).to(device)
            
            # Apply identical validation transform
            augmented = transform(image=img_rgb)
            img_tensor = augmented['image'].unsqueeze(0).to(device)

            # Forward pass
            with torch.amp.autocast('cuda'):
                if args.model == 'deeplabv3_mobilenet':
                    out_dict = model(img_tensor)
                    logits = out_dict['out']
                elif args.model == 'unet_scratch':
                    padded_img = F.pad(img_tensor, (92, 92, 92, 92), mode='reflect')
                    logits = model(padded_img)
                else:
                    logits = model(img_tensor)
            
            # Upsample logits to original image size BEFORE calculating metrics
            logits_up = F.interpolate(logits, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
            
            # Convert to class indices
            preds = torch.argmax(logits_up, dim=1) 
            final_mask_idx = preds.squeeze(0).cpu().numpy().astype(np.uint8)
            
            # --- Convert index prediction back to Colored RGB ---
            final_mask_rgb = index_to_rgb(final_mask_idx)

            # Calculate Slide Metrics
            class_ious, miou = calculate_multiclass_iou(logits_up, gt_tensor)
            psnr_val = calculate_psnr_rgb(final_mask_rgb, gt_rgb)
            
            total_miou += miou
            total_psnr += psnr_val if psnr_val != float('inf') else 100.0
            
            # Print Slide-Style Report for this image
            print(f"\nImage: {file_name}")
            print(f"PSNR: {psnr_val:.2f}")
            iou_strings = []
            for idx in range(1, 6): # Skip background (0)
                if not np.isnan(class_ious[idx]):
                    iou_strings.append(f"{CLASS_NAMES[idx]}: {class_ious[idx]*100:.2f}%")
            print("IoU: " + ", ".join(iou_strings))
            
            # Save final prediction as colored BGR
            save_path = os.path.join(predict_dir, file_name)
            cv2.imwrite(save_path, cv2.cvtColor(final_mask_rgb, cv2.COLOR_RGB2BGR))

    # Final Overall Report
    avg_miou = total_miou / len(test_files)
    avg_psnr = total_psnr / len(test_files)
    
    print("\n" + "="*50)
    print(f"FINAL REPORT: {args.model.upper()}")
    print("="*50)
    print(f"Overall mIoU: {avg_miou*100:.2f}%")
    print(f"Average PSNR: {avg_psnr:.2f}")
    print(f"Predictions saved to: {predict_dir}")
    print("="*50)

if __name__ == "__main__":
    run_inference()