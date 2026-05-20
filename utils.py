import os
import matplotlib
matplotlib.use('Agg') # Force Matplotlib to use the headless 'Agg' backend 
import matplotlib.pyplot as plt
import numpy as np
import math
import torch
import torch.nn as nn  
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2

# --- Constants & Mappings ---
# Class names mapping to indices based on the slides
CLASS_NAMES = {
    0: "Background",
    1: "Chair Pad (Green)",    # 椅墊
    2: "Armrest (Brown)",      # 扶手
    3: "Legs (Black)",         # 椅腳
    4: "Chair Base (Yellow)",  # 椅子底
    5: "Backrest (Gray)"       # 椅背
}

COLOR_MAP = {
    (255, 255, 255): 0, 
    (60, 180, 90): 1,
    (110, 40, 40): 2,   
    (50, 10, 70): 3, 
    (180, 200, 60): 4, 
    (100, 100, 100): 5  
}

# Dynamically generate the reverse map to avoid hardcoding twice
REVERSE_COLOR_MAP = {v: k for k, v in COLOR_MAP.items()}

def rgb_to_index(mask_rgb):
    """Converts the colored ground truth RGB image into a 0-5 index tensor."""
    mask_idx = np.zeros(mask_rgb.shape[:2], dtype=np.int64)
    for rgb, idx in COLOR_MAP.items():
        match = (mask_rgb == rgb).all(axis=-1)
        mask_idx[match] = idx
    return mask_idx

def index_to_rgb(mask_idx):
    """Converts 2D index mask back to 3D RGB image."""
    mask_rgb = np.zeros((mask_idx.shape[0], mask_idx.shape[1], 3), dtype=np.uint8)
    for idx, rgb in REVERSE_COLOR_MAP.items():
        mask_rgb[mask_idx == idx] = rgb
    return mask_rgb

# --- Data Transforms ---
def get_unet_train_transforms():
    return A.Compose([
        A.Resize(388, 388),
        A.HorizontalFlip(p=0.5),
        A.Affine(scale=(0.8, 1.2), translate_percent=0.1, rotate=(-10, 10), p=0.5),
        A.Perspective(scale=(0.02, 0.08), p=0.3),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
        A.Normalize(mean=(0.481, 0.449, 0.395), std=(0.228, 0.225, 0.226)),
        ToTensorV2(),
    ])

def get_unet_valid_transforms():
    return A.Compose([
        A.Resize(388, 388),
        A.Normalize(mean=(0.481, 0.449, 0.395), std=(0.228, 0.225, 0.226)),
        ToTensorV2(),
    ])

# --- Metrics ---
def calculate_multiclass_iou(preds, targets, num_classes=6):
    preds = torch.argmax(preds, dim=1)
    ious = []
    
    for cls in range(num_classes):
        pred_inds = preds == cls
        target_inds = targets == cls
        
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        
        if union > 0:
            ious.append((intersection / union).item())
        else:
            ious.append(float('nan'))
            
    valid_ious = [iou for iou in ious if not np.isnan(iou)]
    miou = sum(valid_ious) / len(valid_ious) if valid_ious else 0.0
    return ious, miou

def calculate_psnr_rgb(pred_rgb, gt_rgb):
    """
    Calculates the Peak Signal-to-Noise Ratio (PSNR) between the predicted 
    and ground truth RGB masks, assuming a maximum pixel intensity of 255.
    """
    mse = np.mean((pred_rgb.astype(np.float32) - gt_rgb.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

# --- Loss and Visualization ---
class CombinedCEDiceLoss(nn.Module):
    def __init__(self, ce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        probs = torch.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=6).permute(0, 3, 1, 2).float()
        
        intersection = (probs * targets_one_hot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        
        dice_score = (2. * intersection + 1e-5) / (union + 1e-5)
        dice_loss = 1.0 - dice_score.mean()
        
        return (self.ce_weight * ce_loss) + (self.dice_weight * dice_loss)
    
def save_validation_results(imgs, masks, preds, epoch, image_names, save_dir):
    """Saves 3 side-by-side comparison images for the homework requirement."""
    os.makedirs(save_dir, exist_ok=True)
    for i in range(min(3, imgs.shape[0])):
        img = imgs[i].cpu().numpy().transpose(1, 2, 0)
        mean, std = np.array([0.481, 0.449, 0.395]), np.array([0.228, 0.225, 0.226])
        img = np.clip(std * img + mean, 0, 1)

        gt_mask = masks[i].cpu().numpy()
        pred_mask = torch.argmax(preds[i], dim=0).cpu().numpy()

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img); axes[0].set_title(f"Input: {image_names[i]}"); axes[0].axis("off")
        axes[1].imshow(index_to_rgb(gt_mask)); axes[1].set_title("Ground Truth"); axes[1].axis("off")
        axes[2].imshow(index_to_rgb(pred_mask)); axes[2].set_title("Prediction"); axes[2].axis("off")

        plt.tight_layout(pad=2.0)
        save_path = os.path.join(save_dir, f"epoch_{epoch}_{image_names[i]}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight') 
        plt.close()