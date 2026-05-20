import os
import random
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import the lightweight MobileNetV3 backbone DeepLabV3+
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, DeepLabV3_MobileNet_V3_Large_Weights

from utils import (
    get_unet_train_transforms, 
    get_unet_valid_transforms, 
    calculate_multiclass_iou, 
    CombinedCEDiceLoss,
    save_validation_results
)
from dataset import SofaComponentDataset 

def train_deeplabv3_mobilenet():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Launching Lightweight SOTA Fine-Tuning: DeepLabV3+ (MobileNetV3-Large)")
    
    # --- ISOLATED DIRECTORY SETUP ---
    out_dir = "outputs/deeplabv3_mobilenet/"
    os.makedirs(out_dir, exist_ok=True)
    
    dataset_dir = "data/" 
    train_input_dir = os.path.join(dataset_dir, "train", "input")
    train_gt_dir = os.path.join(dataset_dir, "train", "GT")
    
    all_train_files = [f for f in os.listdir(train_input_dir) if f.endswith('.png')]
    all_image_names = sorted([f.replace('.png', '') for f in all_train_files])
    
    random.seed(42) 
    random.shuffle(all_image_names)
    
    split_idx = int(len(all_image_names) * 0.9)
    train_names, val_names = all_image_names[:split_idx], all_image_names[split_idx:]
    tracked_images = val_names[:3] 

    train_ds = SofaComponentDataset(
        image_dir=train_input_dir, mask_dir=train_gt_dir, image_names=train_names, transform=get_unet_train_transforms()
    )
    val_ds = SofaComponentDataset(
        image_dir=train_input_dir, mask_dir=train_gt_dir, image_names=val_names, transform=get_unet_valid_transforms()
    )
    
    # Batch size of 16 is completely safe for MobileNetV3-Large on your remaining VRAM
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    # --- INITIALIZE LIGHTWEIGHT SOTA MODEL ---
    model = deeplabv3_mobilenet_v3_large(weights=DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT)
    
    # 1. Freeze backbone entirely to preserve ImageNet features
    for param in model.backbone.parameters():
        param.requires_grad = False
        
    # 2. Adapt both heads to your 6 sofa classes
    model.classifier[4] = nn.Conv2d(256, 6, kernel_size=(1, 1), stride=(1, 1))
    if model.aux_classifier is not None:
        model.aux_classifier[4] = nn.Conv2d(10, 6, kernel_size=(1, 1), stride=(1, 1))
        
    model = model.to(device)
    
    # 3. Only pass parameters that actually require gradients
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-3, weight_decay=1e-2)
    
    epochs = 80
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = CombinedCEDiceLoss(ce_weight=0.5, dice_weight=0.5) 
    scaler = torch.amp.GradScaler('cuda')
    
    best_val_miou = 0.0 
    history_train = []
    history_val = []

    for epoch in range(epochs):
        model.train()
        t_miou_total = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for imgs, masks, _ in loop:
            imgs, masks = imgs.to(device), masks.to(device)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                out_dict = model(imgs) 
                out = out_dict['out'] 
                
                # Main segmentation loss
                main_loss = criterion(out, masks)
                
                # --- Calculate and add weighted auxiliary loss ---
                if 'aux' in out_dict:
                    aux_out = out_dict['aux']
                    aux_loss = criterion(aux_out, masks)
                    loss = main_loss + 0.4 * aux_loss
                else:
                    loss = main_loss
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            _, miou = calculate_multiclass_iou(out, masks, num_classes=6)
            t_miou_total += miou
            loop.set_postfix(loss=loss.item())

        scheduler.step()

        model.eval()
        v_miou_total = 0
        
        with torch.no_grad():
            for imgs, masks, names in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                
                with torch.amp.autocast('cuda'):
                    out_dict = model(imgs)
                    out = out_dict['out']
                    
                    _, miou = calculate_multiclass_iou(out, masks, num_classes=6)
                    v_miou_total += miou
                
                if (epoch + 1) in [5, 10, 20]:
                    for i, name in enumerate(names):
                        if name in tracked_images:
                            img_save_dir = os.path.join(out_dir, "val_results")
                            save_validation_results(imgs[i:i+1], masks[i:i+1], out[i:i+1], epoch + 1, [name], img_save_dir)

        avg_t_miou = t_miou_total / len(train_loader)
        avg_v_miou = v_miou_total / len(val_loader)
        
        print(f"Train mIoU: {avg_t_miou:.4f} | Val mIoU: {avg_v_miou:.4f}")
        history_train.append(avg_t_miou)
        history_val.append(avg_v_miou)

        if avg_v_miou > best_val_miou:
            best_val_miou = avg_v_miou
            torch.save(model.state_dict(), os.path.join(out_dir, "best_deeplabv3_mobilenet.pth"))
            print(f"New MobileNet Record: {best_val_miou:.4f}")

    print("Training complete. Saving CSV logs...")
    df = pd.DataFrame({
        'Epoch': range(1, epochs + 1),
        'Train_mIoU': history_train,
        'Val_mIoU': history_val
    })
    df.to_csv(os.path.join(out_dir, "training_log.csv"), index=False)
    print(f"Logs successfully saved to {out_dir}")

if __name__ == "__main__":
    train_deeplabv3_mobilenet()