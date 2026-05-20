import os
import random
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.pretrained_resnet34_unet import PretrainedResNet34UNet
# from utils import get_unet_train_transforms, get_unet_valid_transforms, calculate_multiclass_iou, CombinedCEDiceLoss
from dataset import SofaComponentDataset 

from utils import (
    get_unet_train_transforms, 
    get_unet_valid_transforms, 
    calculate_multiclass_iou, 
    CombinedCEDiceLoss,
    save_validation_results 
)

def train_pretrained_unet():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Launching Fine-Tuning: Pre-trained ResNet34-UNet")
    
    # --- ISOLATED DIRECTORY SETUP ---
    out_dir = "outputs/resnet34_unet_finetune/"
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
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize the new Pre-trained Model
    model = PretrainedResNet34UNet(out_classes=6).to(device)
    
    # Lower learning rate (1e-4) specifically for fine-tuning pre-trained weights
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    
    epochs = 80
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = CombinedCEDiceLoss(ce_weight=0.5, dice_weight=0.5) 
    scaler = torch.amp.GradScaler('cuda')
    
    best_val_miou = 0.0
    history_train, history_val = [], []

    for epoch in range(epochs):
        model.train()
        t_miou_total = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for imgs, masks, _ in loop:
            imgs, masks = imgs.to(device), masks.to(device)
            # F.pad removed: Native padding in this architecture handles 388x388 inputs cleanly
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                out = model(imgs) 
                loss = criterion(out, masks)
            
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
                    out = model(imgs)
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

        # --- NEW: Store the metrics ---
        history_train.append(avg_t_miou)
        history_val.append(avg_v_miou)

        if avg_v_miou > best_val_miou:
            best_val_miou = avg_v_miou
            torch.save(model.state_dict(), os.path.join(out_dir, "best_resnet34_unet_finetune.pth"))
            print(f"New Pre-trained Record: {best_val_miou:.4f}")
    
    print("Training complete. Saving CSV logs...")
    df = pd.DataFrame({'Epoch': range(1, epochs + 1), 'Train_mIoU': history_train, 'Val_mIoU': history_val})
    df.to_csv(os.path.join(out_dir, "training_log.csv"), index=False)
    print(f"Logs successfully saved to {out_dir}")

if __name__ == "__main__":
    train_pretrained_unet()