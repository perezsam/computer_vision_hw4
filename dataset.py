import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class SofaComponentDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_names, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_names = image_names
        self.transform = transform
        
        # Color mapping based on assignment guidelines
        self.color_map = {
            (255, 255, 255): 0, # Background (flattened from transparent)
            (60, 180, 90): 1,   # 椅墊 (Seat cushion)
            (110, 40, 40): 2,   # 扶手 (Armrest)
            (50, 10, 70): 3,    # 椅腳 (Legs)
            (180, 200, 60): 4,  # 椅子底 (Chair base)
            (100, 100, 100): 5  # 椅背 (Backrest)
        }

    def _flatten_rgba(self, img, bg_color=(255, 255, 255)):
        if img.mode == 'RGBA':
            background = Image.new('RGB', img.size, bg_color)
            background.paste(img, mask=img.split()[3]) 
            return np.array(background)
        return np.array(img.convert('RGB'))

    def _rgb_to_mask(self, mask_img):
        mask = np.zeros((mask_img.shape[0], mask_img.shape[1]), dtype=np.int64)
        for rgb, idx in self.color_map.items():
            match = np.all(mask_img == rgb, axis=-1)
            mask[match] = idx
        return mask

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        base_name = self.image_names[idx]
        img_path = os.path.join(self.image_dir, f"{base_name}.png")
        mask_path = os.path.join(self.mask_dir, f"{base_name}_pix.png")

        img_pil = Image.open(img_path)
        mask_pil = Image.open(mask_path).convert('RGB')

        # --- Force mask to match image size exactly ---
        if img_pil.size != mask_pil.size:
            mask_pil = mask_pil.resize(img_pil.size, resample=Image.NEAREST)

        image = self._flatten_rgba(img_pil)
        mask_rgb = np.array(mask_pil)
        mask_idx = self._rgb_to_mask(mask_rgb)

        if self.transform:
            augmented = self.transform(image=image, mask=mask_idx)
            image = augmented['image']
            mask_idx = augmented['mask']
            
        return image, mask_idx.long(), base_name