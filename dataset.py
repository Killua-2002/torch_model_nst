import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np

class ChromosomeDataset(Dataset):
    def __init__(self, split_dir: str | Path, augment: bool = False):
        self.split_dir = Path(split_dir)
        self.augment = augment
        
        self.img_dir = self.split_dir / "images"
        self.mask_a_dir = self.split_dir / "masks_A"
        self.mask_b_dir = self.split_dir / "masks_B"
        self.mask_c_dir = self.split_dir / "masks_C"
        
        # Collect all valid samples
        self.samples = []
        if self.img_dir.exists():
            for p in sorted(self.img_dir.glob("*.png")):
                name = p.name
                ma = self.mask_a_dir / name
                mb = self.mask_b_dir / name
                mc = self.mask_c_dir / name
                if ma.exists() and mb.exists() and mc.exists():
                    self.samples.append((p, ma, mb, mc))
                    
        if not self.samples:
            print(f"Warning: No valid samples found in {split_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_p, ma_p, mb_p, mc_p = self.samples[idx]
        
        # Read as numpy
        img = np.array(Image.open(img_p).convert("L"), dtype=np.float32) / 255.0
        ma = np.array(Image.open(ma_p).convert("L"), dtype=np.float32) / 255.0
        mb = np.array(Image.open(mb_p).convert("L"), dtype=np.float32) / 255.0
        mc = np.array(Image.open(mc_p).convert("L"), dtype=np.float32) / 255.0
        
        # Binarize masks
        ma = (ma > 0.5).astype(np.float32)
        mb = (mb > 0.5).astype(np.float32)
        mc = (mc > 0.5).astype(np.float32)
        
        if self.augment:
            # Random horizontal flip
            if np.random.rand() > 0.5:
                img = np.fliplr(img)
                ma = np.fliplr(ma)
                mb = np.fliplr(mb)
                mc = np.fliplr(mc)
            # Random vertical flip
            if np.random.rand() > 0.5:
                img = np.flipud(img)
                ma = np.flipud(ma)
                mb = np.flipud(mb)
                mc = np.flipud(mc)
                
        # Convert to tensors [C, H, W]
        img_t = torch.from_numpy(img.copy()).unsqueeze(0)
        mask_t = torch.from_numpy(np.stack([ma, mb, mc], axis=0).copy())
        
        return img_t, mask_t

