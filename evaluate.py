import os
import argparse
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import json

from dataset import ChromosomeDataset
from model import UNet

def compute_dice(pred, target):
    smooth = 1e-6
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output-dir", type=str, default="results_all_in_one")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_ds = ChromosomeDataset(Path(args.dataset_dir) / args.split, augment=False)
    
    model = UNet(in_channels=1, out_channels=3, base_filters=32).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    os.makedirs(Path(args.output_dir) / "predictions", exist_ok=True)

    dice_c_list = []
    
    with torch.no_grad():
        for idx in tqdm(range(len(test_ds)), desc="Evaluating", mininterval=2.0, ncols=100):
            img_t, mask_t = test_ds[idx]
            img_t = img_t.unsqueeze(0).to(device)
            
            # mask_t: [3, H, W]
            target_c = mask_t[2].cpu().numpy()
            
            preds = model(img_t)
            preds_prob = torch.sigmoid(preds)[0].cpu().numpy()
            
            pred_c = (preds_prob[2] > 0.5).astype(np.float32)
            
            dice_c = compute_dice(pred_c, target_c)
            dice_c_list.append(dice_c)
            
            # Save predictions for visualization (first 20)
            if idx < 20:
                pred_img = (pred_c * 255).astype(np.uint8)
                Image.fromarray(pred_img).save(Path(args.output_dir) / "predictions" / f"pred_c_{idx:03d}.png")

    avg_dice_c = np.mean(dice_c_list)
    print(f"[{args.split}] Average Dice for Overlap C: {avg_dice_c:.4f}")
    
    with open(Path(args.output_dir) / f"metrics_{args.split}.json", "w") as f:
        json.dump({"dice_c": float(avg_dice_c)}, f, indent=2)

if __name__ == "__main__":
    evaluate()

