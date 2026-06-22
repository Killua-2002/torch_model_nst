import os
import argparse
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import csv

from dataset import ChromosomeDataset
from model import UNet

def compute_dice(pred, target):
    smooth = 1e-6
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def save_visualization(out_path, img_gray, gt_a, gt_b, gt_c, pr_a, pr_b, pr_c):
    if img_gray.max() <= 1.0:
        img_gray = (img_gray * 255).astype(np.uint8)
        
    gt_rgb = np.zeros((*gt_a.shape, 3), dtype=np.uint8)
    gt_rgb[gt_a == 1, 0] = 255
    gt_rgb[gt_b == 1, 1] = 255
    gt_rgb[gt_c == 1, 0] = 255
    gt_rgb[gt_c == 1, 1] = 255
    
    pr_rgb = np.zeros((*pr_a.shape, 3), dtype=np.uint8)
    pr_rgb[pr_a == 1, 0] = 255
    pr_rgb[pr_b == 1, 1] = 255
    pr_rgb[pr_c == 1, 0] = 255
    pr_rgb[pr_c == 1, 1] = 255
    
    img_rgb = np.stack([img_gray]*3, axis=-1)
    overlay = (img_rgb * 0.5 + pr_rgb * 0.5).astype(np.uint8)
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(img_gray, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(gt_rgb)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    axes[2].imshow(pr_rgb)
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    axes[3].imshow(overlay)
    axes[3].set_title('Overlay Heatmap')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--base-filters", type=int, default=32)
    parser.add_argument("--output-dir", type=str, default="results_all_in_one")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_ds = ChromosomeDataset(Path(args.dataset_dir) / args.split, augment=False)
    
    model = UNet(in_channels=1, out_channels=3, base_filters=args.base_filters).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    os.makedirs(Path(args.output_dir) / "visualizations", exist_ok=True)

    dice_a_list = []
    dice_b_list = []
    dice_c_list = []
    
    with torch.no_grad():
        for idx in tqdm(range(len(test_ds)), desc="Evaluating", mininterval=2.0, ncols=100):
            img_t, mask_t = test_ds[idx]
            img_t = img_t.unsqueeze(0).to(device)
            
            # mask_t: [3, H, W]
            target_a = mask_t[0].cpu().numpy()
            target_b = mask_t[1].cpu().numpy()
            target_c = mask_t[2].cpu().numpy()
            
            preds = model(img_t)
            preds_prob = torch.sigmoid(preds)[0].cpu().numpy()
            
            pred0 = (preds_prob[0] > 0.5).astype(np.float32)
            pred1 = (preds_prob[1] > 0.5).astype(np.float32)
            pred_c = (preds_prob[2] > 0.5).astype(np.float32)
            
            # Bipartite matching for A and B
            score1 = compute_dice(pred0, target_a) + compute_dice(pred1, target_b)
            score2 = compute_dice(pred1, target_a) + compute_dice(pred0, target_b)
            
            if score1 >= score2:
                pred_a = pred0
                pred_b = pred1
            else:
                pred_a = pred1
                pred_b = pred0
                
            dice_a = compute_dice(pred_a, target_a)
            dice_b = compute_dice(pred_b, target_b)
            dice_c = compute_dice(pred_c, target_c)
            
            dice_a_list.append(dice_a)
            dice_b_list.append(dice_b)
            dice_c_list.append(dice_c)
            
            # Save visualizations for the first 20 samples
            if idx < 20:
                img_gray = img_t[0, 0].cpu().numpy()
                vis_path = Path(args.output_dir) / "visualizations" / f"showcase_{idx:03d}.png"
                save_visualization(vis_path, img_gray, target_a, target_b, target_c, pred_a, pred_b, pred_c)

    avg_dice_a = np.mean(dice_a_list)
    avg_dice_b = np.mean(dice_b_list)
    avg_dice_c = np.mean(dice_c_list)
    avg_dice_overall = np.mean([avg_dice_a, avg_dice_b, avg_dice_c])
    
    print(f"[{args.split}] Results:")
    print(f"  Dice A: {avg_dice_a:.4f}")
    print(f"  Dice B: {avg_dice_b:.4f}")
    print(f"  Dice C: {avg_dice_c:.4f}")
    print(f"  Overall: {avg_dice_overall:.4f}")
    
    metrics = {
        "dice_a": float(avg_dice_a),
        "dice_b": float(avg_dice_b),
        "dice_c": float(avg_dice_c),
        "dice_overall": float(avg_dice_overall)
    }
    
    with open(Path(args.output_dir) / f"metrics_{args.split}.json", "w") as f:
        json.dump(metrics, f, indent=2)
        
    with open(Path(args.output_dir) / f"metrics_{args.split}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "score"])
        for k, v in metrics.items():
            writer.writerow([k, f"{v:.4f}"])

if __name__ == "__main__":
    evaluate()

