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
import cv2

from dataset import ChromosomeDataset
from model import UNet

def compute_dice(pred, target):
    smooth = 1e-6
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def apply_mask_with_border(img_rgb, mask, color_rgb, fill_opacity=0.2):
    mask_bool = mask == 1
    if not np.any(mask_bool):
        return img_rgb
        
    colored_mask = np.zeros_like(img_rgb)
    colored_mask[:] = color_rgb
    
    # 20% opacity fill
    img_rgb[mask_bool] = (img_rgb[mask_bool] * (1 - fill_opacity) + colored_mask[mask_bool] * fill_opacity).astype(np.uint8)
    
    # 100% opacity border
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_rgb, contours, -1, color_rgb, 1)
    return img_rgb

def save_visualization(out_path, img_gray, gt_a, gt_b, gt_c, pr_a, pr_b, pr_c, prob_a, prob_b, prob_c):
    if img_gray.max() <= 1.0:
        img_gray = (img_gray * 255).astype(np.uint8)
        
    img_rgb = np.stack([img_gray]*3, axis=-1)
    
    # Deduce Top/Bottom
    visible_mask = (img_gray > 0).astype(np.float32)
    vis_if_a_top = np.clip(pr_a + np.maximum(0, pr_b - pr_c), 0, 1)
    vis_if_b_top = np.clip(pr_b + np.maximum(0, pr_a - pr_c), 0, 1)
    
    if np.sum(np.abs(visible_mask - vis_if_a_top)) <= np.sum(np.abs(visible_mask - vis_if_b_top)):
        vis_a = pr_a
        vis_b = np.maximum(0, pr_b - pr_c)
        lbl_a = "Vis A (Top - Intact)"
        lbl_b = "Vis B (Bottom - Split)"
    else:
        vis_a = np.maximum(0, pr_a - pr_c)
        vis_b = pr_b
        lbl_a = "Vis A (Bottom - Split)"
        lbl_b = "Vis B (Top - Intact)"
        
    pr_overlay = img_rgb.copy()
    pr_overlay = apply_mask_with_border(pr_overlay, pr_a, (255, 0, 0), fill_opacity=0.1)
    pr_overlay = apply_mask_with_border(pr_overlay, pr_b, (0, 255, 0), fill_opacity=0.1)
    pr_overlay = apply_mask_with_border(pr_overlay, pr_c, (255, 255, 0), fill_opacity=0.1)
    
    max_prob = np.max([prob_a, prob_b, prob_c], axis=0)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Apply texture
    vis_a_disp = img_gray * vis_a
    vis_b_disp = img_gray * vis_b
    pr_c_disp = img_gray * pr_c
    pr_a_disp = img_gray * pr_a
    pr_b_disp = img_gray * pr_b
    
    # ROW 1: Before Recovery
    axes[0, 0].imshow(img_gray, cmap='gray')
    axes[0, 0].set_title('Original Input')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(vis_a_disp, cmap='gray')
    axes[0, 1].set_title(lbl_a)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(vis_b_disp, cmap='gray')
    axes[0, 2].set_title(lbl_b)
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(pr_c_disp, cmap='gray')
    axes[0, 3].set_title('Overlap C')
    axes[0, 3].axis('off')
    
    # ROW 2: After Recovery
    axes[1, 0].imshow(pr_a_disp, cmap='gray')
    axes[1, 0].set_title('Full Pred A')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(pr_b_disp, cmap='gray')
    axes[1, 1].set_title('Full Pred B')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(pr_overlay)
    axes[1, 2].set_title('Overlay (20% Opacity)')
    axes[1, 2].axis('off')
    
    im = axes[1, 3].imshow(max_prob, cmap='Blues', vmin=0, vmax=1)
    axes[1, 3].set_title('Prob Heatmap (Blues)')
    axes[1, 3].axis('off')
    
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
    os.makedirs(Path(args.output_dir) / "predictions", exist_ok=True)

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
                prob_a = preds_prob[0]
                prob_b = preds_prob[1]
            else:
                pred_a = pred1
                pred_b = pred0
                prob_a = preds_prob[1]
                prob_b = preds_prob[0]
                
            prob_c = preds_prob[2]
                
            dice_a = compute_dice(pred_a, target_a)
            dice_b = compute_dice(pred_b, target_b)
            dice_c = compute_dice(pred_c, target_c)
            
            dice_a_list.append(dice_a)
            dice_b_list.append(dice_b)
            dice_c_list.append(dice_c)
            
            # Save raw masks for ALL samples
            raw_pred_dir = Path(args.output_dir) / "predictions" / f"sample_{idx:05d}"
            os.makedirs(raw_pred_dir, exist_ok=True)
            Image.fromarray((pred_a * 255).astype(np.uint8)).save(raw_pred_dir / "pred_a.png")
            Image.fromarray((pred_b * 255).astype(np.uint8)).save(raw_pred_dir / "pred_b.png")
            Image.fromarray((pred_c * 255).astype(np.uint8)).save(raw_pred_dir / "pred_c.png")
            
            # Save visualizations for the first 30 samples
            if idx < 30:
                img_gray = img_t[0, 0].cpu().numpy()
                vis_path = Path(args.output_dir) / "visualizations" / f"showcase_{idx:05d}.png"
                save_visualization(vis_path, img_gray, target_a, target_b, target_c, pred_a, pred_b, pred_c, prob_a, prob_b, prob_c)

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

