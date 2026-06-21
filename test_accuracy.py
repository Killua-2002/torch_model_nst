import os
import argparse
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
import json

from dataset import ChromosomeDataset
from model import UNet

def compute_iou(pred, target):
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    if union == 0:
        return 1.0 if pred.sum() == 0 else 0.0
    return inter / union

def evaluate_accuracy():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--threshold", type=float, default=0.90, help="IoU threshold to mark an image as correct")
    parser.add_argument("--base-filters", type=int, default=32, help="Set to 64 if evaluating Teacher, 32 for Student")
    parser.add_argument("--save-vis", action="store_true", help="Save visualization images of predictions")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_ds = ChromosomeDataset(Path(args.dataset_dir) / args.split, augment=False)
    
    model = UNet(in_channels=1, out_channels=3, base_filters=args.base_filters).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    correct_count = 0
    total_count = len(test_ds)
    
    iou_scores = []

    with torch.no_grad():
        for idx in tqdm(range(total_count), desc="Checking Accuracy", mininterval=2.0, ncols=100):
            img_t, mask_t = test_ds[idx]
            img_t = img_t.unsqueeze(0).to(device)
            
            # mask_t: [3, H, W] -> A, B, C
            true_a = mask_t[0].cpu().numpy()
            true_b = mask_t[1].cpu().numpy()
            
            preds = model(img_t)
            preds_prob = torch.sigmoid(preds)[0].cpu().numpy()
            
            pred_a = (preds_prob[0] > 0.5).astype(np.float32)
            pred_b = (preds_prob[1] > 0.5).astype(np.float32)
            
            # Compare both permutations
            # Option 1: Pred A -> True A, Pred B -> True B
            iou_a_1 = compute_iou(pred_a, true_a)
            iou_b_1 = compute_iou(pred_b, true_b)
            score_1 = min(iou_a_1, iou_b_1)
            
            # Option 2: Pred A -> True B, Pred B -> True A
            iou_a_2 = compute_iou(pred_a, true_b)
            iou_b_2 = compute_iou(pred_b, true_a)
            score_2 = min(iou_a_2, iou_b_2)
            
            best_score = max(score_1, score_2)
            iou_scores.append(best_score)
            
            if best_score >= args.threshold:
                correct_count += 1
                
            if args.save_vis and idx < 50:
                try:
                    import matplotlib.pyplot as plt
                    vis_dir = os.path.join(args.dataset_dir, "..", f"vis_{args.split}")
                    os.makedirs(vis_dir, exist_ok=True)
                    
                    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
                    axes[0].imshow(img_t[0].cpu().numpy(), cmap="gray")
                    axes[0].set_title("Input Image")
                    axes[1].imshow(true_a, cmap="Reds")
                    axes[1].set_title("GT Chromosome A")
                    axes[2].imshow(true_b, cmap="Blues")
                    axes[2].set_title("GT Chromosome B")
                    
                    # Ensure Pred A aligns with True A visually for the chart
                    if score_1 >= score_2:
                        disp_a, disp_b = pred_a, pred_b
                    else:
                        disp_a, disp_b = pred_b, pred_a
                        
                    axes[3].imshow(disp_a, cmap="Reds")
                    axes[3].set_title(f"Pred A (IoU={max(iou_a_1, iou_b_2)*100:.1f}%)")
                    axes[4].imshow(disp_b, cmap="Blues")
                    axes[4].set_title(f"Pred B (IoU={max(iou_b_1, iou_a_2)*100:.1f}%)")
                    
                    for ax in axes:
                        ax.axis("off")
                        
                    status = "PASS" if best_score >= args.threshold else "FAIL"
                    plt.suptitle(f"Sample {idx} - {status} (Score: {best_score*100:.1f}%)")
                    plt.savefig(os.path.join(vis_dir, f"result_{idx:03d}.png"))
                    plt.close()
                except ImportError:
                    pass

    accuracy = correct_count / total_count
    avg_iou = np.mean(iou_scores)
    
    print("-" * 50)
    print(f"Split: {args.split}")
    print(f"Total images checked: {total_count}")
    print(f"Threshold (IoU): >= {args.threshold*100:.1f}%")
    print(f"Correctly identified: {correct_count}")
    print(f"STRICT ACCURACY: {accuracy * 100:.2f}%")
    print(f"Average IoU (for A & B): {avg_iou * 100:.2f}%")
    print("-" * 50)

if __name__ == "__main__":
    evaluate_accuracy()

