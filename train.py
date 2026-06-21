import os
import argparse
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ChromosomeDataset
from model import UNet

def bipartite_loss(pred_logits, target_masks):
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    
    # Channel 2 is always Overlap C
    loss_c = loss_fn(pred_logits[:, 2:3], target_masks[:, 2:3]).mean(dim=(1,2,3))
    
    # Option 1: Pred0 -> A, Pred1 -> B
    loss_opt1 = loss_fn(pred_logits[:, 0:1], target_masks[:, 0:1]).mean(dim=(1,2,3)) + \
                loss_fn(pred_logits[:, 1:2], target_masks[:, 1:2]).mean(dim=(1,2,3))
                
    # Option 2: Pred0 -> B, Pred1 -> A
    loss_opt2 = loss_fn(pred_logits[:, 0:1], target_masks[:, 1:2]).mean(dim=(1,2,3)) + \
                loss_fn(pred_logits[:, 1:2], target_masks[:, 0:1]).mean(dim=(1,2,3))
                
    loss_ab = torch.min(loss_opt1, loss_opt2)
    return (loss_ab + loss_c).mean()

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output-dir", type=str, default="results_all_in_one")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds = ChromosomeDataset(Path(args.dataset_dir) / "train", augment=True)
    val_ds = ChromosomeDataset(Path(args.dataset_dir) / "val", augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = UNet(in_channels=1, out_channels=3, base_filters=32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.output_dir, exist_ok=True)
    
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]", mininterval=2.0, ncols=100)
        for imgs, masks in pbar:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = bipartite_loss(preds, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]", leave=False, mininterval=2.0, ncols=100):
                imgs, masks = imgs.to(device), masks.to(device)
                preds = model(imgs)
                loss = bipartite_loss(preds, masks)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pth"))
            print("  --> Saved new best model")

if __name__ == "__main__":
    train()

