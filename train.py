import os
import argparse
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import csv

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
    parser.add_argument("--base-filters", type=int, default=32)
    parser.add_argument("--output-dir", type=str, default="results_all_in_one")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint if available")
    parser.add_argument("--patience", type=int, default=20, help="Number of epochs with no improvement after which training will be stopped")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds = ChromosomeDataset(Path(args.dataset_dir) / "train", augment=True)
    val_ds = ChromosomeDataset(Path(args.dataset_dir) / "val", augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = UNet(in_channels=1, out_channels=3, base_filters=args.base_filters).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.output_dir, exist_ok=True)
    
    best_val_loss = float("inf")
    start_epoch = 0
    patience_counter = 0
    history_train = []
    history_val = []

    checkpoint_path = os.path.join(args.output_dir, "latest_checkpoint.pth")
    if args.resume and os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        patience_counter = checkpoint.get("patience_counter", 0)
        history_train = checkpoint.get("history_train", [])
        history_val = checkpoint.get("history_val", [])
        print(f"Resumed at epoch {start_epoch} with best_val_loss {best_val_loss:.4f} (Patience: {patience_counter}/{args.patience})")

    for epoch in range(start_epoch, args.epochs):
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
        
        history_train.append(train_loss)
        history_val.append(val_loss)
        
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pth"))
            print("  --> Saved new best model")
        else:
            patience_counter += 1
            print(f"  --> Early stopping patience: {patience_counter}/{args.patience}")

        # Save latest checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "patience_counter": patience_counter,
            "history_train": history_train,
            "history_val": history_val
        }, checkpoint_path)
        
        if patience_counter >= args.patience:
            print(f"\\n[!] Early stopping triggered after {epoch+1} epochs because val_loss did not improve for {args.patience} consecutive epochs.")
            break

    # Save history to CSV
    csv_path = os.path.join(args.output_dir, "history.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        for i, (t, v) in enumerate(zip(history_train, history_val)):
            writer.writerow([i+1, t, v])
            
    # Plot history
    plt.figure(figsize=(10, 6))
    epochs_range = range(1, len(history_train) + 1)
    plt.plot(epochs_range, history_train, label="Train Loss")
    plt.plot(epochs_range, history_val, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, "loss_curve.png"))
    plt.close()

if __name__ == "__main__":
    train()

