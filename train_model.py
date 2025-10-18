import random
from pathlib import Path
from typing import List, Dict, Optional
import h5py
from tqdm import tqdm
from datetime import datetime
import pandas as pd
import argparse
import json
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms
import numpy as np

# Import from our config and utils files
import config
from utils import create_inference_visualization, set_seed, save_loss_plot
from utils import apply_style_image, calculate_metrics, save_metrics_plot

from unetplusplus import Model

# --- Dataset Class and MaskedBCELoss (Unchanged from your version) ---
class PreprocessedBinaryVegDataset(Dataset):
    def __init__(self, data_dir: Path, style_dir: Path, use_fda: bool = False, photometric_augs=None, geometric_augs=None):
        self.data_dir = data_dir
        self.style_dir = style_dir
        self.use_fda = use_fda
        self.photometric_augs = photometric_augs
        self.geometric_augs = geometric_augs
        self.patch_basenames: List[str] = []
        self.style_paths: List[Path] = []
        if not data_dir.exists():
            print(f"Warning: Data directory not found: {self.data_dir}", file=sys.stderr)
        else:
            self.patch_basenames = [p.name.replace("_source.h5", "") for p in data_dir.glob("*_source.h5")]
            if not self.patch_basenames:
                print(f"Warning: No patches found in {self.data_dir}", file=sys.stderr)
        if style_dir and style_dir.exists():
            self.style_paths = list(style_dir.glob("*.h5"))
        if self.use_fda and not self.style_paths:
            print(f"    - FDA is enabled, but no style images were found. FDA will not be applied.", file=sys.stderr)
        self.num_input_channels: Optional[int] = None
        if self.patch_basenames:
            with h5py.File(self.data_dir / f"{self.patch_basenames[0]}_source.h5", "r") as hf:
                self.num_input_channels = hf["image"].shape[0]

    def __len__(self):
        return len(self.patch_basenames)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        basename = self.patch_basenames[idx]
        try:
            with h5py.File(self.data_dir / f"{basename}_source.h5", "r") as hf: image = torch.from_numpy(hf["image"][:])
            with h5py.File(self.data_dir / f"{basename}_target_binary.h5", "r") as hf: target = torch.from_numpy(hf["target"][:]).unsqueeze(0)
            with h5py.File(self.data_dir / f"{basename}_mask_binary.h5", "r") as hf: mask = torch.from_numpy(hf["mask"][:]).unsqueeze(0)
            style_image = torch.zeros_like(image)
            if self.use_fda and self.style_paths and random.random() < config.STYLE_IMAGE_AUGMENTATION_PROBABILITY:
                with h5py.File(random.choice(self.style_paths), "r") as hf: style_image = torch.from_numpy(hf["style_image"][:])
                image = apply_style_image(image.byte(), style_image.byte(), beta=0.05)
            if self.photometric_augs: image = self.photometric_augs(image.byte())
            if self.geometric_augs:
                stacked = torch.cat((image, target, mask, style_image), dim=0)
                augmented = self.geometric_augs(stacked)
                image, target, mask, style_image = torch.split(augmented, [self.num_input_channels, 1, 1, self.num_input_channels], dim=0)
            return {'image': image.float(), 'target': target.float(), 'mask': mask.float(), 'style_image': style_image.float()}
        except Exception as e:
            print(f"Error loading HDF5 file for basename {basename}: {e}", file=sys.stderr); raise

class MaskedBCELoss(nn.Module):
    def __init__(self):
        super().__init__(); self.criterion = nn.BCEWithLogitsLoss(reduction='none')
    def forward(self, outputs, targets, mask):
        pixel_loss = self.criterion(outputs, targets); masked_loss = pixel_loss * mask
        num_valid_pixels = mask.sum().clamp(min=1)
        return masked_loss.sum() / num_valid_pixels

# --- Training Loop ---
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, deep_supervision, inv_loss_weight, model_save_path, results_csv_path):

    # --- Track the best metrics across all epochs ---
    best_val_f1 = -1.0
    best_val_metrics = {'recall': 0.0, 'precision': 0.0, 'f1': 0.0}

    # --- List to store detailed epoch-by-epoch logs ---
    epoch_log_history = []

    invariance_criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        # Use stderr for tqdm to keep stdout clean for the final JSON result
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", file=sys.stderr, dynamic_ncols=True)
        for batch in pbar:
            # Training step... (logic is unchanged)
            inputs, targets, masks, style_images = \
                batch['image'].to(device), batch['target'].to(device), batch['mask'].to(device), batch['style_image'].to(device)
            inputs_augmented = torch.clamp(inputs + style_images, 0, 255)
            optimizer.zero_grad()
            logits_clean1, logits_clean2, logits_clean3, final_logits_clean, features_clean = model(inputs)
            _, _, _, _, features_aug = model(inputs_augmented)
            if deep_supervision:
                loss1, loss2, loss3, loss4 = criterion(logits_clean1, targets, masks), criterion(logits_clean2, targets, masks), criterion(logits_clean3, targets, masks), criterion(final_logits_clean, targets, masks)
                segmentation_loss = loss1 + loss2 + loss3 + loss4
            else:
                segmentation_loss = criterion(final_logits_clean, targets, masks)
            invariance_loss = invariance_criterion(features_clean, features_aug)
            total_loss = segmentation_loss + (inv_loss_weight * invariance_loss)
            if torch.isfinite(total_loss):
                total_loss.backward(); optimizer.step()
                running_loss += total_loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # --- Validation Step ---
        model.eval()
        val_loss, val_tp, val_fp, val_fn = 0.0, 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets, masks = batch['image'].to(device), batch['target'].to(device), batch['mask'].to(device)
                _, _, _, logits, _ = model(inputs)
                loss = criterion(logits, targets, masks)
                val_loss += loss.item()
                preds = torch.sigmoid(logits) > 0.5
                tp, fp, fn = calculate_metrics(preds, targets, masks)
                val_tp += tp.item(); val_fp += fp.item(); val_fn += fn.item()

        avg_val_loss = val_loss / len(val_loader)
        val_precision = val_tp / (val_tp + val_fp + 1e-6)
        val_recall = val_tp / (val_tp + val_fn + 1e-6)
        val_f1 = 2 * (val_precision * val_recall) / (val_precision + val_recall + 1e-6)

        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val F1: {val_f1:.4f}", file=sys.stderr)

        # --- Check if this is the best model so far ---
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_metrics = {'recall': val_recall, 'precision': val_precision, 'f1': val_f1}
            torch.save(model.state_dict(), model_save_path)
            print(f"  -> New best model saved with F1: {val_f1:.4f}", file=sys.stderr)

        # --- NEW: Log detailed metrics for this epoch and save to CSV ---
        epoch_log = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_f1': val_f1
        }
        epoch_log_history.append(epoch_log)
        pd.DataFrame(epoch_log_history).to_csv(results_csv_path, index=False)
        # --- END NEW ---

    return best_val_metrics

def main(args):
    lr = args.learning_rate
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    deep_supervision = args.deep_supervision
    inv_loss_weight = args.invariance_loss_weight
    model_save_path = Path(args.model_save_path)
    # --- NEW: Get the results CSV path ---
    results_csv_path = Path(args.results_csv_path)

    set_seed(config.GLOBAL_RANDOM_SEED)
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    results_csv_path.parent.mkdir(parents=True, exist_ok=True)

    photometric_transforms = transforms.Compose([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))])
    geometric_transforms = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()])

    style_dir = config.PREPROCESSED_DATA_DIR / "style_images"
    train_dataset = PreprocessedBinaryVegDataset(config.TRAIN_DIR, style_dir=style_dir, use_fda=True, photometric_augs=photometric_transforms, geometric_augs=geometric_transforms)
    val_dataset = PreprocessedBinaryVegDataset(config.VAL_DIR, style_dir=style_dir, use_fda=False)

    if len(train_dataset) == 0:
        print("Error: Training dataset is empty. Exiting.", file=sys.stderr); return

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = Model(in_channels=train_dataset.num_input_channels, out_channels=1).to(config.DEVICE)
    criterion = MaskedBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if len(train_dataset) > 0 and len(val_dataset) > 0:
        best_metrics = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, config.DEVICE, deep_supervision, inv_loss_weight, model_save_path, results_csv_path)

        final_results = {
            "best_val_recall": best_metrics['recall'],
            "best_val_precision": best_metrics['precision'],
            "best_val_f1": best_metrics['f1'],
            "model_path": str(model_save_path)
        }
        # Print final JSON result to stdout for the search script
        print(json.dumps(final_results))
    else:
        print("Warning: Train or validation loader is empty. Skipping training.", file=sys.stderr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model with specific hyperparameters.")
    parser.add_argument('--learning-rate', type=float, required=True, help="Learning rate")
    parser.add_argument('--batch-size', type=int, required=True, help="Batch size")
    parser.add_argument('--num-epochs', type=int, default=config.NUM_EPOCHS, help="Number of training epochs")
    parser.add_argument('--deep-supervision', type=lambda x: (str(x).lower() == 'true'), required=True, help="Enable deep supervision (True/False)")
    parser.add_argument('--invariance-loss-weight', type=float, required=True, help="Weight for the invariance loss")
    parser.add_argument('--model-save-path', type=str, required=True, help="Full path to save the best model checkpoint")
    # --- NEW ARGUMENT ---
    parser.add_argument('--results-csv-path', type=str, required=True, help="Full path to save the epoch-by-epoch metrics CSV log")

    args = parser.parse_args()
    main(args)

