import random
from pathlib import Path
from typing import List, Dict, Optional
import h5py
from tqdm import tqdm
from datetime import datetime
import pandas as pd

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

# --- Dataset Class (Unchanged) ---
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
            print(f"Warning: Data directory not found: {self.data_dir}")
        else:
            self.patch_basenames = [p.name.replace("_source.h5", "") for p in data_dir.glob("*_source.h5")]
            if not self.patch_basenames:
                print(f"Warning: No patches found in {self.data_dir}")
            else:
                print(f"  Dataset for {data_dir.name}: Found {len(self.patch_basenames)} samples.")
        if style_dir and style_dir.exists():
            self.style_paths = list(style_dir.glob("*.h5"))
        if self.use_fda and not self.style_paths:
            print(f"    - FDA is enabled for {data_dir.name}, but no style images were found. FDA will not be applied.")
        self.num_input_channels: Optional[int] = None
        if self.patch_basenames:
            with h5py.File(self.data_dir / f"{self.patch_basenames[0]}_source.h5", "r") as hf:
                self.num_input_channels = hf["image"].shape[0]

    def __len__(self):
        return len(self.patch_basenames)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        basename = self.patch_basenames[idx]
        try:
            with h5py.File(self.data_dir / f"{basename}_source.h5", "r") as hf:
                image = torch.from_numpy(hf["image"][:])
            with h5py.File(self.data_dir / f"{basename}_target_binary.h5", "r") as hf:
                target = torch.from_numpy(hf["target"][:]).unsqueeze(0)
            with h5py.File(self.data_dir / f"{basename}_mask_binary.h5", "r") as hf:
                mask = torch.from_numpy(hf["mask"][:]).unsqueeze(0)

            style_image = torch.zeros_like(image)

            if self.use_fda and self.style_paths and random.random() < config.STYLE_IMAGE_AUGMENTATION_PROBABILITY:
                with h5py.File(random.choice(self.style_paths), "r") as hf:
                    style_image = torch.from_numpy(hf["style_image"][:])
                image = apply_style_image(image.byte(), style_image.byte(), beta=0.05)

            if self.photometric_augs:
                image = self.photometric_augs(image.byte())

            if self.geometric_augs:
                stacked = torch.cat((image, target, mask, style_image), dim=0)
                augmented = self.geometric_augs(stacked)
                image, target, mask, style_image = torch.split(augmented, [self.num_input_channels, 1, 1, self.num_input_channels], dim=0)

            return {'image': image.float(), 'target': target.float(), 'mask': mask.float(), 'style_image': style_image.float()}
        except Exception as e:
            print(f"Error loading HDF5 file for basename {basename}: {e}"); raise

# --- Masked Loss Function (Unchanged) ---
class MaskedBCELoss(nn.Module):
    def __init__(self):
        super().__init__(); self.criterion = nn.BCEWithLogitsLoss(reduction='none')
    def forward(self, outputs, targets, mask):
        pixel_loss = self.criterion(outputs, targets); masked_loss = pixel_loss * mask
        num_valid_pixels = mask.sum().clamp(min=1)
        return masked_loss.sum() / num_valid_pixels

# --- Training Loop (MODIFIED) ---
def train_model(model, train_dataset, val_dataset, test_dataset, criterion, optimizer, num_epochs, device):
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    print(f"--- Starting training on {device} ---")

    # --- NEW: Create filename and prepare for CSV logging ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Create a string from key config settings for the filename
    config_str = (
        f"LR_{config.LEARNING_RATE}_BS_{config.BATCH_SIZE}_EPOCHS_{config.NUM_EPOCHS}"
        f"_DS_{config.DEEP_SUPERVISION}_ILW_{config.INVARIANCE_LOSS_WEIGHT}"
    )
    csv_filename = f"validation_scores_{timestamp}_{config_str}.csv"
    csv_filepath = config.VISUALIZATION_DIR / csv_filename
    validation_log = [] # List to store dicts of epoch results

    print(f"Validation scores will be saved to: {csv_filepath}")
    # --- END NEW ---

    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    train_metrics_history, val_metrics_history = [], []
    invariance_criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_tp, train_fp, train_fn = 0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")

        for batch in pbar:
            inputs, targets, masks, style_images = \
                batch['image'].to(device), batch['target'].to(device), batch['mask'].to(device), batch['style_image'].to(device)

            inputs_augmented = torch.clamp(inputs + style_images, 0, 255)
            optimizer.zero_grad()

            logits_clean1, logits_clean2, logits_clean3, final_logits_clean, features_clean = model(inputs)
            _, _, _, _, features_aug = model(inputs_augmented)

            if config.DEEP_SUPERVISION:
                loss1 = criterion(logits_clean1, targets, masks)
                loss2 = criterion(logits_clean2, targets, masks)
                loss3 = criterion(logits_clean3, targets, masks)
                loss4 = criterion(final_logits_clean, targets, masks)
                segmentation_loss = loss1 + loss2 + loss3 + loss4
            else:
                segmentation_loss = criterion(final_logits_clean, targets, masks)

            invariance_loss = invariance_criterion(features_clean, features_aug)
            total_loss = segmentation_loss + (config.INVARIANCE_LOSS_WEIGHT * invariance_loss)

            if torch.isfinite(total_loss):
                total_loss.backward(); optimizer.step()
                running_loss += total_loss.item()
                pbar.set_postfix(seg_loss=f"{segmentation_loss.item():.4f}", inv_loss=f"{invariance_loss.item():.4f}")
            else:
                print('loss is inf!')

            preds = torch.sigmoid(final_logits_clean) > 0.5
            tp, fp, fn = calculate_metrics(preds, targets, masks)
            train_tp += tp.item(); train_fp += fp.item(); train_fn += fn.item()

        avg_train_loss = running_loss / len(train_loader)
        train_precision = train_tp / (train_tp + train_fp + 1e-6)
        train_recall = train_tp / (train_tp + train_fn + 1e-6)
        train_f1 = 2 * (train_precision * train_recall) / (train_precision + train_recall + 1e-6)
        train_losses.append(avg_train_loss);
        train_metrics_history.append({'precision': train_precision, 'recall': train_recall, 'f1': train_f1})

        model.eval()
        val_loss = 0.0
        val_tp, val_fp, val_fn = 0, 0, 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
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
        val_losses.append(avg_val_loss)
        val_metrics_history.append({'precision': val_precision, 'recall': val_recall, 'f1': val_f1})
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val F1: {val_f1:.4f}")

        # --- NEW: Log results for this epoch and save to CSV ---
        epoch_log = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_f1': val_f1
        }
        validation_log.append(epoch_log)

        # Convert to DataFrame and save, overwriting the file each time
        log_df = pd.DataFrame(validation_log)
        log_df.to_csv(csv_filepath, index=False)
        # --- END NEW ---

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), config.CHECKPOINT_DIR / "best_model_binary.pth")
            print(f"  -> New best model saved with val loss: {best_val_loss:.4f}")

        save_loss_plot(train_losses, val_losses, config.VISUALIZATION_DIR / "loss_curve.png")
        save_metrics_plot(train_metrics_history, val_metrics_history, config.VISUALIZATION_DIR / "metrics_curve.png")
        create_inference_visualization(model, train_dataset, device, config.VISUALIZATION_DIR, epoch=epoch + 1, num_samples=10, split_name='train')
        create_inference_visualization(model, val_dataset, device, config.VISUALIZATION_DIR, epoch=epoch + 1, num_samples=10, split_name='val')
        create_inference_visualization(model, test_dataset, device, config.VISUALIZATION_DIR, epoch=epoch + 1, num_samples=10, split_name='test')

    print("--- Training finished. ---")

# --- Main function (Unchanged) ---
def main():
    set_seed(config.GLOBAL_RANDOM_SEED)
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    config.VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)

    photometric_transforms = transforms.Compose([
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
    ])
    geometric_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ])

    print("Loading datasets...")

    style_dir = config.PREPROCESSED_DATA_DIR / "style_images"

    train_dataset = PreprocessedBinaryVegDataset(
        config.TRAIN_DIR, style_dir=style_dir, use_fda=True,
        photometric_augs=photometric_transforms, geometric_augs=geometric_transforms
    )
    val_dataset = PreprocessedBinaryVegDataset(config.VAL_DIR, style_dir=style_dir, use_fda=False)
    test_dataset = PreprocessedBinaryVegDataset(config.TEST_DIR, style_dir=style_dir, use_fda=False)

    if len(train_dataset) == 0 or train_dataset.num_input_channels is None:
        print("Error: Training dataset is empty. Exiting."); return

    print(f"Initializing model with {train_dataset.num_input_channels} input channels.")
    model = Model(in_channels=train_dataset.num_input_channels, out_channels=1).to(config.DEVICE)
    criterion = MaskedBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    if len(train_dataset) > 0 and len(val_dataset) > 0:
        train_model(model, train_dataset, val_dataset, test_dataset, criterion, optimizer, config.NUM_EPOCHS, config.DEVICE)
    else:
        print("Warning: Train or validation loader is empty. Skipping training.")

    print(f"\nScript finished. Best model saved to: '{config.CHECKPOINT_DIR / 'best_model_binary.pth'}'")
    print(f"Visualizations and loss curve saved in: '{config.VISUALIZATION_DIR}'")

if __name__ == "__main__":
    main()

