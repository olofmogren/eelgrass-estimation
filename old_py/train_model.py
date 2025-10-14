import os
import random
from pathlib import Path
from typing import List, Dict, Optional
import argparse
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import numpy as np

# --- Configuration ---
PREPROCESSED_DATA_DIR = Path("preprocessed_data")
TRAIN_DIR = PREPROCESSED_DATA_DIR / "train"
VAL_DIR = PREPROCESSED_DATA_DIR / "val"
CHECKPOINT_DIR = Path("checkpoints")
VISUALIZATION_DIR = Path("visualizations")

BATCH_SIZE = 8
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GLOBAL_RANDOM_SEED = 42

# --- Dataset Class ---
class PreprocessedBinaryVegDataset(Dataset):
    """
    Dataset to load preprocessed patches for binary vegetation segmentation.
    It expects source, target, and mask files for each patch.
    """
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.patch_basenames: List[str] = []
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        # Find all source files and store their basenames
        for h5_file in self.data_dir.glob("*_source.h5"):
            self.patch_basenames.append(h5_file.name.replace("_source.h5", ""))
        
        if not self.patch_basenames:
            print(f"Warning: No patches found in {self.data_dir}")
            
        # Determine the number of input channels from the first patch
        self.num_input_channels: Optional[int] = None
        if self.patch_basenames:
            with h5py.File(self.data_dir / f"{self.patch_basenames[0]}_source.h5", "r") as hf:
                self.num_input_channels = hf["image"].shape[0]
            print(f"  Dataset for {data_dir.name}: {len(self.patch_basenames)} samples found with {self.num_input_channels} input channels.")

    def __len__(self):
        return len(self.patch_basenames)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        basename = self.patch_basenames[idx]
        source_path = self.data_dir / f"{basename}_source.h5"
        target_path = self.data_dir / f"{basename}_target_binary.h5"
        
        # NOTE: Assumes a mask file exists. The preprocessing script must generate this.
        # This mask should be 1 at every annotated pixel and 0 elsewhere.
        mask_path = self.data_dir / f"{basename}_mask_binary.h5"

        try:
            with h5py.File(source_path, "r") as hf:
                inputs_np = hf["image"][:].astype(np.float32)
            with h5py.File(target_path, "r") as hf:
                targets_np = hf["target"][:].astype(np.float32)
            
            # Create a mask on the fly if the file doesn't exist
            if mask_path.exists():
                 with h5py.File(mask_path, "r") as hf:
                    mask_np = hf["mask"][:].astype(np.float32)
            else: # Fallback: assume any non-zero value in target implies a valid pixel
                print(f"Warning: Mask file not found for {basename}. Creating a temporary mask from the target. It is recommended to create explicit mask files.")
                mask_np = (targets_np > -1).astype(np.float32) # In this binary case, this captures all annotated pixels

            return {
                'image': torch.from_numpy(inputs_np),
                'target': torch.from_numpy(targets_np).unsqueeze(0), # Add channel dim
                'mask': torch.from_numpy(mask_np).unsqueeze(0) # Add channel dim
            }
        except Exception as e:
            print(f"Error loading HDF5 file for basename {basename}: {e}"); raise

# --- U-Net Model ---
class SimpleUNet(nn.Module):
    """A simple U-Net for binary segmentation."""
    def __init__(self, in_channels, out_channels=1):
        super().__init__()
        self.enc1 = self._conv_block(in_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.pool = nn.MaxPool2d(2)
        
        self.bottleneck = self._conv_block(128, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(128, 64)
        
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def _conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.bottleneck(self.pool(e2))
        d2 = self.upconv2(b)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        return self.final_conv(d1)

# --- Masked Loss Function ---
class MaskedBCELoss(nn.Module):
    """BCEWithLogitsLoss that is masked to only include annotated pixels."""
    def __init__(self):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor):
        pixel_loss = self.criterion(outputs, targets)
        masked_loss = pixel_loss * mask
        
        # Normalize by the number of valid pixels
        num_valid_pixels = mask.sum().clamp(min=1)
        return masked_loss.sum() / num_valid_pixels

# --- Training and Validation ---
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    print(f"--- Starting training on {device} ---")
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch in pbar:
            inputs, targets, masks = batch['image'].to(device), batch['target'].to(device), batch['mask'].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets, masks)
            
            if torch.isfinite(loss):
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        avg_train_loss = running_loss / len(train_loader)
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                inputs, targets, masks = batch['image'].to(device), batch['target'].to(device), batch['mask'].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets, masks)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), CHECKPOINT_DIR / "best_model_binary.pth")
            print(f"  -> New best model saved with val loss: {best_val_loss:.4f}")
        
        # Visualize a few samples from the validation set
        visualize_predictions(model, val_loader, device, epoch)

    print("--- Training finished. ---")

def visualize_predictions(model, loader, device, epoch, num_samples=3):
    """Saves a visualization of model predictions on a few samples."""
    model.eval()
    samples_visualized = 0
    vis_dir = VISUALIZATION_DIR / "epoch_previews"
    vis_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch in loader:
            if samples_visualized >= num_samples: break
            
            inputs, targets = batch['image'].to(device), batch['target'].to(device)
            outputs = model(inputs)
            preds = torch.sigmoid(outputs) > 0.5 # Apply sigmoid and threshold
            
            for i in range(inputs.size(0)):
                if samples_visualized >= num_samples: break
                
                # Prepare images for plotting
                img_rgb = inputs[i][:3].cpu().numpy().astype(np.uint8)
                for ch in range(img_rgb.shape[0]): # Normalize each channel
                    min_val, max_val = np.min(img_rgb[ch]), np.max(img_rgb[ch])
                    if max_val > min_val: img_rgb[ch] = (img_rgb[ch] - min_val) / (max_val - min_val)
                img_rgb = np.transpose(img_rgb, (1, 2, 0))

                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(img_rgb); axes[0].set_title("Input RGB")
                axes[1].imshow(targets[i, 0].cpu().numpy(), cmap='gray'); axes[1].set_title("Ground Truth")
                axes[2].imshow(preds[i, 0].cpu().numpy(), cmap='gray'); axes[2].set_title("Prediction")
                for ax in axes: ax.axis('off')

                plt.suptitle(f"Epoch {epoch+1} - Sample {samples_visualized+1}")
                plt.tight_layout()
                plt.savefig(vis_dir / f"epoch_{epoch+1:02d}_sample_{i}.png")
                plt.close(fig)

                samples_visualized += 1

def main():
    parser = argparse.ArgumentParser(description="Train U-Net model for binary vegetation segmentation.")
    args = parser.parse_args()

    random.seed(GLOBAL_RANDOM_SEED); np.random.seed(GLOBAL_RANDOM_SEED); torch.manual_seed(GLOBAL_RANDOM_SEED)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)

    # --- Initialize Datasets ---
    train_dataset = PreprocessedBinaryVegDataset(TRAIN_DIR)
    val_dataset = PreprocessedBinaryVegDataset(VAL_DIR)

    if len(train_dataset) == 0 or train_dataset.num_input_channels is None:
        print("Error: Training dataset is empty or could not determine input channels. Exiting.")
        return
        
    # --- Initialize Model, Loss, Optimizer ---
    model = SimpleUNet(in_channels=train_dataset.num_input_channels, out_channels=1).to(DEVICE)
    criterion = MaskedBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # --- Create DataLoaders ---
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    if len(train_loader) > 0 and len(val_loader) > 0:
        train_model(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, DEVICE)
    else:
        print("Warning: Train or validation loader is empty. Skipping training.")

    print(f"Script finished. Final model saved to: {CHECKPOINT_DIR/'best_model_binary.pth'}")

if __name__ == "__main__":
    main()
