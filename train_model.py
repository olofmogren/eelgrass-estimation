import random
from pathlib import Path
from typing import List, Dict, Optional
import h5py
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np

# Import from our config and utils files
import config
from utils import create_inference_visualization, set_seed

# --- Dataset Class ---
class PreprocessedBinaryVegDataset(Dataset):
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.patch_basenames: List[str] = []
        if not self.data_dir.exists():
            print(f"Warning: Data directory not found: {self.data_dir}")
            return

        for h5_file in self.data_dir.glob("*_source.h5"):
            self.patch_basenames.append(h5_file.name.replace("_source.h5", ""))
        
        if not self.patch_basenames:
            print(f"Warning: No patches found in {self.data_dir}")
        else:
             print(f"  Dataset for {data_dir.name}: Found {len(self.patch_basenames)} samples.")
            
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
                inputs_np = hf["image"][:].astype(np.float32)
            with h5py.File(self.data_dir / f"{basename}_target_binary.h5", "r") as hf:
                targets_np = hf["target"][:].astype(np.float32)
            with h5py.File(self.data_dir / f"{basename}_mask_binary.h5", "r") as hf:
                mask_np = hf["mask"][:].astype(np.float32)

            return {
                'image': torch.from_numpy(inputs_np),
                'target': torch.from_numpy(targets_np).unsqueeze(0),
                'mask': torch.from_numpy(mask_np).unsqueeze(0)
            }
        except Exception as e:
            print(f"Error loading HDF5 file for basename {basename}: {e}"); raise
# --- U-Net Model (Standard Implementation) ---

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Use a transposed convolution to upsample
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Pad x1 if it's smaller than x2
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class RealUNet(nn.Module):
    """
    A more standard U-Net implementation.
    This is a drop-in replacement for the previous simple version.
    """
    def __init__(self, in_channels, out_channels=1):
        super(RealUNet, self).__init__()

        # Encoder
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        # Decoder
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, out_channels)

    def forward(self, x):
        # --- Encoder Path ---
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # --- Decoder Path with Skip Connections ---
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # --- Final Output ---
        logits = self.outc(x)
        return logits


# --- U-Net Model ---
class SimpleUNet(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super().__init__()
        def _conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, 1, 1, bias=False), nn.BatchNorm2d(out_c), nn.ReLU(True),
                nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False), nn.BatchNorm2d(out_c), nn.ReLU(True),
            )
        self.enc1 = _conv_block(in_channels, 64)
        self.enc2 = _conv_block(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = _conv_block(128, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = _conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = _conv_block(128, 64)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.bottleneck(self.pool(e2))
        d2 = self.upconv2(b)
        d2 = torch.cat([d2, e2], dim=1); d2 = self.dec2(d2)
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1); d1 = self.dec1(d1)
        return self.final_conv(d1)

# --- Masked Loss Function ---
class MaskedBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, outputs, targets, mask):
        pixel_loss = self.criterion(outputs, targets)
        masked_loss = pixel_loss * mask
        num_valid_pixels = mask.sum().clamp(min=1)
        return masked_loss.sum() / num_valid_pixels

# --- Training and Validation ---
def train_model(model, train_dataset, val_dataset, test_dataset, criterion, optimizer, num_epochs, device):
    """ The main training loop, now includes per-epoch visualization. """
    
    # Create DataLoaders here
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    print(f"--- Starting training on {device} ---")
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # --- Training Step ---
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
        
        # --- Validation Step ---
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

        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), config.CHECKPOINT_DIR / "best_model_binary.pth")
            print(f"  -> New best model saved with val loss: {best_val_loss:.4f}")
            
        # --- Per-Epoch Visualization Step ---
        print(f"--- Generating visualizations for Epoch {epoch+1} ---")
        
        # Pass the current epoch number (epoch + 1) to the visualization function.
        # The output directory is now the base visualization directory.
        create_inference_visualization(model, train_dataset, device, config.VISUALIZATION_DIR, epoch=epoch + 1, num_samples=10, split_name='train')
        create_inference_visualization(model, val_dataset, device, config.VISUALIZATION_DIR, epoch=epoch + 1, num_samples=10, split_name='val')
        create_inference_visualization(model, test_dataset, device, config.VISUALIZATION_DIR, epoch=epoch + 1, num_samples=10, split_name='test')

    print("--- Training finished. ---")

def main():
    set_seed(config.GLOBAL_RANDOM_SEED)
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    config.VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)

    # --- Initialize Datasets ---
    print("Loading datasets...")
    train_dataset = PreprocessedBinaryVegDataset(config.TRAIN_DIR)
    val_dataset = PreprocessedBinaryVegDataset(config.VAL_DIR)
    test_dataset = PreprocessedBinaryVegDataset(config.TEST_DIR)

    if len(train_dataset) == 0 or train_dataset.num_input_channels is None:
        print("Error: Training dataset is empty. Preprocessing might have failed. Exiting.")
        return
        
    # --- Initialize Model, Loss, Optimizer ---
    print(f"Initializing model with {train_dataset.num_input_channels} input channels.")
    model = RealUNet(in_channels=train_dataset.num_input_channels, out_channels=1).to(config.DEVICE)
    criterion = MaskedBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # --- Train the Model ---
    if len(train_dataset) > 0 and len(val_dataset) > 0:
        # Pass the datasets to the training function
        train_model(model, train_dataset, val_dataset, test_dataset, criterion, optimizer, config.NUM_EPOCHS, config.DEVICE)
    else:
        print("Warning: Train or validation loader is empty. Skipping training.")

    print(f"\nScript finished. Best model saved to: '{config.CHECKPOINT_DIR / 'best_model_binary.pth'}'")
    print(f"Per-epoch visualizations saved in: '{config.VISUALIZATION_DIR}'")

if __name__ == "__main__":
    main()

