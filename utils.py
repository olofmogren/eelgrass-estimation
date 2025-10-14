from pathlib import Path
import random
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

def set_seed(seed: int = 42):
    """Sets the random seed for all relevant libraries to ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to {seed}")

def save_loss_plot(train_losses: list, val_losses: list, output_path: Path):
    """Generates and saves a plot of training and validation loss curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def apply_fda(source_img: torch.Tensor, target_img: torch.Tensor, beta: float = 0.1):
    """
    Applies Fourier Domain Adaptation, transferring the 'style' of the target
    image to the 'content' of the source image.
    """
    source_img = source_img.float()
    target_img = target_img.float()
    
    is_3d = len(source_img.shape) == 3
    if is_3d:
        source_img = source_img.unsqueeze(0)
        target_img = target_img.unsqueeze(0)

    fft_src = torch.fft.fft2(source_img, dim=(-2, -1))
    fft_trg = torch.fft.fft2(target_img, dim=(-2, -1))
    
    fft_src_shifted = torch.fft.fftshift(fft_src, dim=(-2, -1))
    fft_trg_shifted = torch.fft.fftshift(fft_trg, dim=(-2, -1))
    
    _, _, h, w = fft_src.shape
    b = int(min(h, w) * beta)
    
    center_mask = torch.zeros_like(fft_src_shifted)
    center_mask[:, :, h//2 - b:h//2 + b, w//2 - b:w//2 + b] = 1
    
    mag_src = torch.abs(fft_src_shifted)
    mag_trg = torch.abs(fft_trg_shifted)
    
    fft_src_mag_swapped = mag_trg * center_mask + mag_src * (1 - center_mask)
    fft_src_swapped = fft_src_mag_swapped * torch.exp(1j * torch.angle(fft_src_shifted))
    
    fft_src_swapped_unshifted = torch.fft.ifftshift(fft_src_swapped, dim=(-2, -1))
    fda_img = torch.fft.ifft2(fft_src_swapped_unshifted, dim=(-2, -1))
    
    fda_img = torch.real(fda_img)
    
    if is_3d:
        fda_img = fda_img.squeeze(0)
    
    fda_img = torch.clamp(fda_img, 0, 255)
    
    return fda_img

def create_inference_visualization(model, dataset: Dataset, device: torch.device, output_dir: Path, epoch: int, num_samples: int = 10, split_name: str = 'split'):
    """
    Generates and saves a single PNG file containing a grid of visualizations.
    """
    if not dataset or len(dataset) == 0:
        print(f"Dataset for '{split_name}' is empty. Skipping visualization.")
        return

    split_output_dir = output_dir / split_name
    split_output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    
    num_samples = min(num_samples, len(dataset))
    if num_samples == 0: return

    is_train = split_name == 'train'
    num_cols = 5 if is_train else 4
    fig_width = 20 if is_train else 16
    fig, axes = plt.subplots(nrows=num_samples, ncols=num_cols, figsize=(fig_width, 4 * num_samples))
    if num_samples == 1: axes = np.array([axes])

    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, total=num_samples, desc=f"Visualizing {split_name} samples for epoch {epoch}")):
            if i >= num_samples: break
            
            inputs, targets, masks = batch['image'], batch['target'], batch['mask']
            
            # --- THIS IS THE FIX ---
            # During eval, model only returns logits
            outputs = model(inputs.to(device))
            preds = torch.sigmoid(outputs) > 0.5

            img_np = inputs[0].cpu().numpy()
            target_np = targets[0, 0].cpu().numpy()
            mask_np = masks[0, 0].cpu().numpy()
            pred_np = preds[0, 0].cpu().numpy()
            
            input_pixel_sum = int(np.sum(img_np))
            veg_pixel_count = int(np.sum(target_np))

            img_rgb_to_display = np.transpose(img_np, (1, 2, 0)).astype(np.uint8)

            row_axes = axes[i]

            row_axes[0].imshow(img_rgb_to_display, vmin=0, vmax=255)
            row_axes[1].imshow(target_np, cmap='gray')
            row_axes[2].imshow(pred_np, cmap='gray')
            row_axes[3].imshow(mask_np, cmap='magma')
            
            row_axes[0].set_ylabel(f"Sample {i+1}", size='large')
            
            row_axes[0].text(5, 18, f"Sum: {input_pixel_sum}", 
                             color='white', fontsize=10, 
                             bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2'))
            row_axes[1].text(5, 18, f"Veg Pixels: {veg_pixel_count}", 
                             color='white', fontsize=10, 
                             bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2'))

            if is_train:
                augmented_image_np = img_np
                if hasattr(dataset, 'style_paths') and dataset.style_paths:
                    with h5py.File(random.choice(dataset.style_paths), "r") as hf:
                        style_image_np = hf["style_image"][:]
                    
                    augmented_image_tensor = apply_fda(torch.from_numpy(img_np), torch.from_numpy(style_image_np))
                    augmented_image_np = augmented_image_tensor.numpy()

                augmented_image_display = np.transpose(augmented_image_np, (1, 2, 0)).astype(np.uint8)
                row_axes[4].imshow(augmented_image_display, vmin=0, vmax=255)

            if i == 0:
                row_axes[0].set_title("Input Image (RGB)")
                row_axes[1].set_title("Ground Truth")
                row_axes[2].set_title("Model Prediction")
                row_axes[3].set_title("Loss Mask")
                if is_train:
                    row_axes[4].set_title("Input + FDA Style")
            
            for ax in row_axes:
                ax.set_xticks([]); ax.set_yticks([])

    fig.tight_layout(pad=1.5)
    filename = f"epoch_{epoch:02d}_{split_name}_summary.png"
    plt.savefig(split_output_dir / filename, dpi=120, bbox_inches='tight')
    plt.close(fig)

