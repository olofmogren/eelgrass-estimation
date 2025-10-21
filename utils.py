from pathlib import Path
import random
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
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

def apply_style_image(source_img: torch.Tensor, target_img: torch.Tensor, beta: float = 0.1):
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

    augmented_image = source_img+target_img

    if is_3d:
        augmented_image = augmented_image.squeeze(0)
    
    augmented_image = torch.clamp(augmented_image, 0, 255)
    
    return augmented_image

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


def create_inference_visualization(model, dataset, device, output_dir: Path, epoch: int, num_samples: int, split_name: str):
    """Creates a grid visualization of model predictions on a few samples."""
    if len(dataset) == 0:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Adjust number of samples if dataset is smaller
    num_samples = min(num_samples, len(dataset))
    if num_samples == 0:
        return

    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))

    # Set titles for the top row
    if num_samples > 0:
        axes[0, 0].set_title("Input Image (RGB)", fontsize=10)
        axes[0, 1].set_title("Ground Truth", fontsize=10)
        axes[0, 2].set_title("Model Prediction", fontsize=10)
        axes[0, 3].set_title("Loss Mask", fontsize=10)

    model.eval()
    with torch.no_grad():
        for i in range(num_samples):
            # Get a sample from the dataset
            sample_idx = np.random.randint(0, len(dataset))
            sample = dataset[sample_idx]
            image, target, mask = sample['image'], sample['target'], sample['mask']

            # Prepare for model (add batch dimension)
            input_tensor = image.unsqueeze(0).to(device)

            # Get model prediction
            _, _, _, logits, _ = model(input_tensor)
            pred = torch.sigmoid(logits)

            # --- NEW: Calculate and display metrics for this specific sample ---
            # Move tensors to CPU for metric calculation
            pred_cpu = pred.squeeze(0).cpu()
            target_cpu = target.cpu()
            mask_cpu = mask.cpu()

            # Calculate metrics for this single image
            tp, fp, fn = calculate_metrics(pred_cpu, target_cpu, mask_cpu)
            tp, fp, fn = tp.item(), fp.item(), fn.item()

            # Calculate P, R, F1, adding epsilon to avoid division by zero
            precision = tp / (tp + fp + 1e-6)
            recall = tp / (tp + fn + 1e-6)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

            # Create the text string to display
            metrics_text = f"P: {precision:.2f} | R: {recall:.2f} | F1: {f1:.2f}"

            # --- END NEW ---

            # Convert to numpy for plotting
            image_np = image.cpu().numpy().transpose(1, 2, 0)
            image_np = np.clip(image_np / 255.0, 0, 1) # Normalize if not already
            target_np = target.cpu().numpy().squeeze()
            pred_np = pred.squeeze(0).cpu().numpy().squeeze()
            mask_np = mask.cpu().numpy().squeeze()

            # Plot Input Image
            ax = axes[i, 0]
            ax.imshow(image_np)
            ax.set_ylabel(f"Sample {i+1}", rotation=90, size='large')
            ax.set_xticks([]); ax.set_yticks([])
            # Add the metrics text to the input image subplot
            text_obj = ax.text(5, 20, metrics_text, fontsize=12, color='white', fontweight='bold')
            text_obj.set_path_effects([patheffects.withStroke(linewidth=3, foreground='black')])


            # Plot Ground Truth
            ax = axes[i, 1]
            ax.imshow(target_np, cmap='gray')
            ax.set_xticks([]); ax.set_yticks([])

            # Plot Model Prediction
            ax = axes[i, 2]
            ax.imshow(pred_np, cmap='gray', vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])

            # Plot Loss Mask (Prediction overlaid on mask)
            ax = axes[i, 3]
            loss_mask_viz = np.stack([pred_np, pred_np, np.zeros_like(pred_np)], axis=-1)
            loss_mask_viz[mask_np == 0] = 0 # Black out areas outside the mask
            ax.imshow(loss_mask_viz)
            ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout(pad=0.5)
    plt.savefig(output_dir / f"{split_name}_epoch_{epoch:03d}.png", dpi=150)
    plt.close()

def calculate_metrics(preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor):
    """
    Calculates the number of true positives, false positives, and false negatives
    within the masked regions.
    Args:
        preds: The binary prediction tensor from the model (B, 1, H, W).
        targets: The ground truth tensor (B, 1, H, W).
        mask: The mask tensor to apply (B, 1, H, W).
    Returns:
        A tuple containing the total true positives, false positives, and false negatives.
    """
    # Ensure tensors are boolean
    preds = preds.bool()
    targets = targets.bool()

    # Apply the mask to ignore irrelevant pixels
    preds_masked = preds[mask == 1]
    targets_masked = targets[mask == 1]

    tp = (preds_masked & targets_masked).sum()
    fp = (preds_masked & ~targets_masked).sum()
    fn = (~preds_masked & targets_masked).sum()

    return tp, fp, fn

def save_metrics_plot(train_metrics: list, val_metrics: list, output_path: Path):
    """
    Generates and saves a plot of training and validation metrics (Precision, Recall, F1).
    Args:
        train_metrics: A list of dicts, where each dict contains {'precision', 'recall', 'f1'} for training.
        val_metrics: A list of dicts for validation metrics.
        output_path: The path to save the PNG file.
    """
    epochs = range(1, len(train_metrics) + 1)

    plt.figure(figsize=(15, 5))

    # Plot Precision
    plt.subplot(1, 3, 1)
    plt.plot(epochs, [m['precision'] for m in train_metrics], 'b-o', label='Train Precision')
    plt.plot(epochs, [m['precision'] for m in val_metrics], 'r-o', label='Val Precision')
    plt.title('Precision over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.grid(True)
    plt.legend()

    # Plot Recall
    plt.subplot(1, 3, 2)
    plt.plot(epochs, [m['recall'] for m in train_metrics], 'b-o', label='Train Recall')
    plt.plot(epochs, [m['recall'] for m in val_metrics], 'r-o', label='Val Recall')
    plt.title('Recall over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.grid(True)
    plt.legend()

    # Plot F1 Score
    plt.subplot(1, 3, 3)
    plt.plot(epochs, [m['f1'] for m in train_metrics], 'b-o', label='Train F1 Score')
    plt.plot(epochs, [m['f1'] for m in val_metrics], 'r-o', label='Val F1 Score')
    plt.title('F1 Score over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
