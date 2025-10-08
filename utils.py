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
    """
    Sets the random seed for all relevant libraries to ensure reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to {seed}")


def create_inference_visualization(model, dataset: Dataset, device: torch.device, output_dir: Path, epoch: int, num_samples: int = 10, split_name: str = 'split'):
    """
    Generates and saves a single PNG file containing a grid of visualizations
    for multiple random samples from a dataset.
    """
    if not dataset or len(dataset) == 0:
        print(f"Dataset for '{split_name}' is empty. Skipping visualization.")
        return

    # Create the specific subdirectory for the split (e.g., visualizations/train/)
    split_output_dir = output_dir / split_name
    split_output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    num_samples = min(num_samples, len(dataset))
    if num_samples == 0:
        return

    # --- MODIFICATION: Create one large figure for all samples ---
    # Figure with `num_samples` rows and 4 columns
    fig, axes = plt.subplots(nrows=num_samples, ncols=4, figsize=(16, 4 * num_samples))
    # If num_samples is 1, axes will not be a 2D array, so we fix that
    if num_samples == 1:
        axes = np.array([axes])

    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, total=num_samples, desc=f"Visualizing {split_name} samples for epoch {epoch}")):
            if i >= num_samples:
                break

            inputs, targets, masks = batch['image'].to(device), batch['target'], batch['mask']
            outputs = model(inputs)
            preds = torch.sigmoid(outputs) > 0.5

            img_np, target_np, mask_np, pred_np = inputs[0].cpu().numpy(), targets[0, 0].cpu().numpy(), masks[0, 0].cpu().numpy(), preds[0, 0].cpu().numpy()

            img_rgb = img_np[:3, :, :].astype(np.float32)
            for ch in range(img_rgb.shape[0]):
                min_val, max_val = np.min(img_rgb[ch]), np.max(img_rgb[ch])
                if max_val > min_val:
                    img_rgb[ch] = (img_rgb[ch] - min_val) / (max_val - min_val)
            img_rgb = np.transpose(img_rgb, (1, 2, 0))

            # --- MODIFICATION: Plot on the i-th row of the grid ---
            row_axes = axes[i]

            # Plot the four images for the current sample
            row_axes[0].imshow(img_rgb)
            row_axes[1].imshow(target_np, cmap='gray')
            row_axes[2].imshow(pred_np, cmap='gray')
            row_axes[3].imshow(mask_np, cmap='magma')

            # Add a label to the first column to identify the sample
            row_axes[0].set_ylabel(f"Sample {i+1}", size='large')

            # Set titles only for the very first row to avoid clutter
            if i == 0:
                row_axes[0].set_title("Input Image (RGB)")
                row_axes[1].set_title("Ground Truth")
                row_axes[2].set_title("Model Prediction")
                row_axes[3].set_title("Loss Mask")

            # Turn off axis ticks and labels for all subplots
            for ax in row_axes:
                ax.set_xticks([])
                ax.set_yticks([])

    # --- MODIFICATION: Save the single, large figure ---
    fig.tight_layout(pad=1.5)
    filename = f"epoch_{epoch:02d}_{split_name}_summary.png"
    plt.savefig(split_output_dir / filename, dpi=120, bbox_inches='tight')
    plt.close(fig)

