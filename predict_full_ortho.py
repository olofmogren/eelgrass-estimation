import torch
from torch.utils.data import Dataset, DataLoader
import rasterio
import rasterio.plot
from rasterio.windows import Window
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import geopandas
from shapely.geometry import Point, Polygon, box
from pyproj import Transformer
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import fiona.path
import gc # ADDED: For explicit garbage collection

# --- Import your model and helper functions ---
from unetplusplus import Model
from preprocess_data import get_vegetation_columns
import config

# Set matplotlib backend to non-interactive to save memory on servers
import matplotlib
matplotlib.use('Agg')

# --- HELPER FUNCTIONS FOR ROI AND BOUNDS ---
# (Keep load_roi_polygon_wgs84, check_roi_overlap, update_combined_bounds as they were)
def load_roi_polygon_wgs84(roi_filepath: Path) -> Polygon:
    if not roi_filepath.exists():
        raise FileNotFoundError(f"ROI file not found at: {roi_filepath}")
    points_wgs84 = pd.read_csv(roi_filepath, header=None, names=['lat', 'lon'])
    return Polygon(zip(points_wgs84.lon, points_wgs84.lat))

def check_roi_overlap(tif_path: Path, roi_poly_wgs84: Polygon) -> bool:
    try:
        with rasterio.open(tif_path) as src:
            raster_bounds_poly = box(*src.bounds)
            transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            transformed_roi_poly = Polygon([transformer.transform(x, y) for x, y in roi_poly_wgs84.exterior.coords])
            return raster_bounds_poly.intersects(transformed_roi_poly)
    except Exception as e:
        print(f"Warning: Could not process {tif_path.name} for ROI check. Error: {e}")
        return False

def update_combined_bounds(current_bounds, new_bounds):
    if current_bounds is None: return new_bounds
    curr_min_x, curr_min_y, curr_max_x, curr_max_y = current_bounds
    new_min_x, new_min_y, new_max_x, new_max_y = new_bounds
    return (min(curr_min_x, new_min_x), min(curr_min_y, new_min_y), max(curr_max_x, new_max_x), max(curr_max_y, new_max_y))

# --- DATASET ---
class SlidingWindowDataset(Dataset):
    def __init__(self, raster_path: Path, patch_size: int, stride: int):
        self.raster_path = raster_path
        self.patch_size = patch_size
        self.stride = stride
        with rasterio.open(self.raster_path) as src:
            self.height = src.height
            self.width = src.width
        self.x_patches = max(1, (self.width - self.patch_size + self.stride - 1) // self.stride + 1)
        self.y_patches = max(1, (self.height - self.patch_size + self.stride - 1) // self.stride + 1)
        self.src = None

    def __len__(self):
        return self.x_patches * self.y_patches

    def __getitem__(self, idx):
        if self.src is None:
            self.src = rasterio.open(self.raster_path)
        try:
            row, col = (idx // self.x_patches), (idx % self.x_patches)
            y_off, x_off = row * self.stride, col * self.stride
            window = Window(x_off, y_off, self.patch_size, self.patch_size)
            patch = self.src.read(window=window, boundless=True, fill_value=0)
            if patch.shape[0] > 3: patch = patch[:3, :, :]
            return {'image': torch.from_numpy(patch).float() / 255.0, 'x': x_off, 'y': y_off}
        except rasterio.errors.RasterioIOError as e:
            print(f"WARNING: Corrupt patch in {self.raster_path.name} at {idx}: {e}")
            return None
            
    # ADDED: Clean up file handles when dataset is destroyed
    def __del__(self):
        if self.src is not None:
            self.src.close()

def collate_fn_skip_corrupt(batch):
    batch = [item for item in batch if item is not None]
    if not batch: return {}
    return torch.utils.data.dataloader.default_collate(batch)

def predict_and_visualize(model, device, ortho_path, output_path, annotations_path, batch_size, combined_bounds, base_land_gdf):
    print(f"\n--- Processing {ortho_path.name} ---")

    patch_size = config.PATCH_WIDTH_PIXELS
    stride = int(patch_size * 0.95)
    
    # --- PHASE 1: PREDICTION ---
    dataset = SlidingWindowDataset(ortho_path, patch_size, stride)
    # Reduced num_workers slightly to reduce total system RAM overhead per file
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate_fn_skip_corrupt)
    temp_prediction_tif_path = output_path.with_suffix(".temp.tif")

    with rasterio.open(ortho_path) as src:
        ortho_crs = src.crs
        ortho_height, ortho_width = src.height, src.width
        meta = src.meta.copy()
        meta.update(driver='GTiff', count=1, dtype='uint8', compress='lzw', nodata=None)

        with rasterio.open(temp_prediction_tif_path, 'w', **meta) as dst:
            with torch.inference_mode():
                for batch in tqdm(dataloader, desc=f"Predicting {ortho_path.name}"):
                    if not batch: continue
                    images = batch['image'].to(device)
                    x_coords, y_coords = batch['x'], batch['y']
                    _, _, _, logits, _ = model(images)[3] # Adjusted to match typical unet++ output if index 3 is logits
                    preds = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(np.uint8)

                    for i in range(preds.shape[0]):
                        y, x = y_coords[i].item(), x_coords[i].item()
                        pred_patch = preds[i, :, :, :]
                        write_h = min(pred_patch.shape[1], ortho_height - y)
                        write_w = min(pred_patch.shape[2], ortho_width - x)
                        dst.write(pred_patch[:, :write_h, :write_w], window=Window(x, y, write_w, write_h))

    # --- PHASE 2: VISUALIZATION ---
    print("Creating visualization...")
    
    # 1. Prepare vector data (reproject pre-loaded land data instead of reloading from disk)
    land_gdf_local = base_land_gdf.to_crs(ortho_crs)

    veg_points, non_veg_points = [], []
    if annotations_path and annotations_path.exists():
        # (Loading annotations code same as before...)
        df = pd.read_excel(annotations_path, engine='openpyxl')
        gdf = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df.Longitud, df.Latitud), crs="EPSG:4326")
        gdf = gdf.to_crs(ortho_crs)
        veg_cols = get_vegetation_columns(df)
        df[veg_cols] = df[veg_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        total_veg = df[veg_cols].sum(axis=1)
        for idx, row in gdf.iterrows():
            p = (row.geometry.x, row.geometry.y)
            if total_veg.iloc[idx] >= 40: veg_points.append(p)
            else: non_veg_points.append(p)

    print("Creating visualization...")
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.set_facecolor('lightblue')
    land_gdf_local.plot(ax=ax, facecolor='white', edgecolor='black', linewidth=0.5)

    with rasterio.open(temp_prediction_tif_path) as pred_src:
        cmap_pred = ListedColormap([(0, 0, 0, 0), (1, 0.5, 0.5, 0.6)])
        
        # === FIX: DOWNSAMPLE HUGE RASTERS FOR DISPLAY ===
        # Target ~3000px max dimension for plotting
        decimation = max(1, max(pred_src.width, pred_src.height) // 3000)
        out_shape = (int(pred_src.height // decimation), int(pred_src.width // decimation))
        
        # Read only what's needed for the plot resolution
        pred_data = pred_src.read(1, out_shape=out_shape, resampling=rasterio.enums.Resampling.nearest)
        ax.imshow(pred_data, cmap=cmap_pred, extent=rasterio.plot.plotting_extent(pred_src))
        # ================================================

    if non_veg_points:
        x, y = zip(*non_veg_points)
        ax.scatter(x, y, c='white', s=50, edgecolors='black', label='Annotation (Non-Veg)')
    if veg_points:
        x, y = zip(*veg_points)
        ax.scatter(x, y, c='darkred', s=50, edgecolors='black', label='Annotation (Veg)')

    if combined_bounds:
        ax.set_xlim(combined_bounds[0], combined_bounds[2])
        ax.set_ylim(combined_bounds[1], combined_bounds[3])

    # (Legend code same as before...)
    ax.legend(loc='upper right', fontsize=12)
    ax.set_title(f"Model Predictions on {ortho_path.name}", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    # Explicit cleanup
    plt.close(fig)
    del fig, ax, land_gdf_local, pred_data
    temp_prediction_tif_path.unlink()
    print(f"Visualization saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Run inference on all orthomosaics in a directory and generate visualizations.")
    parser.add_argument('--ortho-dir', type=Path, required=True, help="Path to the directory containing orthomosaic GeoTIFF files.")
    parser.add_argument('--model-path', type=Path, required=True, help="Path to the trained model checkpoint (.pth file).")
    parser.add_argument('--land-shp-path', type=Path, required=True, help="Path to the land polygon shapefile (e.g., 'map_data/ne_10m_land.shp').")
    parser.add_argument('--roi-path', type=Path, required=True, help="Path to the roi.txt file to filter TIFFs and define visualization bounds.")
    parser.add_argument('--annotations-path', type=Path, default=None, help="(Optional) Path to a master Excel file with ground truth annotations.")
    parser.add_argument('--output-dir', type=Path, required=True, help="Path to the directory to save the output visualization PNG files.")
    parser.add_argument('--batch-size', type=int, default=12, help="Batch size for inference.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model once
    model = Model(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # OPTIMIZATION: Load land shapefile ONCE here
    print("Pre-loading land shapefile...")
    base_land_gdf = geopandas.read_file(args.land_shp_path)

    try:
        roi_poly = load_roi_polygon_wgs84(args.roi_path)
    except Exception as e:
        print(f"FATAL: Could not load ROI: {e}"); return

    all_ortho_files = list(args.ortho_dir.glob("**/*.tif"))
    tiffs_to_process = []
    combined_bounds = None

    print("Checking ROI overlaps...")
    for ortho_path in all_ortho_files:
        if check_roi_overlap(ortho_path, roi_poly):
            tiffs_to_process.append(ortho_path)
            with rasterio.open(ortho_path) as src:
                combined_bounds = update_combined_bounds(combined_bounds, src.bounds)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for i, ortho_path in enumerate(tiffs_to_process):
        print(f"\n--- Processing {i+1}/{len(tiffs_to_process)}: {ortho_path.name} ---")
        predict_and_visualize(
            model, device, ortho_path, args.output_dir / (ortho_path.stem + "_prediction.png"),
            args.annotations_path, args.batch_size, combined_bounds, 
            base_land_gdf # Pass the pre-loaded GDF
        )
        
        # CRITICAL: Force garbage collection after every heavy file iteration
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
