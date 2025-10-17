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
from shapely.geometry import Point, Polygon
from pyproj import Transformer
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import fiona.path

# --- Import your model and helper functions ---
from unetplusplus import Model # Make sure this matches your model file name
from preprocess_data import get_vegetation_columns
import config

class SlidingWindowDataset(Dataset):
    """Dataset to create overlapping patches from a large orthomosaic for inference."""
    def __init__(self, raster_path: Path, patch_size: int, stride: int):
        self.raster_path = raster_path
        self.patch_size = patch_size
        self.stride = stride
        with rasterio.open(self.raster_path) as src:
            self.height = src.height
            self.width = src.width
        self.x_patches = (self.width - self.patch_size) // self.stride + 1
        self.y_patches = (self.height - self.patch_size) // self.stride + 1

    def __len__(self):
        return self.x_patches * self.y_patches

    def __getitem__(self, idx):
        row, col = (idx // self.x_patches), (idx % self.x_patches)
        y_off, x_off = row * self.stride, col * self.stride
        with rasterio.open(self.raster_path) as src:
            window = Window(x_off, y_off, self.patch_size, self.patch_size)
            patch = src.read(window=window, boundless=True, fill_value=0)
        patch_tensor = torch.from_numpy(patch).float() / 255.0
        return {'image': patch_tensor, 'x': x_off, 'y': y_off}

def load_roi_polygon_wgs84(roi_filepath: Path) -> Polygon:
    """Loads ROI points from a text file and returns a Polygon in WGS84 CRS."""
    if not roi_filepath.exists():
        raise FileNotFoundError(f"ROI file not found at: {roi_filepath}")
    points_wgs84 = pd.read_csv(roi_filepath, header=None, names=['lat', 'lon'])
    return Polygon(zip(points_wgs84.lon, points_wgs84.lat))


def predict_and_visualize(model, device, ortho_path, output_path, annotations_path, batch_size, roi_poly_wgs84, land_shp_path):
    """
    Main function to run prediction and visualization for a single orthomosaic.
    """
    print(f"\n--- Processing {ortho_path.name} ---")
    
    patch_size = config.PATCH_WIDTH_PIXELS
    stride = patch_size // 2
    dataset = SlidingWindowDataset(ortho_path, patch_size, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    with rasterio.open(ortho_path) as src:
        ortho_height, ortho_width = src.height, src.width
        ortho_crs = src.crs

    full_prediction_mask = np.zeros((ortho_height, ortho_width), dtype=np.float32)
    weight_mask = np.zeros_like(full_prediction_mask)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Predicting on {ortho_path.name}"):
            images = batch['image'].to(device)
            if images.shape[1] > 3: images = images[:, :3, :, :]
            x_coords, y_coords = batch['x'], batch['y']
            
            # --- BUG FIX ---
            # When model is in .eval() mode, it only returns the final logits
            _, _, _, logits, _ = model(images)
            
            preds = torch.sigmoid(logits).cpu().numpy()
            for i in range(preds.shape[0]):
                y, x = y_coords[i], x_coords[i]
                pred_patch = preds[i, 0, :, :]
                # Ensure we don't write out of bounds
                h, w = pred_patch.shape
                y_end, x_end = min(y + h, ortho_height), min(x + w, ortho_width)
                full_prediction_mask[y:y_end, x:x_end] += pred_patch[:y_end-y, :x_end-x]
                weight_mask[y:y_end, x:x_end] += 1
    
    averaged_prediction_mask = full_prediction_mask / (weight_mask + 1e-6)
    binary_prediction_mask = (averaged_prediction_mask > 0.5).astype(np.uint8)

    # Load annotations (unchanged)
    veg_points, non_veg_points = [], []
    if annotations_path and annotations_path.exists():
        df = pd.read_excel(annotations_path, engine='openpyxl')
        gdf = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df.Longitud, df.Latitud), crs="EPSG:4326")
        gdf = gdf.to_crs(ortho_crs)
        veg_cols = get_vegetation_columns(df)
        df[veg_cols] = df[veg_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        total_veg = df[veg_cols].sum(axis=1)
        for idx, row in gdf.iterrows():
            point = (row.geometry.x, row.geometry.y)
            if total_veg.iloc[idx] >= 40: veg_points.append(point)
            else: non_veg_points.append(point)

    # --- VISUALIZATION REVAMPED ---
    print("Creating visualization...")
    fig, ax = plt.subplots(figsize=(20, 20))
    
    # 1. Set background to blue for the sea
    ax.set_facecolor('lightblue')

    # 2. Load and plot land polygons
    land_gdf = geopandas.read_file(land_shp_path)
    # Reproject land data to match the orthomosaic's coordinate system
    land_gdf = land_gdf.to_crs(ortho_crs)
    land_gdf.plot(ax=ax, facecolor='white', edgecolor='black', linewidth=0.5)

    # 3. Overlay the model's predictions
    with rasterio.open(ortho_path) as src:
        # Create a colormap: 0 is transparent, 1 is light red with 60% opacity
        cmap_pred = ListedColormap([(0, 0, 0, 0), (1, 0.5, 0.5, 0.6)])
        # Use rasterio's helper to get the correct plot extent
        ax.imshow(binary_prediction_mask, cmap=cmap_pred, extent=rasterio.plot.plotting_extent(src))

    # 4. Plot annotation points on top
    if non_veg_points:
        x, y = zip(*non_veg_points); ax.scatter(x, y, c='white', s=50, edgecolors='black', label='Annotation (Non-Veg)')
    if veg_points:
        x, y = zip(*veg_points); ax.scatter(x, y, c='darkred', s=50, edgecolors='black', label='Annotation (Veg)')

    # 5. Apply ROI bounds to the plot
    if roi_poly_wgs84:
        try:
            transformer = Transformer.from_crs("EPSG:4326", ortho_crs, always_xy=True)
            roi_poly_transformed = Polygon([transformer.transform(x, y) for x, y in roi_poly_wgs84.exterior.coords])
            minx, miny, maxx, maxy = roi_poly_transformed.bounds
            ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy)
        except Exception as e:
            print(f"Warning: Could not apply ROI bounds. Error: {e}")
    
    # 6. Create a new, descriptive legend
    legend_elements = [
        Patch(facecolor='white', edgecolor='black', label='Land'),
        Patch(facecolor='lightblue', edgecolor='black', label='Sea'),
        Patch(facecolor=(1, 0.5, 0.5, 0.6), label='Predicted Vegetation'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)

    ax.set_title(f"Model Predictions on {ortho_path.name}", fontsize=16)
    ax.set_xlabel("Easting"); ax.set_ylabel("Northing")
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run inference on all orthomosaics in a directory and generate visualizations.")
    parser.add_argument('--ortho-dir', type=Path, required=True, help="Path to the directory containing orthomosaic GeoTIFF files.")
    parser.add_argument('--model-path', type=Path, required=True, help="Path to the trained model checkpoint (.pth file).")
    # --- NEW REQUIRED ARGUMENT ---
    parser.add_argument('--land-shp-path', type=Path, required=True, help="Path to the land polygon shapefile (e.g., 'map_data/ne_10m_land.shp').")
    parser.add_argument('--roi-path', type=Path, default=None, help="(Optional) Path to the roi.txt file to clip the visualization.")
    parser.add_argument('--annotations-path', type=Path, default=None, help="(Optional) Path to a master Excel file with ground truth annotations.")
    parser.add_argument('--output-dir', type=Path, required=True, help="Path to the directory to save the output visualization PNG files.")
    parser.add_argument('--batch-size', type=int, default=8, help="Batch size for inference.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading model from {args.model_path}")
    # --- BUG FIX ---
    # Ensure you are importing the correct model class name
    model = Model(in_channels=3, out_channels=1).to(device) 
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    roi_poly = None
    if args.roi_path:
        try:
            roi_poly = load_roi_polygon_wgs84(args.roi_path)
            print(f"Successfully loaded ROI from {args.roi_path}")
        except Exception as e:
            print(f"Error loading ROI file: {e}. Proceeding without ROI clipping.")

    ortho_files = list(args.ortho_dir.glob("**/*.tif"))
    if not ortho_files:
        print(f"Error: No .tif files found in {args.ortho_dir}. Please check the path.")
        return
    print(f"Found {len(ortho_files)} orthomosaics to process.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for ortho_path in ortho_files:
        output_path = args.output_dir / (ortho_path.stem + "_prediction.png")
        predict_and_visualize(
            model=model,
            device=device,
            ortho_path=ortho_path,
            output_path=output_path,
            annotations_path=args.annotations_path,
            batch_size=args.batch_size,
            roi_poly_wgs84=roi_poly,
            land_shp_path=args.land_shp_path # Pass the new argument
        )

if __name__ == '__main__':
    main()
