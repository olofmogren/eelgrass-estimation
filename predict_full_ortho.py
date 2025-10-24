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
from datetime import datetime # Added for timestamped filenames
import gc # Added for garbage collection
import random
import rasterio.features
import shapely.geometry
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# --- Import your model and helper functions ---
from unetplusplus import Model
# from preprocess_data import get_vegetation_columns
import config

# Set matplotlib backend to non-interactive to save memory on servers
import matplotlib
matplotlib.use('Agg')


def get_vegetation_columns(df: pd.DataFrame) -> list:
    """Identifies the correct vegetation column names from the dataframe."""
    try:
        start_index = df.columns.to_list().index('Ste')
        excluded_cols = {'Dens_veg', 'Mat_Pos', 'Comments', 'Kontroll%', 'geometry'}
        potential_cols = df.columns[start_index:].to_list()
        return [c for c in potential_cols if c not in excluded_cols and 'Unnamed' not in str(c)]
    except ValueError:
        print("Warning: Could not find 'Ste' column to identify vegetation columns.")
        return []

def load_all_annotations(all_xlsx_paths: list) -> geopandas.GeoDataFrame:
    """Scans all XLSX files, intelligently finds the header row, and loads all valid annotations."""
    all_dfs = []
    # Use a generic WGS84 CRS string that doesn't rely on a config file
    wgs84_crs = "EPSG:4326"

    for xlsx_path in tqdm(all_xlsx_paths, desc="Loading all annotations"):
        try:
            df = pd.read_excel(xlsx_path, engine='openpyxl')
            if 'Latitud' not in df.columns or 'Longitud' not in df.columns:
                found_header = False
                # Limit the search to avoid reading huge files without headers
                df_no_header = pd.read_excel(xlsx_path, engine='openpyxl', header=None, nrows=20)
                for i, row in df_no_header.iterrows():
                    if 'Latitud' in row.values and 'Longitud' in row.values:
                        df = pd.read_excel(xlsx_path, engine='openpyxl', header=i)
                        found_header = True; break
                if not found_header:
                    print(f"  - Skipping {xlsx_path.name} (no Lat/Lon columns found)")
                    continue

            df.dropna(subset=['Latitud', 'Longitud'], inplace=True)
            df = df[pd.to_numeric(df['Latitud'], errors='coerce').notna()]
            df = df[pd.to_numeric(df['Longitud'], errors='coerce').notna()]
            if not df.empty: all_dfs.append(df)
        except Exception as e:
            print(f"\nWarning: Could not process file {xlsx_path.name}. Reason: {e}"); continue

    if not all_dfs: return geopandas.GeoDataFrame()

    master_df = pd.concat(all_dfs, ignore_index=True)
    if 'geometry' in master_df.columns:
        master_df = master_df.drop(columns=['geometry'])

    return geopandas.GeoDataFrame(master_df, geometry=geopandas.points_from_xy(master_df.Longitud, master_df.Latitud), crs=wgs84_crs)


def load_roi_polygon_wgs84(roi_filepath: Path) -> Polygon:
    """Loads ROI points from a text file and returns a Polygon in WGS84 CRS."""
    if not roi_filepath.exists():
        raise FileNotFoundError(f"ROI file not found at: {roi_filepath}")
    points_wgs84 = pd.read_csv(roi_filepath, header=None, names=['lat', 'lon'])
    return Polygon(zip(points_wgs84.lon, points_wgs84.lat))

def check_roi_overlap(tif_path: Path, roi_poly_wgs84: Polygon) -> bool:
    """Checks if a GeoTIFF's bounds intersect with the given ROI polygon."""
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
    """Expands the current bounding box to include the new bounds."""
    if current_bounds is None:
        return new_bounds

    curr_min_x, curr_min_y, curr_max_x, curr_max_y = current_bounds
    new_min_x, new_min_y, new_max_x, new_max_y = new_bounds

    min_x = min(curr_min_x, new_min_x)
    min_y = min(curr_min_y, new_min_y)
    max_x = max(curr_max_x, new_max_x)
    max_y = max(curr_max_y, new_max_y)

    return (min_x, min_y, max_x, max_y)

class SlidingWindowDataset(Dataset):
    """Dataset with OPTIMIZED file handling for inference."""
    def __init__(self, raster_path: Path, patch_size: int, stride: int):
        self.raster_path = raster_path # Store path, not file handle
        self.patch_size = patch_size
        self.stride = stride

        with rasterio.open(self.raster_path) as src:
            self.height = src.height
            self.width = src.width

        self.x_patches = max(1, (self.width - self.patch_size + self.stride - 1) // self.stride + 1)
        self.y_patches = max(1, (self.height - self.patch_size + self.stride - 1) // self.stride + 1)
        self.src = None # This will hold the file object for each worker process

    def __len__(self):
        return self.x_patches * self.y_patches

    def __getitem__(self, idx):
        if self.src is None: # Each worker process opens the file once
            self.src = rasterio.open(self.raster_path)
        try:
            row, col = (idx // self.x_patches), (idx % self.x_patches)
            y_off, x_off = row * self.stride, col * self.stride

            window = Window(x_off, y_off, self.patch_size, self.patch_size)
            patch = self.src.read(window=window, boundless=True, fill_value=0)

            if patch.shape[0] > 3:
                patch = patch[:3, :, :]

            #patch_tensor = torch.from_numpy(patch).float() / 255.0
            patch_tensor = torch.from_numpy(patch).float()
            return {'image': patch_tensor, 'x': x_off, 'y': y_off}

        except rasterio.errors.RasterioIOError as e:
            print(f"\nWARNING: Skipping corrupt patch in {self.raster_path.name} at index {idx}. Error: {e}")
            return None

def collate_fn_skip_corrupt(batch):
    """A custom collate_fn that filters out None values from a batch."""
    batch = [item for item in batch if item is not None]
    if not batch:
        return {} # Return an empty dict that the loop can check for
    return torch.utils.data.dataloader.default_collate(batch)

def load_split_annotations(preprocessed_dir: Path) -> dict:
    """Loads annotations from train/val/test JSON files into a dictionary."""
    annotations = {'train': [], 'val': [], 'test': []}
    for split in annotations.keys():
        json_path = preprocessed_dir / split / "annotations.json"
        if json_path.exists():
            print(f"  -> Loading annotations from {json_path}")
            with open(json_path, 'r') as f:
                data = json.load(f)
                # Flatten the data into a simple list of points with labels
                for patch_info in data:
                    for ann in patch_info['annotations']:
                        annotations[split].append({
                            'geometry': Point(ann['x'], ann['y']),
                            'label': ann['label']
                        })
    return annotations

def predict_and_visualize(model, device, ortho_path, output_path, master_annotations_gdf, batch_size, land_gdf):
    """
    Creates a visualization with the ortho background, overlays positive predictions
    in the sea as a red diagonal stripe pattern, and marks the coastline.
    """
    # --- PHASE 1 (Prediction) remains exactly the same ---
    print(f"\n--- Analyzing {ortho_path.name} ---")
    PATCH_WIDTH_PIXELS = 512 # Placeholder
    patch_size = PATCH_WIDTH_PIXELS
    stride = int(patch_size * 0.95)
    dataset = SlidingWindowDataset(ortho_path, patch_size, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn_skip_corrupt)
    prediction_geotiff_path = output_path.with_suffix(".tif")

    with rasterio.open(ortho_path) as src:
        ortho_crs = src.crs
        ortho_height, ortho_width = src.height, src.width
        ortho_bounds = src.bounds
        meta = src.meta.copy()
        meta.update(driver='GTiff', count=1, dtype='uint8', compress='lzw', nodata=None)
        with rasterio.open(prediction_geotiff_path, 'w', **meta) as dst:
            with torch.inference_mode():
                for batch in tqdm(dataloader, desc=f"Predicting on {ortho_path.name}"):
                    if not batch: continue
                    images = batch['image'].to(device).float()
                    x_coords, y_coords = batch['x'], batch['y']
                    _, _, _, logits, _ = model(images)
                    preds = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(np.uint8)
                    for i in range(preds.shape[0]):
                        y, x = y_coords[i].item(), x_coords[i].item()
                        pred_patch = preds[i, :, :, :]
                        write_h = min(pred_patch.shape[1], ortho_height - y)
                        write_w = min(pred_patch.shape[2], ortho_width - x)
                        write_window = Window(x, y, write_w, write_h)
                        data_to_write = pred_patch[:, :write_h, :write_w]
                        dst.write(data_to_write, window=write_window)

    # --- PHASE 2: VISUALIZATION (New Logic) ---
    print("Creating visualization...")

    ortho_poly = shapely.geometry.box(*ortho_bounds)
    split_points = {'train': [], 'val': [], 'test': []}
    
    for split, points in all_annotations.items():
        if not points: continue
        # Create a GeoDataFrame for efficient filtering
        gdf = geopandas.GeoDataFrame(points, crs="EPSG:3006").to_crs(ortho_crs) # Assuming original CRS is SWEREF99 TM
        gdf_filtered = gdf[gdf.geometry.intersects(ortho_poly)]
        if not gdf_filtered.empty:
            split_points[split] = [(row.geometry.x, row.geometry.y, row.label) for _, row in gdf_filtered.iterrows()]

    fig, ax = plt.subplots(figsize=(20, 20))
    
    # 1. Draw the original Orthomosaic as the background (zorder=1)
    with rasterio.open(ortho_path) as ortho_src:
        max_dim = max(ortho_src.width, ortho_src.height)
        decimation = max(1, max_dim // 4000)
        out_shape = (int(ortho_src.height // decimation), int(ortho_src.width // decimation))
        ortho_img = ortho_src.read((1, 2, 3), out_shape=out_shape, resampling=rasterio.enums.Resampling.bilinear)
        ortho_img_plot = np.transpose(ortho_img, (1, 2, 0))
        ax.imshow(ortho_img_plot, extent=rasterio.plot.plotting_extent(ortho_src), zorder=1)

    # 2. Prepare Prediction Data and Land Mask
    with rasterio.open(prediction_geotiff_path) as pred_src:
        # Get the transform for the downsampled grid
        downsampled_transform = pred_src.transform * pred_src.transform.scale(
            (pred_src.width / out_shape[1]),
            (pred_src.height / out_shape[0])
        )
        
        # Read downsampled prediction data
        averaged_data = pred_src.read(1, out_shape=out_shape, resampling=rasterio.enums.Resampling.average)
        pred_data = (averaged_data > 0).astype(np.uint8)

        # Create a land mask by rasterizing the land polygons onto the same grid
        land_gdf_local = land_gdf.to_crs(ortho_crs)
        land_mask = rasterio.features.rasterize(
            land_gdf_local.geometry,
            out_shape=out_shape,
            transform=downsampled_transform,
            fill=0,
            default_value=1,
            dtype=np.uint8
        )
        
        # --- Apply the mask: Set predictions on land to 0 (transparent) ---
        pred_data[land_mask == 1] = 0

        # 3. Convert masked predictions to vector polygons and plot with a pattern
        # This finds contiguous areas of pixels with value 1
        shapes = rasterio.features.shapes(pred_data, transform=downsampled_transform)
        
        # Create a list of shapely Polygons
        prediction_polygons = [shapely.geometry.shape(geom) for geom, val in shapes if val == 1]

        if prediction_polygons:
            # Create a GeoDataFrame from the polygons
            gdf_preds = geopandas.GeoDataFrame(geometry=prediction_polygons, crs=ortho_crs)
            
            # Plot the polygons with a red diagonal hatch pattern
            gdf_preds.plot(ax=ax, facecolor='none', hatch='///', edgecolor='red', linewidth=0, zorder=2)
            
    # 4. Draw the Land/Sea Border on top of everything (zorder=3)
    land_gdf_local.plot(ax=ax, facecolor='none', edgecolor='yellow', linewidth=1.5, zorder=3)
    
    # 5. Draw Differentiated Annotations ---
    # Draw TEST points (opaque)
    if split_points['test']:
        veg = [(x, y) for x, y, label in split_points['test'] if label == 1]
        non_veg = [(x, y) for x, y, label in split_points['test'] if label == 0]
        if non_veg: ax.scatter(*zip(*non_veg), c='white', s=60, edgecolors='black', zorder=5, marker='s')
        if veg: ax.scatter(*zip(*veg), c='darkred', s=60, edgecolors='black', zorder=5, marker='s')

    # Draw VAL points (opaque)
    if split_points['val']:
        veg = [(x, y) for x, y, label in split_points['val'] if label == 1]
        non_veg = [(x, y) for x, y, label in split_points['val'] if label == 0]
        if non_veg: ax.scatter(*zip(*non_veg), c='white', s=50, edgecolors='black', zorder=4, marker='^')
        if veg: ax.scatter(*zip(*veg), c='darkred', s=50, edgecolors='black', zorder=4, marker='^')

    # Draw TRAIN points (semi-transparent)
    if split_points['train']:
        veg = [(x, y) for x, y, label in split_points['train'] if label == 1]
        non_veg = [(x, y) for x, y, label in split_points['train'] if label == 0]
        if non_veg: ax.scatter(*zip(*non_veg), c='white', s=40, edgecolors='black', zorder=4, alpha=0.5)
        if veg: ax.scatter(*zip(*veg), c='darkred', s=40, edgecolors='black', zorder=4, alpha=0.5)

    # Set plot limits
    minx, miny, maxx, maxy = ortho_bounds
    ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy)

    legend_elements = [
        Patch(facecolor='none', edgecolor='red', hatch='///', label='Predicted Vegetation'),
        Line2D([0], [0], color='yellow', lw=2, label='Coastline'),
        Line2D([0], [0], marker='s', ls='none', c='w', mec='k', mew=1, markersize=10, label='Test Point'),
        Line2D([0], [0], marker='^', ls='none', c='w', mec='k', mew=1, markersize=10, label='Validation Point'),
        Line2D([0], [0], marker='o', ls='none', c='w', mec='k', mew=1, markersize=10, alpha=0.5, label='Training Point')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
    ax.set_title(f"Model Predictions on {ortho_path.name}", fontsize=16)
    ax.set_xlabel("Easting"); ax.set_ylabel("Northing")
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Prediction GeoTIFF saved to {prediction_geotiff_path}")
    print(f"Visualization saved to {output_path}") 
    
def main():
    parser = argparse.ArgumentParser(description="Run inference on all orthomosaics in a directory and generate visualizations.")
    parser.add_argument('--ortho-dir', type=Path, default='data', help="Path to the directory containing orthomosaic GeoTIFF files.")
    parser.add_argument('--model-path', type=Path, required=True, help="Path to the trained model checkpoint (.pth file).")
    parser.add_argument('--land-shp-path', type=Path, default='data/shapefiles/Europe_coastline.shp', help="Path to the land polygon shapefile.")
    parser.add_argument('--roi-path', type=Path, default='data/roi.txt', help="Path to the roi.txt file.")
    parser.add_argument('--output-dir', type=Path, default='full-predictions', help="Path to the directory to save the output files.")
    parser.add_argument('--batch-size', type=int, default=12, help="Batch size for inference.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model once
    model = Model(in_channels=3, out_channels=1).to(device)
    print("Loading model from "+str(args.model_path))
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    
    # --- NEW: Create a timestamped subdirectory for this run's outputs ---
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_output_dir = args.output_dir / timestamp
    run_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutputs for this run will be saved in: {run_output_dir}")

    # --- NEW: Load all annotations from the preprocessed data directory ---
    print("\nLoading annotations from preprocessed data splits...")
    all_annotations = load_split_annotations(config.PREPROCESSED_DATA_DIR)

    try:
        roi_poly = load_roi_polygon_wgs84(args.roi_path)
        print(f"Successfully loaded ROI from {args.roi_path}")
    except Exception as e:
        print(f"FATAL: Could not load ROI file: {e}. Exiting.")
        return

    print("Pre-loading land shapefile...")
    land_gdf = geopandas.read_file(args.land_shp_path)

    all_ortho_files = list(args.ortho_dir.glob("**/*.tif"))
    if not all_ortho_files:
        print(f"Error: No .tif files found in {args.ortho_dir}. Please check the path.")
        return
    print(f"Found {len(all_ortho_files)} total orthomosaics. Checking for ROI overlap...")

    tiffs_to_process = []
    for ortho_path in all_ortho_files:
        if check_roi_overlap(ortho_path, roi_poly):
            tiffs_to_process.append(ortho_path)
        else:
            print(f"  - Skipping {ortho_path.name} (no overlap with ROI)")

    if not tiffs_to_process:
        print("\nNo orthomosaics overlap with the ROI. Nothing to process.")
        return

    print(f"\nProceeding to process {len(tiffs_to_process)} overlapping orthomosaics.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    num_to_process = len(tiffs_to_process)
    random.shuffle(tiffs_to_process)
    for i, ortho_path in enumerate(tiffs_to_process):
        print(f"\n--- Processing file {i+1} of {num_to_process}: {ortho_path.name} ---")

        output_filename_base = ortho_path.stem
        output_path = run_output_dir / f"{output_filename_base}.png"

        predict_and_visualize(
            model=model,
            device=device,
            ortho_path=ortho_path,
            output_path=output_path,
            all_annotations=all_annotations, # Pass the dictionary of all annotations
            batch_size=args.batch_size,
            land_gdf=land_gdf
        )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
            
if __name__ == '__main__':
    # NOTE: You will need to create placeholder files for the model and config
    # for this script to run as-is.
    # Example:
    # class Model(torch.nn.Module):
    #     def __init__(self, in_channels=3, out_channels=1):
    #         super().__init__()
    #         self.conv = torch.nn.Conv2d(in_channels, out_channels, 1)
    #     def forward(self, x):
    #         return None, None, None, self.conv(x), None
    main()
