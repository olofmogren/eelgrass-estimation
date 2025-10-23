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

# --- Import your model and helper functions ---
from unetplusplus import Model
#from preprocess_data import get_vegetation_columns
import config

# Set matplotlib backend to non-interactive to save memory on servers
import matplotlib
matplotlib.use('Agg')

# --- HELPER FUNCTIONS FOR ROI AND BOUNDS ---

# could have been imported from preprocess_data:
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

def predict_and_visualize(model, device, ortho_path, output_path, master_annotations_gdf, batch_size, land_gdf):
    # --- PHASE 1 (Prediction) is unchanged ---
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
                    images = batch['image'].to(device)
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

    # --- PHASE 2: VISUALIZATION ---
    print("Creating visualization...")
    
    # Load and filter annotations from the master DataFrame
    veg_points, non_veg_points = [], []
    if not master_annotations_gdf.empty:
        # Reproject all annotations to the ortho's CRS
        gdf = master_annotations_gdf.to_crs(ortho_crs)
        
        # Spatially filter for points within the current ortho's bounds
        ortho_poly = shapely.geometry.box(*ortho_bounds)
        gdf_filtered = gdf[gdf.geometry.intersects(ortho_poly)].copy() # Use .copy() to avoid SettingWithCopyWarning
        
        if not gdf_filtered.empty:
            print(f"  -> Found {len(gdf_filtered)} annotations for this ortho.")
            # Use the robust function to find vegetation columns
            veg_cols = get_vegetation_columns(gdf_filtered)
            
            if veg_cols:
                gdf_filtered[veg_cols] = gdf_filtered[veg_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
                total_veg = gdf_filtered[veg_cols].sum(axis=1)
                for idx, row in gdf_filtered.iterrows():
                    point = (row.geometry.x, row.geometry.y)
                    if total_veg.loc[idx] >= 40:
                        veg_points.append(point)
                    else:
                        non_veg_points.append(point)

    fig, ax = plt.subplots(figsize=(20, 20))
    
    # 1. Draw the original Orthomosaic as the background (zorder=1)
    # We must downsample it to prevent memory crashes, just like the prediction raster.
    with rasterio.open(ortho_path) as ortho_src:
        max_dim = max(ortho_src.width, ortho_src.height)
        decimation = max(1, max_dim // 4000)
        out_shape = (int(ortho_src.height // decimation), int(ortho_src.width // decimation))
        
        # Read the first 3 bands (RGB)
        ortho_img = ortho_src.read((1, 2, 3), out_shape=out_shape, resampling=rasterio.enums.Resampling.bilinear)
        
        # Transpose the array from (channels, height, width) to (height, width, channels) for imshow
        ortho_img_plot = np.transpose(ortho_img, (1, 2, 0))
        
        # Any pixel where all 3 bands are 0 is considered NoData.
        # valid_data_mask = (ortho_img.sum(axis=0) > 0).astype(np.uint8)
        valid_data_mask = ortho_src.read_masks(1, out_shape=out_shape, resampling=rasterio.enums.Resampling.bilinear)
        
        ax.imshow(ortho_img_plot, extent=rasterio.plot.plotting_extent(ortho_src), zorder=1)

    # 2. Draw the Prediction Layer on top (zorder=2)
    with rasterio.open(prediction_geotiff_path) as pred_src:
        # Use the same downsampling shape as the ortho for perfect alignment
        averaged_data = pred_src.read(1, out_shape=out_shape, resampling=rasterio.enums.Resampling.average)
        pred_data = (averaged_data > 0).astype(np.uint8)
        
        # The colormap for '0' is transparent, for '1' it's semi-transparent red
        cmap_pred = ListedColormap([(0, 0, 0, 0), (1, 0.5, 0.5, 0.6)])
        
        ax.imshow(pred_data, cmap=cmap_pred, extent=rasterio.plot.plotting_extent(pred_src), zorder=2, vmin=0, vmax=1)
        
        # --- NEW: Apply both masks ---
        pred_data[land_mask == 1] = 0        # Erase predictions on land
        pred_data[valid_data_mask == 0] = 0  # Erase predictions in NoData areas


        # 3. Convert final masked predictions to polygons and plot
        shapes = rasterio.features.shapes(pred_data, transform=downsampled_transform)
        prediction_polygons = [shapely.geometry.shape(geom) for geom, val in shapes if val == 1]

        if prediction_polygons:
            gdf_preds = geopandas.GeoDataFrame(geometry=prediction_polygons, crs=ortho_crs)
            gdf_preds.plot(ax=ax, facecolor='none', hatch='///', edgecolor='red', linewidth=0.5, zorder=2)
            
    # 4. Draw the Land/Sea Border on top of the imagery (zorder=3)
    land_gdf_local = land_gdf.to_crs(ortho_crs)
    # Set 'facecolor' to 'none' to only draw the outline
    land_gdf_local.plot(ax=ax, facecolor='none', edgecolor='yellow', linewidth=1.5, zorder=3)



    if non_veg_points:
        x, y = zip(*non_veg_points)
        ax.scatter(x, y, c='white', s=50, edgecolors='black', label='Annotation (Non-Veg)', zorder=4)
    if veg_points:
        x, y = zip(*veg_points)
        ax.scatter(x, y, c='darkred', s=50, edgecolors='black', label='Annotation (Veg)', zorder=4)
        
    print('veg points', len(veg_points), ', non veg points', len(non_veg_points))

    minx, miny, maxx, maxy = ortho_bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)

    # (Legend code same as before...)
    ax.legend(loc='upper right', fontsize=12)
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
    parser.add_argument('--ortho-dir', type=Path, required=True, help="Path to the directory containing orthomosaic GeoTIFF files.")
    parser.add_argument('--model-path', type=Path, required=True, help="Path to the trained model checkpoint (.pth file).")
    parser.add_argument('--land-shp-path', type=Path, required=True, help="Path to the land polygon shapefile.")
    parser.add_argument('--roi-path', type=Path, required=True, help="Path to the roi.txt file.")
    parser.add_argument('--output-dir', type=Path, required=True, help="Path to the directory to save the output files.")
    parser.add_argument('--batch-size', type=int, default=12, help="Batch size for inference.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model once
    model = Model(in_channels=3, out_channels=1).to(device)
    print("Loading model from "+str(args.model_path))
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    print("\nSearching for annotation files...")
    all_xlsx_paths = list(args.ortho_dir.glob("**/*.xlsx"))
    if not all_xlsx_paths:
        print("  -> No .xlsx annotation files found. Continuing without annotations.")
        master_annotations_gdf = geopandas.GeoDataFrame()
    else:
        print(f"  -> Found {len(all_xlsx_paths)} annotation files. Loading...")
        master_annotations_gdf = load_all_annotations(all_xlsx_paths)
        print(f"  -> Successfully loaded a total of {len(master_annotations_gdf)} annotations.")

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

        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S-%f")
        output_path = args.output_dir / f"{timestamp}_prediction.png"

        predict_and_visualize(
            model=model,
            device=device,
            ortho_path=ortho_path,
            output_path=output_path,
            master_annotations_gdf=master_annotations_gdf, # Pass the pre-loaded DataFrame
            batch_size=args.batch_size,
            land_gdf=land_gdf  # Pass the pre-loaded GeoDataFrame
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
