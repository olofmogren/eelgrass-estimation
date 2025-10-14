import os
import glob
import random
from pathlib import Path
import argparse
import h5py
import math
import numpy as np
import pandas as pd
import geopandas
from shapely.geometry import Polygon, box
import rasterio
from rasterio.windows import from_bounds
from pyproj import Transformer, CRS
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- Configuration ---
BASE_DATA_DIR = Path("data")
PREPROCESSED_DATA_DIR = Path("preprocessed_data")
VISUALIZATION_DIR = PREPROCESSED_DATA_DIR / "visualizations"
ROI_FILE_NAME = "roi.txt"
WGS84_CRS = "EPSG:4326"
RANDOM_SEED = 42

PATCH_WIDTH_PIXELS = 256
PATCH_HEIGHT_PIXELS = 256
NUM_VISUALIZATIONS_PER_SPLIT = 10

def get_vegetation_columns(df: pd.DataFrame) -> list:
    """
    Identifies the correct vegetation column names from the dataframe.
    This is now safer and explicitly excludes non-vegetation columns.
    """
    try:
        # The vegetation data is assumed to start at the 'Ste' column.
        start_index = df.columns.to_list().index('Ste')

        # Define columns that appear after 'Ste' but are not vegetation percentages.
        # CRITICAL: Also exclude the 'geometry' column that GeoPandas uses.
        excluded_cols = {'Dens_veg', 'Mat_Pos', 'Comments', 'Kontroll%', 'geometry'}

        potential_cols = df.columns[start_index:].to_list()

        # Filter the list to get only the valid vegetation columns.
        veg_cols = [
            col for col in potential_cols
            if col not in excluded_cols and 'Unnamed' not in str(col)
        ]
        return veg_cols
    except ValueError:
        # This error occurs if the 'Ste' column is not found in the dataframe.
        return []


def load_all_annotations(all_xlsx_paths: list) -> geopandas.GeoDataFrame:
    """
    Scans all XLSX files, intelligently finds the header row, loads all valid annotations,
    and returns a single master GeoDataFrame.
    """
    all_dfs = []
    for xlsx_path in tqdm(all_xlsx_paths, desc="Loading all annotations"):
        try:
            df = pd.read_excel(xlsx_path, engine='openpyxl')

            if 'Latitud' not in df.columns or 'Longitud' not in df.columns:
                found_header = False
                df_no_header = pd.read_excel(xlsx_path, engine='openpyxl', header=None, nrows=20)
                for i, row in df_no_header.iterrows():
                    if 'Latitud' in row.values and 'Longitud' in row.values:
                        df = pd.read_excel(xlsx_path, engine='openpyxl', header=i)
                        found_header = True
                        break
                if not found_header:
                    continue

            if 'Latitud' not in df.columns or 'Longitud' not in df.columns:
                 continue

            df.dropna(subset=['Latitud', 'Longitud'], inplace=True)
            df = df[pd.to_numeric(df['Latitud'], errors='coerce').notna()]
            df = df[pd.to_numeric(df['Longitud'], errors='coerce').notna()]

            if not df.empty:
                all_dfs.append(df)

        except Exception as e:
            print(f"\nWarning: Could not process file {xlsx_path.name}. Reason: {e}")
            continue

    if not all_dfs:
        return geopandas.GeoDataFrame()

    master_df = pd.concat(all_dfs, ignore_index=True)
    if 'geometry' in master_df.columns:
        master_df = master_df.drop(columns=['geometry'])

    master_gdf = geopandas.GeoDataFrame(
        master_df,
        geometry=geopandas.points_from_xy(master_df.Longitud, master_df.Latitud),
        crs=WGS84_CRS
    )
    return master_gdf


def load_roi_polygon_wgs84(data_dir: Path) -> Polygon:
    """Loads ROI points from roi.txt and returns a Polygon in WGS84 CRS."""
    roi_filepath = data_dir / ROI_FILE_NAME
    if not roi_filepath.exists():
        raise FileNotFoundError(f"{ROI_FILE_NAME} not found in {data_dir}. Cannot proceed.")
    points_wgs84 = pd.read_csv(roi_filepath, header=None, names=['lat', 'lon'])
    return Polygon(zip(points_wgs84.lon, points_wgs84.lat))

def get_roi_overlapping_geotiffs(geotiff_paths: list, roi_poly_wgs84: Polygon) -> list:
    """Filters a list of GeoTIFF paths to include only those whose bounds intersect the ROI."""
    overlapping_files = []
    for tif_path in tqdm(geotiff_paths, desc="Checking for ROI overlap"):
        try:
            with rasterio.open(tif_path) as src:
                raster_bounds_poly = box(*src.bounds)
                transformer = Transformer.from_crs(WGS84_CRS, src.crs, always_xy=True)
                transformed_roi_poly = Polygon([transformer.transform(x, y) for x, y in roi_poly_wgs84.exterior.coords])
                if raster_bounds_poly.intersects(transformed_roi_poly):
                    overlapping_files.append(tif_path)
        except Exception as e:
            print(f"Warning: Could not process {tif_path.name} for ROI check. Error: {e}")
    return overlapping_files

def process_geotiff_and_annotations(tif_path: Path, master_annotations_gdf: geopandas.GeoDataFrame, roi_polygon_wgs84: Polygon, output_dir: Path, generated_patches: list):
    """Processes a GeoTIFF by spatially querying the master annotation database."""
    try:
        with rasterio.open(tif_path) as src:
            image_crs = src.crs
            image_bounds_poly = box(*src.bounds)

            annotations_in_img_crs = master_annotations_gdf.to_crs(image_crs)
            annotations_in_image = annotations_in_img_crs[annotations_in_img_crs.geometry.intersects(image_bounds_poly)].copy()

            if annotations_in_image.empty:
                return

            transformer = Transformer.from_crs(WGS84_CRS, image_crs, always_xy=True)
            roi_poly_image_crs = Polygon([transformer.transform(x, y) for x, y in roi_polygon_wgs84.exterior.coords])
            annotations_in_roi = annotations_in_image[annotations_in_image.geometry.within(roi_poly_image_crs)].copy()

            if annotations_in_roi.empty:
                return

            veg_cols = get_vegetation_columns(annotations_in_roi)
            if not veg_cols:
                return

            annotations_in_roi[veg_cols] = annotations_in_roi[veg_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

            print(f"  -> SUCCESS: Found {len(annotations_in_roi)} valid annotations for {tif_path.name}. Generating patches...")

            patch_count = 0
            for _, annotation in annotations_in_roi.iterrows():
                patch_count += 1

                px, py = annotation.geometry.x, annotation.geometry.y
                pixel_width_m, pixel_height_m = src.transform.a, -src.transform.e
                patch_width_m, patch_height_m = PATCH_WIDTH_PIXELS*pixel_width_m, PATCH_HEIGHT_PIXELS*pixel_height_m

                offset_x, offset_y = random.uniform(-patch_width_m/2, patch_width_m/2), random.uniform(-patch_height_m/2, patch_height_m/2)
                center_x, center_y = px - offset_x, py - offset_y
                minx, maxx, miny, maxy = center_x-patch_width_m/2, center_x+patch_width_m/2, center_y-patch_height_m/2, center_y+patch_height_m/2

                try:
                    patch_window = from_bounds(minx, miny, maxx, maxy, src.transform)
                    source_patch = src.read(window=patch_window)
                    _, actual_height, actual_width = source_patch.shape
                    if actual_height != PATCH_HEIGHT_PIXELS or actual_width != PATCH_WIDTH_PIXELS:
                        padded_patch = np.zeros((src.count, PATCH_HEIGHT_PIXELS, PATCH_WIDTH_PIXELS), dtype=source_patch.dtype)
                        padded_patch[:, :actual_height, :actual_width] = source_patch
                        source_patch = padded_patch
                except Exception: continue

                patch_bounds_poly = box(minx, miny, maxx, maxy)
                annotations_in_patch = annotations_in_roi[annotations_in_roi.geometry.within(patch_bounds_poly)]
                if annotations_in_patch.empty: continue

                target_binary = np.zeros((PATCH_HEIGHT_PIXELS, PATCH_WIDTH_PIXELS), dtype=np.uint8)
                mask_binary = np.zeros_like(target_binary)

                for _, ann_in_patch in annotations_in_patch.iterrows():
                    col = int((ann_in_patch.geometry.x - minx) / pixel_width_m)
                    row = int((maxy - ann_in_patch.geometry.y) / pixel_height_m)
                    if not (0 <= row < PATCH_HEIGHT_PIXELS and 0 <= col < PATCH_WIDTH_PIXELS): continue

                    if ann_in_patch[veg_cols].sum() >= 40:
                        target_binary[row, col] = 1
                    mask_binary[row, col] = 1

                base_name = f"{tif_path.stem}_patch_{patch_count:04d}"
                output_dir.mkdir(parents=True, exist_ok=True)
                with h5py.File(output_dir / f"{base_name}_source.h5", "w") as hf: hf.create_dataset("image", data=source_patch)
                # --- THIS LINE IS NOW FIXED ---
                with h5py.File(output_dir / f"{base_name}_target_binary.h5", "w") as hf: hf.create_dataset("target", data=target_binary)
                with h5py.File(output_dir / f"{base_name}_mask_binary.h5", "w") as hf: hf.create_dataset("mask", data=mask_binary)

                generated_patches.append({'base_name': base_name, 'output_dir': output_dir})

    except Exception as e:
        print(f"FATAL: Failed to process GeoTIFF {tif_path.name}. Error: {e}")

def create_visualization(patch_info: dict, vis_dir: Path):
    """Creates and saves a visualization for a single data patch."""
    base_name, data_dir = patch_info['base_name'], patch_info['output_dir']
    try:
        with h5py.File(data_dir / f"{base_name}_source.h5", 'r') as hf: source_img = hf['image'][:]
        with h5py.File(data_dir / f"{base_name}_target_binary.h5", 'r') as hf: target_bin = hf['target'][:]
        with h5py.File(data_dir / f"{base_name}_mask_binary.h5", 'r') as hf: mask_bin = hf['mask'][:]

        rgb_img = source_img[:3, :, :].astype(np.float32)
        for i in range(rgb_img.shape[0]):
            min_val, max_val = np.min(rgb_img[i]), np.max(rgb_img[i])
            if max_val > min_val: rgb_img[i] = (rgb_img[i] - min_val) / (max_val - min_val)
        rgb_img = np.transpose(rgb_img, (1, 2, 0))

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(rgb_img); axes[0].set_title("Source RGB"); axes[0].axis('off')
        axes[1].imshow(target_bin, cmap='gray'); axes[1].set_title("Binary Target"); axes[1].axis('off')
        axes[2].imshow(mask_bin, cmap='magma'); axes[2].set_title("Annotation Mask"); axes[2].axis('off')

        plt.tight_layout()
        vis_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(vis_dir / f"{base_name}.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"    - Could not create visualization for {base_name}. Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess GeoTIFFs by finding overlapping annotations from all XLSX files.")
    parser.add_argument("--data_dir", type=str, default=str(BASE_DATA_DIR), help="Path to the root data directory.")
    parser.add_argument("--output_dir", type=str, default=str(PREPROCESSED_DATA_DIR), help="Path to the output directory.")
    args = parser.parse_args()

    data_dir, output_dir = Path(args.data_dir), Path(args.output_dir)
    print("--- Starting Preprocessing ---")
    random.seed(RANDOM_SEED)

    all_xlsx_paths = list(data_dir.glob("**/*.xlsx"))
    master_annotations_gdf = load_all_annotations(all_xlsx_paths)
    if master_annotations_gdf.empty:
        print("CRITICAL: No valid annotations found in any XLSX files. Exiting."); return
    print(f"\nLoaded a total of {len(master_annotations_gdf)} annotations from all files.")

    all_geotiff_paths = list(data_dir.glob("**/*.tif"))
    if not all_geotiff_paths: print("No GeoTIFF files found. Exiting."); return

    try:
        roi_poly_wgs84 = load_roi_polygon_wgs84(data_dir)
        relevant_geotiffs = get_roi_overlapping_geotiffs(all_geotiff_paths, roi_poly_wgs84)
        print(f"\nFound {len(relevant_geotiffs)} GeoTIFFs overlapping with the ROI.")
        if not relevant_geotiffs: print("No GeoTIFFs overlap with ROI. Exiting."); return
    except Exception as e:
        print(f"Error during ROI processing: {e}. Aborting."); return

    random.shuffle(relevant_geotiffs)
    num_files = len(relevant_geotiffs)
    test_files, val_files, train_files = [], [], []
    if num_files >= 3:
        test_files, val_files, train_files = relevant_geotiffs[0:2], relevant_geotiffs[2:3], relevant_geotiffs[3:]
    elif num_files == 2:
        test_files, val_files = relevant_geotiffs[0:1], relevant_geotiffs[1:2]
    elif num_files == 1:
        val_files = relevant_geotiffs[0:1]

    file_splits = {'train': train_files, 'val': val_files, 'test': test_files}
    print("\n--- Data Split Summary ---")
    for split, files in file_splits.items():
        print(f"  {split.capitalize()} set ({len(files)} files): {[p.name for p in files]}")
    print("--------------------------\n")

    generated_patches = {'train': [], 'val': [], 'test': []}
    for split, tif_files in file_splits.items():
        if not tif_files: continue
        print(f"--- Processing {split.upper()} Set ---")
        split_output_dir = output_dir / split

        for tif_path in tif_files:
            process_geotiff_and_annotations(tif_path, master_annotations_gdf, roi_poly_wgs84, split_output_dir, generated_patches[split])

    print("\n--- Creating Visualizations ---")
    for split, patches in generated_patches.items():
        if not patches:
            print(f"No patches generated for '{split}' set, skipping visualization.")
            continue

        num_samples = min(NUM_VISUALIZATIONS_PER_SPLIT, len(patches))
        print(f"Generating {num_samples} visualizations for '{split}' set...")

        samples_to_visualize = random.sample(patches, num_samples)
        split_vis_dir = VISUALIZATION_DIR / split

        for patch_info in tqdm(samples_to_visualize, desc=f"Visualizing {split} samples"):
            create_visualization(patch_info, split_vis_dir)
        print(f"Visualizations for '{split}' set saved to {split_vis_dir}")

    print("\n--- Preprocessing Complete ---")

if __name__ == "__main__":
    main()

