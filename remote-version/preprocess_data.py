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
from rasterio.windows import Window
from pyproj import Transformer, CRS
from tqdm import tqdm
import shutil

# Import from config
import config

# --- (Functions from get_vegetation_columns to process_geotiff_and_annotations remain unchanged) ---

def get_vegetation_columns(df: pd.DataFrame) -> list:
    """Identifies the correct vegetation column names from the dataframe."""
    try:
        start_index = df.columns.to_list().index('Ste')
        excluded_cols = {'Dens_veg', 'Mat_Pos', 'Comments', 'Kontroll%', 'geometry'}
        potential_cols = df.columns[start_index:].to_list()
        return [c for c in potential_cols if c not in excluded_cols and 'Unnamed' not in str(c)]
    except ValueError:
        return []

def load_all_annotations(all_xlsx_paths: list) -> geopandas.GeoDataFrame:
    """Scans all XLSX files, intelligently finds the header row, and loads all valid annotations."""
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
                        found_header = True; break
                if not found_header: continue
            if 'Latitud' not in df.columns or 'Longitud' not in df.columns: continue
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
    return geopandas.GeoDataFrame(master_df, geometry=geopandas.points_from_xy(master_df.Longitud, master_df.Latitud), crs=config.WGS84_CRS)

def load_roi_polygon_wgs84(data_dir: Path) -> Polygon:
    """Loads ROI points from roi.txt and returns a Polygon in WGS84 CRS."""
    roi_filepath = data_dir / config.ROI_FILE_NAME
    if not roi_filepath.exists(): raise FileNotFoundError(f"{config.ROI_FILE_NAME} not found in {data_dir}.")
    points_wgs84 = pd.read_csv(roi_filepath, header=None, names=['lat', 'lon'])
    return Polygon(zip(points_wgs84.lon, points_wgs84.lat))

def get_roi_overlapping_geotiffs(geotiff_paths: list, roi_poly_wgs84: Polygon) -> list:
    """Filters a list of GeoTIFF paths to include only those whose bounds intersect the ROI."""
    overlapping_files = []
    for tif_path in tqdm(geotiff_paths, desc="Checking for ROI overlap"):
        try:
            with rasterio.open(tif_path) as src:
                raster_bounds_poly = box(*src.bounds)
                transformer = Transformer.from_crs(config.WGS84_CRS, src.crs, always_xy=True)
                transformed_roi_poly = Polygon([transformer.transform(x, y) for x, y in roi_poly_wgs84.exterior.coords])
                if raster_bounds_poly.intersects(transformed_roi_poly):
                    overlapping_files.append(tif_path)
        except Exception as e:
            print(f"Warning: Could not process {tif_path.name} for ROI check. Error: {e}")
    return overlapping_files

def extract_style_patches(train_dir: Path, style_dir: Path):
    """
    Scans generated training patches and saves a selection to be used as 'style' images for FDA.
    """
    print("\n--- Extracting Style Images from Training Patches ---")
    style_metadata = []
    train_patch_paths = list(train_dir.glob("*_source.h5"))
    for patch_path in tqdm(train_patch_paths, desc="Scanning training patches for styles"):
        with h5py.File(patch_path, 'r') as hf:
            patch = hf['image'][:]
            if patch.shape[0] < 3: continue
            score = np.std(patch)
            if score > 20:
                style_metadata.append({'path': patch_path, 'score': score})
    if not style_metadata:
        print("Warning: No suitable style patches found. FDA will not be available.")
        return
    style_metadata.sort(key=lambda x: x['score'], reverse=True)
    num_to_keep = max(1, len(style_metadata) // 4)
    top_styles_metadata = style_metadata[:num_to_keep]
    print(f"Found {len(style_metadata)} potential style patches. Keeping and copying the top {len(top_styles_metadata)}.")
    style_dir.mkdir(parents=True, exist_ok=True)
    for f in style_dir.glob('*.h5'):
        f.unlink()
    for i, patch_info in enumerate(tqdm(top_styles_metadata, desc="Copying best style images")):
        with h5py.File(patch_info['path'], 'r') as hf_in:
             patch = hf_in['image'][:]
        with h5py.File(style_dir / f"style_{i:05d}.h5", "w") as hf_out:
            hf_out.create_dataset("style_image", data=patch)
    print("--- Finished Extracting Style Images ---")

def process_geotiff_and_annotations(tif_path: Path, master_annotations_gdf: geopandas.GeoDataFrame, roi_polygon_wgs84: Polygon, output_dir: Path):
    """Processes a GeoTIFF by generating patches with 7x7 (config.ANNOTATION_SQUARE_SIZE) target squares."""
    total_patches_generated_for_file = 0
    total_discarded_for_file = 0
    try:
        with rasterio.open(tif_path) as src:
            image_crs = src.crs
            annotations_in_img_crs = master_annotations_gdf.to_crs(image_crs)
            annotations_in_image = annotations_in_img_crs[annotations_in_img_crs.geometry.intersects(box(*src.bounds))].copy()
            if annotations_in_image.empty: return 0, 0
            transformer = Transformer.from_crs(config.WGS84_CRS, image_crs, always_xy=True)
            roi_poly_image_crs = Polygon([transformer.transform(x, y) for x, y in roi_polygon_wgs84.exterior.coords])
            annotations_in_roi = annotations_in_image[annotations_in_image.geometry.within(roi_poly_image_crs)].copy()
            if annotations_in_roi.empty: return 0, 0
            veg_cols = get_vegetation_columns(annotations_in_roi)
            if not veg_cols: return 0, 0
            annotations_in_roi[veg_cols] = annotations_in_roi[veg_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
            # This print statement can be noisy if you have many files, but is good for debugging
            print(f"  -> SUCCESS: Found {len(annotations_in_roi)} valid annotations for {tif_path.name}. Generating patches...")
            for ann_idx, annotation in annotations_in_roi.iterrows():
                try:
                    true_row, true_col = src.index(annotation.geometry.x, annotation.geometry.y)
                except rasterio.errors.OutOfBoundTransformError: continue
                for i in range(config.NUM_PATCHES_PER_ANNOTATION):
                    h, w = config.PATCH_HEIGHT_PIXELS, config.PATCH_WIDTH_PIXELS
                    patch_row_in_img = random.randint(0, h - 1);
                    patch_col_in_img = random.randint(0, w - 1)
                    top = true_row - patch_row_in_img
                    left = true_col - patch_col_in_img
                    if top < 0 or left < 0 or (top + h) > src.height or (left + w) > src.width:
                        total_discarded_for_file += 1; continue
                    patch_window = Window(left, top, w, h)
                    source_patch = src.read(window=patch_window)
                    if source_patch.shape[1] != h or source_patch.shape[2] != w: continue
                    if source_patch.shape[0] > 3: source_patch = source_patch[:3, :, :]
                    if np.max(source_patch) == 0 or (np.issubdtype(source_patch.dtype, np.integer) and np.min(source_patch) == np.iinfo(source_patch.dtype).max):
                        total_discarded_for_file += 1; continue
                    patch_bounds_geo = src.window_bounds(patch_window)
                    patch_poly = box(*patch_bounds_geo)
                    annotations_in_patch = annotations_in_roi[annotations_in_roi.geometry.intersects(patch_poly)]
                    if annotations_in_patch.empty: continue
                    target_binary = np.zeros((h, w), dtype=np.uint8); mask_binary = np.zeros_like(target_binary)
                    for _, ann_in_patch in annotations_in_patch.iterrows():
                        try:
                            abs_row, abs_col = src.index(ann_in_patch.geometry.x, ann_in_patch.geometry.y)
                            rel_row, rel_col = abs_row - top, abs_col - left
                            if not (0 <= rel_row < h and 0 <= rel_col < w): continue
                            square_size = config.ANNOTATION_SQUARE_SIZE; offset = square_size // 2
                            r_min, r_max = max(0, rel_row - offset), min(h, rel_row + offset + 1)
                            c_min, c_max = max(0, rel_col - offset), min(w, rel_col + offset + 1)
                            mask_binary[r_min:r_max, c_min:c_max] = 1
                            if ann_in_patch[veg_cols].sum() >= 40:
                                target_binary[r_min:r_max, c_min:c_max] = 1
                        except Exception: continue
                    total_patches_generated_for_file += 1
                    base_name = f"{tif_path.stem}_patch_{ann_idx}_{i}"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    with h5py.File(output_dir / f"{base_name}_source.h5", "w") as hf: hf.create_dataset("image", data=source_patch)
                    with h5py.File(output_dir / f"{base_name}_target_binary.h5", "w") as hf: hf.create_dataset("target", data=target_binary)
                    with h5py.File(output_dir / f"{base_name}_mask_binary.h5", "w") as hf: hf.create_dataset("mask", data=mask_binary)
            return total_patches_generated_for_file, total_discarded_for_file
    except Exception as e:
        print(f"FATAL: Failed to process GeoTIFF {tif_path.name}. Error: {e}")
        return 0, 0

# --- NEW FUNCTION ---
def split_and_move_patches(all_patches_dir: Path, output_dir: Path):
    """
    Shuffles all generated patches and splits them into train, val, and test sets.
    """
    print("\n--- Splitting generated patches into train, validation, and test sets ---")

    # Get unique basenames of all generated patches from the source files
    source_files = list(all_patches_dir.glob("*_source.h5"))
    if not source_files:
        print("Warning: No patches were generated. Cannot create splits.")
        return

    basenames = [p.name.replace("_source.h5", "") for p in source_files]
    random.shuffle(basenames)

    # Calculate split sizes: 20% for test, 10% for validation
    total_patches = len(basenames)
    test_size = int(total_patches * 0.20)
    val_size = int(total_patches * 0.10)

    # Distribute the patches
    test_patches = basenames[:test_size]
    val_patches = basenames[test_size : test_size + val_size]
    train_patches = basenames[test_size + val_size :]

    print(f"Total patches: {total_patches}")
    print(f"  - Training set size:   {len(train_patches)} patches (~{100*len(train_patches)/total_patches:.1f}%)")
    print(f"  - Validation set size: {len(val_patches)} patches (~{100*len(val_patches)/total_patches:.1f}%)")
    print(f"  - Test set size:       {len(test_patches)} patches (~{100*len(test_patches)/total_patches:.1f}%)")
    print("------------------------------------------------------------------")

    # Define final split directories
    split_dirs = {
        "train": output_dir / "train",
        "val": output_dir / "val",
        "test": output_dir / "test"
    }

    # Helper function to move files for a given list of basenames
    def move_files(patch_list, dest_dir):
        dest_dir.mkdir(parents=True, exist_ok=True)
        for basename in tqdm(patch_list, desc=f"Moving files to {dest_dir.name}"):
            shutil.move(str(all_patches_dir / f"{basename}_source.h5"), str(dest_dir))
            shutil.move(str(all_patches_dir / f"{basename}_target_binary.h5"), str(dest_dir))
            shutil.move(str(all_patches_dir / f"{basename}_mask_binary.h5"), str(dest_dir))

    # Move the files to their final destinations
    move_files(train_patches, split_dirs["train"])
    move_files(val_patches, split_dirs["val"])
    move_files(test_patches, split_dirs["test"])

    # Clean up the temporary directory
    shutil.rmtree(all_patches_dir)
    print("\nPatch splitting and moving complete.")


def main():
    parser = argparse.ArgumentParser(description="Preprocess GeoTIFFs, create patches, then extract style images.")
    parser.add_argument("--data_dir", type=str, default=str(config.DATA_DIR), help="Path to the root data directory.")
    parser.add_argument("--output_dir", type=str, default=str(config.PREPROCESSED_DATA_DIR), help="Path to the output directory.")
    parser.add_argument("--years", type=str, default=None, help="Optional: Comma-separated list of years to process (e.g., '2023,2024').")
    args = parser.parse_args()

    random.seed(config.GLOBAL_RANDOM_SEED)
    data_dir, output_dir = Path(args.data_dir), Path(args.output_dir)
    print("--- Starting Preprocessing ---")

    years_to_process = args.years.split(',') if args.years else None
    if years_to_process:
        print(f"Filtering dataset to include only years: {', '.join(years_to_process)}")

    # --- (Annotation and GeoTIFF loading/filtering logic remains the same) ---
    all_xlsx_paths = list(data_dir.glob("**/*.xlsx"))
    if years_to_process:
        initial_count = len(all_xlsx_paths)
        all_xlsx_paths = [p for p in all_xlsx_paths if any(y in str(p) for y in years_to_process)]
        print(f"Filtered XLSX files: Kept {len(all_xlsx_paths)} out of {initial_count}")
    master_annotations_gdf = load_all_annotations(all_xlsx_paths)
    if master_annotations_gdf.empty:
        print("CRITICAL: No valid annotations found in any XLSX files. Exiting."); return
    print(f"\nLoaded a total of {len(master_annotations_gdf)} annotations from all files.")

    veg_cols = get_vegetation_columns(master_annotations_gdf)
    if veg_cols:
        temp_df = master_annotations_gdf.copy()
        temp_df[veg_cols] = temp_df[veg_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        total_vegetation = temp_df[veg_cols].sum(axis=1)
        binary_labels = (total_vegetation >= 40).astype(int)
        label_counts = binary_labels.value_counts()
        print("\n--- Raw Annotation Label Distribution ---")
        print(f"Total annotations loaded: {len(master_annotations_gdf)}")
        print(f"  - Class 0 (Non-Vegetation, < 40%): {label_counts.get(0, 0)}")
        print(f"  - Class 1 (Vegetation, >= 40%):    {label_counts.get(1, 0)}")
        print("-----------------------------------------\n")

    all_geotiff_paths = list(data_dir.glob("**/*.tif"))
    if not all_geotiff_paths: print("No GeoTIFF files found. Exiting."); return
    try:
        roi_poly_wgs84 = load_roi_polygon_wgs84(data_dir)
        relevant_geotiffs = get_roi_overlapping_geotiffs(all_geotiff_paths, roi_poly_wgs84)
        print(f"\nFound {len(relevant_geotiffs)} GeoTIFFs overlapping with the ROI.")
        if not relevant_geotiffs: print("No GeoTIFFs overlap with ROI. Exiting."); return
    except Exception as e:
        print(f"Error during ROI processing: {e}. Aborting."); return

    # --- ENTIRE DATA SPLIT LOGIC IS REPLACED ---

    # 1. Create a single temporary directory to store ALL generated patches
    all_patches_dir = output_dir / "all_patches"
    if all_patches_dir.exists(): shutil.rmtree(all_patches_dir)
    all_patches_dir.mkdir(parents=True)

    # Clean out old final directories
    if config.TRAIN_DIR.exists(): shutil.rmtree(config.TRAIN_DIR)
    if config.VAL_DIR.exists(): shutil.rmtree(config.VAL_DIR)
    if config.TEST_DIR.exists(): shutil.rmtree(config.TEST_DIR)

    print("\n--- Generating all patches from relevant GeoTIFFs into a temporary directory ---")
    total_generated = 0
    total_discarded = 0
    # 2. Process all relevant GeoTIFFs and save patches to the temporary directory
    for tif_path in tqdm(relevant_geotiffs, desc="Processing GeoTIFFs"):
        generated, discarded = process_geotiff_and_annotations(
            tif_path,
            master_annotations_gdf,
            roi_poly_wgs84,
            all_patches_dir  # Save all patches to one central place
        )
        total_generated += generated
        total_discarded += discarded

    print("\n--- Finished patch generation ---")
    print(f"  - Total patches generated: {total_generated}")
    print(f"  - Total patches discarded: {total_discarded}")

    # 3. Now, split all the generated patches into train, val, and test sets
    if total_generated > 0:
        split_and_move_patches(all_patches_dir, output_dir)
    else:
        print("\nNo patches were generated, skipping split.")

    # 4. Extract style images from the final training set that was just created
    if config.TRAIN_DIR.exists():
        extract_style_patches(config.TRAIN_DIR, config.STYLE_IMAGES_DIR)

    print("\n--- Preprocessing Complete ---")

if __name__ == "__main__":
    main()

