import os
import glob
import random
from pathlib import Path
import argparse
import h5py
import json  # Added for JSON output
import math
from datetime import datetime
import numpy as np
import pandas as pd
import geopandas
from shapely.geometry import Point, Polygon, box
import rasterio
from rasterio.windows import Window
from pyproj import Transformer, CRS
from tqdm import tqdm
import shutil
from scipy.ndimage import gaussian_filter
import fiona

# Import from config
import config

# --- HELPER FUNCTIONS (UNCHANGED) ---

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

def extract_style_patches(train_dir: Path, style_dir: Path, grayscale: bool = False, num_styles: int = 200, sigma: int = 3):
    """
    Applies a Gaussian filter to training patches to extract high-frequency patterns,
    saving them as style images.
    """
    print("\n--- Extracting Style Images using Gaussian High-Pass Filter ---")
    train_patch_paths = list(train_dir.glob("*_source.h5"))

    if not train_patch_paths:
        print("Warning: No training patches found. Cannot generate style images.")
        return

    num_to_process = min(num_styles, len(train_patch_paths))
    selected_patches = random.sample(train_patch_paths, num_to_process)
    print(f"Randomly selected {num_to_process} training patches to generate style images.")
    if grayscale:
        print("Style images will be converted to 3-channel grayscale.")

    style_dir.mkdir(parents=True, exist_ok=True)
    for f in style_dir.glob('*.h5'):
        f.unlink()

    for i, patch_path in enumerate(tqdm(selected_patches, desc="Generating style images")):
        with h5py.File(patch_path, 'r') as hf:
            original_image = hf['image'][:]

        if original_image.shape[0] < 3: continue

        low_freq = gaussian_filter(original_image, sigma=[0, sigma, sigma])
        high_freq = original_image - low_freq
        style_image = high_freq
        if grayscale:
            gray_channel = np.mean(style_image, axis=0, keepdims=True)
            style_image = np.repeat(gray_channel, 3, axis=0)

        with h5py.File(style_dir / f"style_{i:05d}.h5", "w") as hf_out:
            hf_out.create_dataset("style_image", data=style_image)

    print("--- Finished Extracting Style Images ---")


def generate_random_negative_annotations(src: rasterio.io.DatasetReader, land_gdf: geopandas.GeoDataFrame, num_points: int) -> list:
    """
    Generates a list of random Shapely Point objects that are guaranteed
    to be on land and not in a NoData (black) area of the raster.
    """
    if num_points == 0 or land_gdf.empty:
        return []

    generated_points = []
    min_x, min_y, max_x, max_y = src.bounds
    attempts, max_attempts = 0, num_points * 20
    nodata_val, use_nodata_meta = src.nodata, src.nodata is not None

    pbar = tqdm(total=num_points, desc="    - Generating random negative points")
    while len(generated_points) < num_points and attempts < max_attempts:
        attempts += 1
        rand_x, rand_y = random.uniform(min_x, max_x), random.uniform(min_y, max_y)
        random_point = Point(rand_x, rand_y)

        if not land_gdf.contains(random_point).any(): continue

        try:
            row, col = src.index(rand_x, rand_y)
            pixel_val = src.read((1, 2, 3), window=Window(col, row, 1, 1))
            is_nodata = np.all(pixel_val == nodata_val) if use_nodata_meta else pixel_val.sum() < 15
            if is_nodata: continue
            
            generated_points.append(random_point)
            pbar.update(1)
        except (rasterio.errors.RasterioIOError, IndexError):
            continue
    
    pbar.close()
    if attempts >= max_attempts:
        print(f"    - Warning: Reached max attempts. Proceeding with {len(generated_points)} points.")
    return generated_points

# --- CORE PROCESSING FUNCTION (MODIFIED) ---

def process_geotiff_and_annotations(tif_path: Path, master_annotations_gdf: geopandas.GeoDataFrame, land_gdf: geopandas.GeoDataFrame, roi_polygon_wgs84: Polygon, output_dir: Path):
    """
    Processes a GeoTIFF by generating patches from both real annotations
    and newly generated random negative annotations within a buffered "inland" area.
    The buffer operation is now robustly handled in a projected CRS.
    """
    total_patches_generated = 0
    total_discarded = 0
    all_patch_annotations_for_file = [] # Make sure to re-add this if it was removed

    try:
        with rasterio.open(tif_path) as src:
            image_crs = src.crs

            # --- ROBUST BUFFERING LOGIC ---
            # Define the standard projected CRS for Sweden (in meters)
            projected_crs = "EPSG:3006" # SWEREF99 TM

            # 1. Reproject the global land GDF to our standard projected CRS
            land_gdf_projected = land_gdf.to_crs(projected_crs)

            # 2. Perform the buffer operation accurately in meters
            if config.COASTLINE_BUFFER_METERS > 0:
                print(f"    - Applying a {-config.COASTLINE_BUFFER_METERS}m buffer in {projected_crs}.")
                inland_geometry_projected = land_gdf_projected.geometry.buffer(-config.COASTLINE_BUFFER_METERS)
                inland_gdf_projected = geopandas.GeoDataFrame(geometry=inland_geometry_projected, crs=projected_crs)
                inland_gdf_projected = inland_gdf_projected[~inland_gdf_projected.is_empty]
            else:
                inland_gdf_projected = land_gdf_projected

            # 3. Reproject the final (possibly buffered) geometry to the image's local CRS for sampling
            inland_gdf_for_sampling = inland_gdf_projected.to_crs(image_crs)
            # --- END ROBUST BUFFERING LOGIC ---

            # --- (The rest of the function proceeds as before) ---
            annotations_in_img_crs = master_annotations_gdf.to_crs(image_crs)
            annotations_in_image = annotations_in_img_crs[annotations_in_img_crs.geometry.intersects(box(*src.bounds))].copy()
            transformer = Transformer.from_crs(config.WGS84_CRS, image_crs, always_xy=True)
            roi_poly_image_crs = Polygon([transformer.transform(x, y) for x, y in roi_polygon_wgs84.exterior.coords])
            annotations_in_roi = annotations_in_image[annotations_in_image.geometry.within(roi_poly_image_crs)].copy()

            veg_cols = get_vegetation_columns(annotations_in_roi)
            if not veg_cols:
                annotations_in_roi = geopandas.GeoDataFrame(columns=annotations_in_roi.columns)
            else:
                annotations_in_roi[veg_cols] = annotations_in_roi[veg_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

            random_points = []
            if config.NUM_NEGATIVE_LAND_ANNOTATIONS_PER_TIF > 0 and not inland_gdf_for_sampling.empty:
                random_points = generate_random_negative_annotations(src, inland_gdf_for_sampling, config.NUM_NEGATIVE_LAND_ANNOTATIONS_PER_TIF)

            if random_points:
                new_rows = {'geometry': random_points}
                for col in veg_cols: new_rows[col] = 0
                neg_gdf = geopandas.GeoDataFrame(new_rows, crs=image_crs)
                all_annotations_for_file = pd.concat([annotations_in_roi, neg_gdf], ignore_index=True)
                print(f"  -> Generated and added {len(random_points)} random negative annotations.")
            else:
                all_annotations_for_file = annotations_in_roi

            if all_annotations_for_file.empty:
                return 0, 0, []

            print(f"  -> SUCCESS: Found {len(all_annotations_for_file)} total annotations for {tif_path.name}. Generating patches...")
            for ann_idx, annotation in all_annotations_for_file.iterrows():
                try:
                    true_row, true_col = src.index(annotation.geometry.x, annotation.geometry.y)
                except rasterio.errors.OutOfBoundTransformError: continue
                for i in range(config.NUM_PATCHES_PER_ANNOTATION):
                    h, w = config.PATCH_HEIGHT_PIXELS, config.PATCH_WIDTH_PIXELS
                    top, left = true_row - random.randint(0, h - 1), true_col - random.randint(0, w - 1)
                    if not (0 <= top and 0 <= left and (top + h) <= src.height and (left + w) <= src.width):
                        total_discarded += 1; continue
                    
                    patch_window = Window(left, top, w, h)
                    source_patch = src.read(window=patch_window)
                    if source_patch.shape[1] != h or source_patch.shape[2] != w: continue
                    if source_patch.shape[0] > 3: source_patch = source_patch[:3, :, :]
                    if np.max(source_patch) == 0:
                        total_discarded += 1; continue

                    patch_bounds_geo = src.window_bounds(patch_window)
                    annotations_in_patch = all_annotations_for_file[all_annotations_for_file.geometry.intersects(box(*patch_bounds_geo))]
                    if annotations_in_patch.empty: continue

                    annotations_for_this_patch_json = []
                    target_binary = np.zeros((h, w), dtype=np.uint8); mask_binary = np.zeros_like(target_binary)
                    for _, ann_in_patch in annotations_in_patch.iterrows():
                        is_veg = 1 if ann_in_patch[veg_cols].sum() >= 40 else 0
                        annotations_for_this_patch_json.append({'x': ann_in_patch.geometry.x, 'y': ann_in_patch.geometry.y, 'label': is_veg})
                        
                        abs_row, abs_col = src.index(ann_in_patch.geometry.x, ann_in_patch.geometry.y)
                        rel_row, rel_col = abs_row - top, abs_col - left
                        if (0 <= rel_row < h and 0 <= rel_col < w):
                            mask_binary[rel_row, rel_col] = 1
                            if is_veg: target_binary[rel_row, rel_col] = 1
                    
                    total_patches_generated += 1
                    base_name = f"{tif_path.stem}_patch_{ann_idx}_{i}"
                    
                    all_patch_annotations_for_file.append({'basename': base_name, 'source_geotiff': str(tif_path.name), 'crs': str(image_crs), 'annotations': annotations_for_this_patch_json})

                    output_dir.mkdir(parents=True, exist_ok=True)
                    with h5py.File(output_dir / f"{base_name}_source.h5", "w") as hf: hf.create_dataset("image", data=source_patch)
                    with h5py.File(output_dir / f"{base_name}_target_binary.h5", "w") as hf: hf.create_dataset("target", data=target_binary)
                    with h5py.File(output_dir / f"{base_name}_mask_binary.h5", "w") as hf: hf.create_dataset("mask", data=mask_binary)
            
            return total_patches_generated, total_discarded, all_patch_annotations_for_file
    except Exception as e:
        print(f"FATAL: Failed to process GeoTIFF {tif_path.name}. Error: {e}")
        return 0, 0, []

# --- NEW HELPER FUNCTION FOR SAVING JSON ---

def save_annotations_to_json(basenames_in_split: list, all_annotations_data: list, output_path: Path):
    """Filters the master annotation list and saves the result to a JSON file."""
    if not basenames_in_split: return
    print(f"  -> Filtering and saving annotations for {output_path.parent.name} split...")
    basenames_set = set(basenames_in_split)
    split_annotations = [item for item in all_annotations_data if item['basename'] in basenames_set]

    # Ensure the parent directory (e.g., 'preprocessed_data/train/') exists before writing the file.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # --- END FIX ---
    
    with open(output_path, 'w') as f:
        json.dump(split_annotations, f, indent=2)
    print(f"     - Saved {len(split_annotations)} patch records to {output_path.name}")


# --- SPLITTING FUNCTION (MODIFIED) ---

def split_and_move_patches(all_patches_dir: Path, output_dir: Path, all_annotations_data: list):
    """
    Shuffles all generated patches, splits them into train/val/test sets,
    and saves a corresponding annotations.json file for each split.
    """
    print("\n--- Splitting generated patches and annotations ---")
    source_files = list(all_patches_dir.glob("*_source.h5"))
    if not source_files:
        print("Warning: No patches were generated. Cannot create splits."); return

    basenames = [p.name.replace("_source.h5", "") for p in source_files]
    random.shuffle(basenames)

    total_patches = len(basenames)
    test_size, val_size = int(total_patches * 0.20), int(total_patches * 0.10)
    test_patches, val_patches, train_patches = basenames[:test_size], basenames[test_size : test_size + val_size], basenames[test_size + val_size :]

    split_dirs = {"train": output_dir / "train", "val": output_dir / "val", "test": output_dir / "test"}

    save_annotations_to_json(train_patches, all_annotations_data, split_dirs["train"] / "annotations.json")
    save_annotations_to_json(val_patches, all_annotations_data, split_dirs["val"] / "annotations.json")
    save_annotations_to_json(test_patches, all_annotations_data, split_dirs["test"] / "annotations.json")

    def move_files(patch_list, dest_dir):
        dest_dir.mkdir(parents=True, exist_ok=True)
        for basename in tqdm(patch_list, desc=f"Moving files to {dest_dir.name}"):
            shutil.move(str(all_patches_dir / f"{basename}_source.h5"), str(dest_dir))
            shutil.move(str(all_patches_dir / f"{basename}_target_binary.h5"), str(dest_dir))
            shutil.move(str(all_patches_dir / f"{basename}_mask_binary.h5"), str(dest_dir))

    move_files(train_patches, split_dirs["train"]); move_files(val_patches, split_dirs["val"]); move_files(test_patches, split_dirs["test"])
    shutil.rmtree(all_patches_dir)
    print("\nPatch splitting, moving, and annotation saving complete.")

# --- MAIN FUNCTION (MODIFIED) ---

def main():
    parser = argparse.ArgumentParser(description="Preprocess GeoTIFFs, create patches, then extract style images.")
    parser.add_argument("--data_dir", type=str, default=str(config.DATA_DIR), help="Path to the root data directory.")
    parser.add_argument("--output_dir", type=str, default=str(config.PREPROCESSED_DATA_DIR), help="Path to the output directory.")
    parser.add_argument("--years", type=str, default=None, help="Optional: Comma-separated list of years to process (e.g., '2023,2024').")
    parser.add_argument("--grayscale_style", action='store_true', help="Convert style images to 3-channel grayscale.")
    args = parser.parse_args()

    random.seed(config.GLOBAL_RANDOM_SEED)
    data_dir, output_dir = Path(args.data_dir), Path(args.output_dir)
    print("--- Starting Preprocessing ---")

    preprocessing_settings = {
        'timestamp': datetime.now().isoformat(),
        'cli_args': {
            'data_dir': str(args.data_dir),
            'output_dir': str(args.output_dir),
            'years': args.years,
            'grayscale_style': args.grayscale_style
        },
        'config_settings': {
            'PATCH_HEIGHT_PIXELS': getattr(config, 'PATCH_HEIGHT_PIXELS', None),
            'PATCH_WIDTH_PIXELS': getattr(config, 'PATCH_WIDTH_PIXELS', None),
            'NUM_PATCHES_PER_ANNOTATION': getattr(config, 'NUM_PATCHES_PER_ANNOTATION', None),
            'NUM_NEGATIVE_LAND_ANNOTATIONS_PER_TIF': getattr(config, 'NUM_NEGATIVE_LAND_ANNOTATIONS_PER_TIF', None),
            'COASTLINE_BUFFER_METERS': getattr(config, 'COASTLINE_BUFFER_METERS', None),
            'LAND_SHP_PATH': getattr(config, 'LAND_SHP_PATH', None),
            'ROI_FILE_NAME': getattr(config, 'ROI_FILE_NAME', None),
            'GLOBAL_RANDOM_SEED': getattr(config, 'GLOBAL_RANDOM_SEED', None)
        }
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    settings_file_path = output_dir / "dataset_settings.json"
    try:
        with open(settings_file_path, 'w') as f:
            json.dump(preprocessing_settings, f, indent=4)
        print(f"\n--- Saved preprocessing settings to {settings_file_path} ---")
    except Exception as e:
        print(f"\n--- WARNING: Could not save preprocessing settings. Reason: {e} ---")

    try:
        land_shp_path = data_dir / config.LAND_SHP_PATH
        if not land_shp_path.exists():
            raise FileNotFoundError
        land_gdf = geopandas.read_file(land_shp_path)
        print("\nSuccessfully pre-loaded land shapefile for negative annotation generation.")
    except Exception as e:
        # This will catch ANY error and print its exact type and message.
        print(f"CRITICAL: An unexpected error occurred while loading the shapefile.")
        print(f"  -> Path being checked: {land_shp_path}")
        print(f"  -> Error Type: {type(e).__name__}")
        print(f"  -> Error Message: {e}")
        return
    #except (FileNotFoundError, AttributeError):
    #    print(f"CRITICAL: Land shapefile not found or LAND_SHP_PATH not set in config.py. Cannot generate negative annotations.", land_shp_path)
    #    return

    years_to_process = args.years.split(',') if args.years else None
    all_xlsx_paths = list(data_dir.glob("**/*.xlsx"))
    if years_to_process:
        all_xlsx_paths = [p for p in all_xlsx_paths if any(y in str(p) for y in years_to_process)]
    
    master_annotations_gdf = load_all_annotations(all_xlsx_paths)
    if master_annotations_gdf.empty:
        print("CRITICAL: No valid annotations found in any XLSX files. Exiting."); return
    print(f"\nLoaded a total of {len(master_annotations_gdf)} annotations from all files.")

    all_geotiff_paths = list(data_dir.glob("**/*.tif"))
    if not all_geotiff_paths: print("No GeoTIFF files found. Exiting."); return
    try:
        roi_poly_wgs84 = load_roi_polygon_wgs84(data_dir)
        relevant_geotiffs = get_roi_overlapping_geotiffs(all_geotiff_paths, roi_poly_wgs84)
        if not relevant_geotiffs: print("No GeoTIFFs overlap with ROI. Exiting."); return
    except Exception as e:
        print(f"Error during ROI processing: {e}. Aborting."); return

    all_patches_dir = output_dir / "all_patches"
    if all_patches_dir.exists(): shutil.rmtree(all_patches_dir)
    all_patches_dir.mkdir(parents=True, exist_ok=True)
    if config.TRAIN_DIR.exists(): shutil.rmtree(config.TRAIN_DIR)
    if config.VAL_DIR.exists(): shutil.rmtree(config.VAL_DIR)
    if config.TEST_DIR.exists(): shutil.rmtree(config.TEST_DIR)

    all_annotations_data_master = []
    total_generated, total_discarded = 0, 0
    print("\n--- Generating all patches from relevant GeoTIFFs into a temporary directory ---")
    for tif_path in tqdm(relevant_geotiffs, desc="Processing GeoTIFFs"):
        generated, discarded, annotations_from_file = process_geotiff_and_annotations(
            tif_path, master_annotations_gdf, land_gdf, roi_poly_wgs84, all_patches_dir
        )
        total_generated += generated
        total_discarded += discarded
        if annotations_from_file:
            all_annotations_data_master.extend(annotations_from_file)

    print(f"\n--- Finished patch generation ---\n  - Total patches generated: {total_generated}\n  - Total patches discarded: {total_discarded}")

    if total_generated > 0:
        split_and_move_patches(all_patches_dir, output_dir, all_annotations_data_master)
    else:
        print("\nNo patches were generated, skipping split.")

    if config.TRAIN_DIR.exists() and any(config.TRAIN_DIR.glob("*_source.h5")):
        print("\n--- Found training patches. Generating style images... ---")
        extract_style_patches(config.TRAIN_DIR, config.STYLE_IMAGES_DIR, grayscale=args.grayscale_style)
    else:
        print("\n--- No training patches found. Skipping style image generation. ---")


    print("\n--- Preprocessing Complete ---")

if __name__ == "__main__":
    main()
