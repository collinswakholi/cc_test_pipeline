# imports
import os
import glob
import pandas as pd
from tqdm import tqdm
import cv2
import numpy as np
import colour
import concurrent.futures

from utils.logger_ import log_
from key_functions import (
    extract_color_charts,
    get_illuminant_from,
    adapt_chart,
    to_float64,
    get_metrics
)

# parameters
ILLUMINANT_FROM = 'D65'
CMFS = 'CIE 1931 2 Degree Standard Observer'
DATA_FOLDER = 'Data/Light_Temperatures/Results'
LIGHT_TEMPS = ['Amazon', 'Dalatin', 'D50', 'D65']
IMG_SUFFIX = '_Corrected.jpg'

# list all sub-folders in the data folder
data_folders = [
    f for f in os.listdir(DATA_FOLDER)
    if os.path.isdir(os.path.join(DATA_FOLDER, f))
]

# get reference data
REF_ILLUMINANT = get_illuminant_from(ILLUMINANT_FROM, CMFS=CMFS)
REFERENCE_CHART = colour.CCS_COLOURCHECKERS['ColorChecker24 - After November 2014']
REFERENCE_CHART = adapt_chart(REFERENCE_CHART, REF_ILLUMINANT)
data = list(REFERENCE_CHART.data.values())
names = list(REFERENCE_CHART.data.keys())

# Compute reference RGB once and clip to [0,1]
REFERENCE_RGB = colour.XYZ_to_sRGB(
    colour.xyY_to_XYZ(data),
    illuminant=REF_ILLUMINANT,
    apply_cctf_encoding=True
)
REFERENCE_RGB = np.clip(REFERENCE_RGB, 0, 1)


def process_image(image_path, working_dir, ref_illuminant, reference_rgb):
    """
    Processes a single image: reads it, extracts the color chart,
    computes metrics, and saves the results as CSV files.
    """
    try:
        # Extract image name without extension
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            log_(f"Failed to read image: {image_name}", "red", "italic", "warn")
            return

        # Extract color chart(s); adjust indexing if your function returns a tuple/list
        n_charts = 1
        # if image name contains "Combo", extract 2 charts
        if "_Combo_" in image_name:
            log_(f"Extracting 2 charts from image: {image_name}", "cyan", "italic")
            n_charts = 2
        charts = extract_color_charts(image_bgr, n_charts=n_charts)[0]
        if charts is None:
            log_(f"Image does not contain a color chart: {image_name}", "red", "italic", "warn")
            return

        summary_list = []
        all_metrics_list = []

        # Process each detected chart
        for chart_index, chart in enumerate(charts, start=1):
            chart_float = to_float64(chart)
            metrics = get_metrics(chart_float, reference_rgb, illuminant=ref_illuminant, c_space='sRGB')

            # Copy the DataFrames to avoid modifying originals and add a Chart column
            summary_df = metrics.Data_summary.copy()
            all_df = metrics.Data.copy()
            summary_df.insert(0, 'Chart', chart_index)
            all_df.insert(0, 'Chart', chart_index)
            all_df.index = names
            summary_df.index = [image_name+'_CC']

            summary_list.append(summary_df)
            all_metrics_list.append(all_df)

        # Concatenate DataFrames if there is more than one chart
        final_summary = pd.concat(summary_list, ignore_index=False) if len(summary_list) > 1 else summary_list[0]
        final_all = pd.concat(all_metrics_list, ignore_index=False) if len(all_metrics_list) > 1 else all_metrics_list[0]

        # Save CSV files to the working directory
        summary_csv_path = os.path.join(working_dir, f"{image_name}_Summary_Metrics.csv")
        all_csv_path = os.path.join(working_dir, f"{image_name}_All_Metrics.csv")
        final_summary.to_csv(summary_csv_path, float_format='%.12f')
        final_all.to_csv(all_csv_path, float_format='%.12f')

    except Exception as exc:
        log_(f"Error processing image {image_path}: {exc}", "red", "italic", "warn")


def process_folder(working_dir):
    """
    Processes all images in a given folder using parallel processing.
    """
    # Retrieve image files with the designated suffix
    image_files = glob.glob(os.path.join(working_dir, f"*{IMG_SUFFIX}"), recursive=True)
    if not image_files:
        log_(f"No images found in folder: {working_dir}", "yellow", "italic", "warn")
        return

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(process_image, image, working_dir, REF_ILLUMINANT, REFERENCE_RGB): image
            for image in image_files
        }
        # Display progress and catch exceptions for each future
        for future in tqdm(concurrent.futures.as_completed(futures),
                           total=len(futures),
                           desc="Processing images"):
            try:
                future.result()  # Re-raise exceptions if any
            except Exception as exc:
                log_(f"Error in processing future: {exc}", "red", "italic", "warn")


def main():
    """
    Main function to loop over all folders and light temperature subfolders.
    """
    for i, folder in enumerate(data_folders):
        for ii, light_temp in enumerate(LIGHT_TEMPS):
            working_dir = os.path.join(DATA_FOLDER, folder, light_temp)
            if not os.path.exists(working_dir):
                log_(f"Folder does not exist: '{working_dir}'", "red", "italic", "warn")
                continue
            log_(f"Processing folder: {folder} - {light_temp}, \t folder {(i * len(LIGHT_TEMPS) + ii + 1)}/{len(data_folders) * len(LIGHT_TEMPS)}", "magenta", "italic")
            process_folder(working_dir)


if __name__ == '__main__':
    main()
    log_(f'Done...', 'green', 'Bold', 'info')