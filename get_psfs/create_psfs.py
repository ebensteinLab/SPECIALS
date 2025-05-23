from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pims
import scipy.io as sio
import trackpy as tp
from tifffile import tifffile

from utils import CROP_SIZE, centers_to_rows_and_columns


def process_frame(frame):
    """
    Run trackpy locate, can add preprocessing if wanted.
    """
    return tp.locate(frame, diameter=7, minmass=100)


def find_peaks_using_trackpy(frames_numpy, border_exclude_size):
    """
    Find peaks in a stack of images.
    """
    with ThreadPoolExecutor() as executor:  # multithreading for speed
        futures = []
        for i, frame in enumerate(frames_numpy):
            futures.append(executor.submit(process_frame, frame))

    results = [f.result() for f in futures]
    peaks = []
    sigma = []
    for features in results:
        features["x"] += (
            border_exclude_size  # We give a cropped image to exclude borders, so shift back the locations.
        )
        features["y"] += border_exclude_size
        p_x = features["x"].values
        p_y = features["y"].values
        sigma.append(np.mean(features["size"].values) / np.sqrt(2))
        peaks.append(np.column_stack((p_x, p_y)))
    return peaks, np.mean(np.asarray(sigma))


def new_analyze_ref_stack(stack_path, skip_factor=1, border_exclude_size=13):
    """
    Run peak finding on a stack of images without borders.
    """
    frames = pims.open(stack_path)
    frames = frames[::skip_factor]
    frames_numpy = np.array(
        [
            frame[
                border_exclude_size:-border_exclude_size,
                border_exclude_size:-border_exclude_size,
            ]
            for frame in frames
        ]
    )
    peak_locations, sigmas = find_peaks_using_trackpy(frames_numpy, border_exclude_size)
    return peak_locations, sigmas


def average_crop_one_picture(Im, centers, red_x_location):
    """
    Average all crops in a picture (reference)
    Do all operations with numpy (without loops) to ensure efficiency
    """
    rows, cols = centers_to_rows_and_columns(centers, red_x_location)
    crops = Im[rows, cols]

    # Compute border median for all crops at once
    border = np.concatenate(
        [crops[:, 0, :], crops[:, -1, :], crops[:, :, 0], crops[:, :, -1]], axis=1
    )
    border_median = np.median(border, axis=1)[:, None, None]

    # Apply operations to all crops simultaneously
    crops = crops - border_median
    np.clip(crops, 0, None, out=crops)
    crops /= np.max(crops, axis=(1, 2))[:, None, None]

    # Compute average
    average_array = np.mean(crops, axis=0)

    return average_array


def average_crop_all_pictures(spectral_ref_stack_paths, peaks, red_x_location):
    """
    Average all crops for each color, on all images (reference).
    """

    num_channels = len(spectral_ref_stack_paths)
    average_crop_all_pictures_all_colors = np.zeros(
        (CROP_SIZE[0], CROP_SIZE[1], num_channels)
    )
    for i, color in enumerate(spectral_ref_stack_paths):
        with tifffile.TiffFile(color) as tif:
            num_images = len(tif.pages)
            stack = tif.asarray()  # Load all images at once for efficiency

        average_crop_all_pictures_one_color = np.zeros(CROP_SIZE)
        for j in range(num_images):
            average_crop_all_pictures_one_color += average_crop_one_picture(
                stack[j], np.round(peaks[j]), red_x_location
            )  # perhaps can be improved, but not critical.
        average_crop_all_pictures_one_color = (
            average_crop_all_pictures_one_color / num_images
        )
        average_crop_all_pictures_all_colors[:, :, i] = (
            average_crop_all_pictures_one_color
        )

    return average_crop_all_pictures_all_colors


def import_external_ref_psfs(
    external_ref_psfs_stack_path,
    external_ref_psfs_mat_path,
    external_ref_psfs_names_path,
):
    stack = tifffile.TiffFile(external_ref_psfs_stack_path).asarray()
    # num_external_psfs = len(tif.pages)
    # stack = tif.asarray()  # Load all images at once for efficiency
    psfs_mat = sio.loadmat(external_ref_psfs_mat_path)
    psfs_names = sio.loadmat(external_ref_psfs_names_path)

    return stack, psfs_mat, psfs_names
