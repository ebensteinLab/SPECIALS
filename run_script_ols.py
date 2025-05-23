import os
import time
import cupy as cp
import numpy as np
from tifffile import tifffile

from create_psfs import new_analyze_ref_stack, average_crop_all_pictures
from deconvolution import decompose_to_channels_batched_only_gpu
from get_crops import get_crops_all_pictures
from ism import run_ism
from utils import flatten, myimrescale


def main():
    # Define paths
    ref_path = r"C:\Users\User\Desktop\CoCoS_ISM\Data\All_Refs"
    images_path = r"C:\Users\User\Desktop\CoCoS_ISM\Data\170nmPSbeads-0604"

    # Spectral reference files
    ref_red_name = (
        "Ref640_N250_LPN2_TL8_TR1389p5_TCon100_TCoff15_TD5p56_TA0_Gain500_64010.tif"
    )
    ref_green_name = (
        "Ref560_N250_LPN2_TL8_TR1389p5_TCon100_TCoff15_TD5p56_TA0_Gain500_56010.tif"
    )
    ref_blue_name = (
        "Ref488_N250_LPN2_TL8_TR1389p5_TCon100_TCoff15_TD5p56_TA0_Gain500_488I10.tif"
    )

    # Red reference to register spots locations
    ref_red_name_im = "ref_170nmPS_640_N250_LPN2_TL8_TR1390_TCon100_TCoff15_TD5p56_TA0_Gain5_640I10.tif"

    # Image stack to analyze
    im_rgb_name = "FOV2_170nmPS_All_N250_LPN2_TL8_TR1390_TCon200_TCoff15_TD5p56_TA0_Gain5_AllI10.tif"

    spectral_ref_stack_paths = [
        os.path.join(ref_path, ref_red_name),
        os.path.join(ref_path, ref_green_name),
        os.path.join(ref_path, ref_blue_name),
    ]

    # references
    t = time.time()
    reference_peaks = new_analyze_ref_stack(os.path.join(ref_path, ref_red_name))
    psfs = average_crop_all_pictures(spectral_ref_stack_paths, reference_peaks)

    peaks = new_analyze_ref_stack(os.path.join(images_path, ref_red_name_im))
    print("Reference time:" + str(time.time() - t))

    # get the crops
    t = time.time()
    rgb_path = os.path.join(images_path, im_rgb_name)
    with tifffile.TiffFile(rgb_path) as tif:
        stack = tif.asarray()
    all_crops, all_crops_max, stack_mean, stack_std = get_crops_all_pictures(
        stack, peaks
    )

    flattened_P = np.array(flatten(peaks))
    high_intensity_crops_index = np.where(all_crops_max > stack_mean + 6 * stack_std)[
        0
    ]  # this is not always good, may return way too many crops. #TODO: improve
    crops_with_high_intensity = all_crops[high_intensity_crops_index, :, :]
    peaks_with_high_intensity = flattened_P[high_intensity_crops_index]
    rounded_peaks = np.round(peaks_with_high_intensity)
    print("get crops time:" + str(time.time() - t))

    # get image!

    t = time.time()
    decomposed_image_batched_new, all_convolved_crops_new = (
        decompose_to_channels_batched_only_gpu(
            crops_with_high_intensity, psfs, rounded_peaks
        )
    )
    print("get decomposed time:" + str(time.time() - t))

    # get ism image!
    t = time.time()
    final_ism_opt = run_ism(
        all_convolved_crops=all_convolved_crops_new,
        M=8,
        peaks=peaks_with_high_intensity,
        scatter_add_batch_size=5000,
    )
    print("get ism time:" + str(time.time() - t))

    return decomposed_image_batched_new, final_ism_opt


if __name__ == "__main__":
    decomposed, ism = main()
    rescaled_image = myimrescale(cp.asnumpy(decomposed))  #
