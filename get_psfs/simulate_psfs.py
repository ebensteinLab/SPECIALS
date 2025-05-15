import os

import numpy as np
import pandas as pd
from scipy.io import loadmat


def load_spectrum_files(base_folder, pattern):
    """Load spectrum files matching a given pattern."""
    spectrum_files = []
    for root, _, files in os.walk(base_folder):
        for file in files:
            if file.endswith(pattern):
                spectrum_files.append(os.path.join(root, file))
    return spectrum_files


def read_spectrum(file_path):
    """Read a spectrum file into a NumPy array using pandas to handle complex formatting."""
    try:
        df = pd.read_csv(file_path, header=None, comment="%", delimiter="\t")
        return df.values
    except Exception as e:
        raise ValueError(f"Error reading spectrum file {file_path}: {e}")


def gaussian_2d(x, y, x0, y0, sigma):
    """Generate a 2D Gaussian distribution."""
    return np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))


def clean_fluorophore_name(name):
    """Clean and format fluorophore name by removing common prefixes and suffixes."""
    name = name.replace(" - Em", "")
    name = name.replace("FocalCheck ", "FC ")
    name = name.replace(" Ring", "")
    name = name.replace("Double", "")
    return name


def simulate_psf(
    spectral_data,
    poly3_coeffs,
    poly1_coeffs,
    filter_data,
    camera_eff_curve,
    crop_size,
    red_shift,
    sigma=1.25,
):
    """Simulate PSFs based on spectral data, polynomial fits, and camera efficiency."""
    psf_image = np.zeros(crop_size)
    x = np.arange(crop_size[1])
    y = np.arange(crop_size[0])
    xv, yv = np.meshgrid(x, y)

    for wl, intensity in spectral_data:
        # Evaluate polynomial fits
        x_shift = np.polyval(poly3_coeffs, wl)
        y_shift = np.polyval(poly1_coeffs, x_shift)

        # Interpolate filter transmission and camera efficiency
        filter_transmission = np.interp(wl, filter_data[:, 0], filter_data[:, 1])
        camera_efficiency = np.interp(
            wl, camera_eff_curve[:, 0], camera_eff_curve[:, 1]
        )

        if filter_transmission > 0 and camera_efficiency > 0:
            x0 = -(red_shift - 4) + x_shift
            y0 = y_shift - 1

            gaussian = gaussian_2d(xv, yv, x0, y0, sigma)
            psf_image += gaussian * intensity * filter_transmission * camera_efficiency

    return psf_image


def process_all_spectral_files(
    base_folder,
    poly3_coeffs,
    poly1_coeffs,
    filter_data,
    camera_eff_curve,
    crop_size,
    red_shift=4,
    sigma=1.25,
    wl_lim=(400, 900),
):
    """Process all spectral files and calculate spectral PSFs."""
    em_files = load_spectrum_files(base_folder, "Em.txt")
    psf_results = {}
    max_emission_wl = []

    for file_path in em_files:
        # Read and filter spectral data
        spectral_data = read_spectrum(file_path)
        spectral_data = spectral_data[
            (spectral_data[:, 0] >= wl_lim[0]) & (spectral_data[:, 0] <= wl_lim[1])
        ]

        # Calculate maximum emission wavelength
        max_emission_wl.append(
            np.where(spectral_data[:, 1] == max(spectral_data[:, 1]))[0]
        )

        # Generate PSF image
        psf_image = simulate_psf(
            spectral_data,
            poly3_coeffs,
            poly1_coeffs,
            filter_data,
            camera_eff_curve,
            crop_size,
            red_shift,
            sigma,
        )

        # Store results with cleaned fluorophore name
        fluorophore_name = os.path.splitext(os.path.basename(file_path))[0]
        fluorophore_name = clean_fluorophore_name(fluorophore_name)
        psf_results[fluorophore_name] = psf_image

    return psf_results, np.concatenate(max_emission_wl)


def get_camera_efficiency_curve():
    """Define and return the camera efficiency curve."""
    return np.array(
        [
            [400, 0.7],
            [450, 0.85],
            [500, 0.88],
            [600, 0.9],
            [650, 0.92],
            [700, 0.9],
            [750, 0.85],
            [800, 0.75],
            [850, 0.6],
            [900, 0.43],
        ]
    )


def simulate_psfs(
    crop_size=(9, 19),
    red_shift=4,
    sigma=1.25,
    base_folder="/DATA2/Data_CoCoS_HD/COCOS_ISM/Matlab files",
    filter_file="/DATA2/Data_CoCoS_HD/COCOS_ISM/Matlab files/FF01-440_521_607_700.txt",
    mat_file_processed="/DATA2/Data_CoCoS_HD/COCOS_ISM/Matlab files/DispersionFitsProcessed.mat",
):
    """Main function to simulate PSFs with given parameters."""

    # Load required data
    filter_data = read_spectrum(filter_file)
    processed_data = loadmat(mat_file_processed)
    camera_eff_vals = get_camera_efficiency_curve()

    # Extract polynomial coefficients
    poly3_coeffs = processed_data["coeffs_PixToWl"].flatten()
    poly1_coeffs = processed_data["coeffs_Y"].flatten()

    # Process all spectral files
    psf_results, max_emission_wl = process_all_spectral_files(
        base_folder,
        poly3_coeffs,
        poly1_coeffs,
        filter_data,
        camera_eff_vals,
        crop_size,
        red_shift=red_shift,
        sigma=sigma,
    )

    # Sort results by emission wavelength
    psfs = np.asarray(list(psf_results.values()))
    fl_names = list(psf_results.keys())
    ind_sorted = np.argsort(max_emission_wl)[::-1]
    psfs = psfs[ind_sorted]
    fl_names = [fl_names[ind] for ind in ind_sorted]

    return psfs.transpose(1, 2, 0), fl_names
