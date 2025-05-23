import os

import numpy as np
import pandas as pd
from scipy.io import loadmat

from utils import CROP_SIZE


def get_optimal_red_shift(max_emission_wl, poly3_coeffs, selected_indices):
    max_em_selected = max_emission_wl[selected_indices]
    max_x_selected = np.polyval(poly3_coeffs, max_em_selected)
    min_edge_dist = CROP_SIZE[1] - np.min(max_x_selected)
    max_edge_dist = CROP_SIZE[1] - np.max(max_x_selected)
    red_shift = np.round(np.mean([min_edge_dist, max_edge_dist]))
    return red_shift


def read_spectrum(file_path):
    """Read a spectrum file into a NumPy array using pandas to handle complex formatting."""
    try:
        df = pd.read_csv(file_path, header=None, comment="%", delimiter="\t")
        return df.values
    except Exception as e:
        raise ValueError(f"Error reading spectrum file {file_path}: {e}")


def camera_QE_curve_vals():
    # Define camera efficiency curve
    camera_eff_vals = np.array(
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
    return camera_eff_vals

    # Simulate PSFs for all spectral files
    # crop_size = (9, 19)


def simulate_psf(
    spectrum,
    poly3_coeffs,
    poly1_coeffs,
    filter_data,
    camera_eff_curve,
    crop_size=(9, 19),
    red_shift=4,
    sigma=1.25,
):
    def gaussian_2d(x, y, x0, y0, sigma):
        """Generate a 2D Gaussian distribution."""
        return np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))

    """Simulate PSFs based on spectral data, polynomial fits, and camera efficiency."""
    psf_image = np.zeros(crop_size)
    x = np.arange(crop_size[1])
    y = np.arange(crop_size[0])
    xv, yv = np.meshgrid(x, y)

    for wl, intensity in spectrum:
        # Evaluate polynomial fits
        x_shift = np.polyval(poly3_coeffs, wl)
        y_shift = np.polyval(poly1_coeffs, x_shift)

        # Interpolate filter transmission and camera efficiency
        filter_transmission = np.interp(wl, filter_data[:, 0], filter_data[:, 1])
        camera_efficiency = np.interp(
            wl, camera_eff_curve[:, 0], camera_eff_curve[:, 1]
        )

        if (
            filter_transmission > 0 and camera_efficiency > 0
        ):  # Process only valid wavelengths
            x0 = -(red_shift - 4) + x_shift
            y0 = y_shift - 1

            gaussian = gaussian_2d(xv, yv, x0, y0, sigma)
            psf_image += gaussian * intensity * filter_transmission * camera_efficiency

    return psf_image


def simulate_selected_psfs(
    spectral_data_all,
    selected_indices,
    base_folder,
    filter_file,
    mat_file_processed,
    crop_size=(9, 19),
    red_shift=4,
    sigma=1.25,
):
    filter_data, poly3_coeffs, poly1_coeffs = get_filter_and_dispersion_curve(
        base_folder, filter_file, mat_file_processed
    )
    camera_eff_vals = camera_QE_curve_vals()
    psfs = []
    for i in selected_indices:
        psf_image = simulate_psf(
            spectral_data_all[i],
            poly3_coeffs,
            poly1_coeffs,
            filter_data,
            camera_eff_vals,
            crop_size,
            red_shift=red_shift,
            sigma=sigma,
        )
        psfs.append(psf_image)
    psfs = np.asarray(psfs)
    return psfs.transpose(1, 2, 0)


def get_fluorophores_data(base_folder, wl_lim=(400, 900)):
    def load_spectrum_files(base_folder, pattern):
        """Load spectrum files matching a given pattern."""
        spectrum_files = []
        for root, _, files in os.walk(base_folder):
            for file in files:
                if file.endswith(pattern):
                    spectrum_files.append(os.path.join(root, file))
        return spectrum_files

    em_files = load_spectrum_files(base_folder, "Em.txt")
    max_emission_wl = []
    fluorophore_names = []
    spectral_data_all = []
    for file_path in em_files:
        spectral_data = read_spectrum(file_path)

        # Filter spectral data to within wavelength limits
        spectral_data = spectral_data[
            (spectral_data[:, 0] >= wl_lim[0]) & (spectral_data[:, 0] <= wl_lim[1])
        ]
        max_emission_wl.append(
            spectral_data[
                np.where(spectral_data[:, 1] == max(spectral_data[:, 1]))[0], 0
            ]
        )
        fluorophore_name = os.path.splitext(os.path.basename(file_path))[0].replace(
            " - Em", ""
        )
        fluorophore_name = fluorophore_name.replace("FocalCheck ", "FC ")
        fluorophore_name = fluorophore_name.replace(" Ring", "")
        fluorophore_name = fluorophore_name.replace("Double", "")
        fluorophore_names.append(fluorophore_name)
        spectral_data_all.append(spectral_data)
    max_emission_wl = np.concatenate(max_emission_wl)
    ind_sorted = np.argsort(
        max_emission_wl,
    )
    # fluorophore_names=np.array(fluorophore_names)[ind_sorted]
    fluorophore_names[:] = [fluorophore_names[ind] for ind in ind_sorted[::-1]]
    spectral_data_all = [spectral_data_all[ind] for ind in ind_sorted[::-1]]
    max_emission_wl = max_emission_wl[ind_sorted[::-1]]
    return fluorophore_names, max_emission_wl, spectral_data_all


def get_filter_and_dispersion_curve(base_folder, filter_file, mat_file_processed):
    filter_file_path = os.path.join(base_folder, filter_file)
    mat_file_processed_path = os.path.join(base_folder, mat_file_processed)
    # MATLAB file containing polynomial coefficients for the wavelength
    # dependent dispersion in X (poly3 coefficients - 'coeffs_PixToWl')
    # and linear shift in Y (poly1 coefficients - 'coeffs_Y').

    # Load data
    filter_data = read_spectrum(filter_file_path)
    processed_data = loadmat(mat_file_processed_path)

    # Extract polynomial coefficients for dispersion fits
    poly3_coeffs = processed_data["coeffs_PixToWl"].flatten()
    poly1_coeffs = processed_data["coeffs_Y"].flatten()
    return filter_data, poly3_coeffs, poly1_coeffs
