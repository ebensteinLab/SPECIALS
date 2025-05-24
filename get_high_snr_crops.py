import numpy as np

from compute_backend import backend
from utils import flatten

SNR_THRESHOLD = 0.1


def estimate_noise_variance_gpu(crops):
    """
    Estimate noise variance in the data, combining Poisson and Gaussian components.
    Automatically uses GPU if available, falls back to CPU otherwise.

    Args:
        crops: 3D numpy array (noisy data - not normalized).
    Returns:
        Tuple of (estimated noise variance, SNR values) as numpy arrays.
    """
    sigma = 0.8
    data = backend.asarray(crops, dtype=np.float32)
    dims = data.ndim

    # Poisson noise: mean signal level
    poisson_var = backend.mean(data, axis=(-2, -1))

    # Gaussian noise: residual variance after smoothing
    if dims == 3:
        border_mask = backend.ones_like(data[0, ...], dtype=bool)
    else:
        border_mask = backend.ones_like(data, dtype=bool)

    border_mask[1:-1, 1:-1] = False

    # Apply Gaussian filtering in both directions
    smoothedX = backend.gaussian_filter1d(data, sigma=sigma, axis=-2)
    smoothed = backend.gaussian_filter1d(smoothedX, sigma=sigma, axis=-1)

    if dims == 3:
        border_mean = backend.mean(data[:, border_mask], axis=1)
        data_mean = backend.mean(data, axis=(-2, -1))
        bad_ind = backend.where(border_mean > data_mean, True, False)[0]
        residuals = data[:, border_mask] - smoothed[:, border_mask]
    else:
        residuals = data[border_mask] - smoothed[border_mask]
        bad_ind = backend.asarray([])  # Empty array for consistency

    # Calculate Gaussian variance from residuals
    gaussian_var = backend.var(residuals, axis=(-2, -1))

    # Calculate SNR
    snr = backend.max(smoothed, axis=(-2, -1)) / (poisson_var + gaussian_var)

    # Set bad indices to 0 SNR
    if dims == 3 and len(bad_ind) > 0:
        snr_copy = snr.copy()
        for idx in bad_ind:
            snr_copy[idx] = 0.0
        snr = snr_copy

    # Total variance is the sum of both components
    total_variance = poisson_var + gaussian_var ** 2

    return backend.asnumpy(total_variance), backend.asnumpy(snr)


def get_high_snr_crops(all_crops, peaks):
    """
    Filter crops and peaks based on SNR threshold.
    Works with both GPU and CPU backends automatically.

    Args:
        all_crops: Array of image crops
        peaks: List of peak coordinates

    Returns:
        Tuple of (filtered peaks, filtered crops) with high SNR
    """
    noise, snr = estimate_noise_variance_gpu(all_crops)
    flattened_peaks = np.array(flatten(peaks))
    high_snr_ind = np.where(snr > SNR_THRESHOLD)[0]
    peaks_with_high_intensity = flattened_peaks[high_snr_ind]
    crops_with_high_intensity = all_crops[high_snr_ind]
    return peaks_with_high_intensity, crops_with_high_intensity