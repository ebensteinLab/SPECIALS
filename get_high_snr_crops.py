import cupyx
import cupy as cp
import numpy as np

from utils import flatten

SNR_THRESHOLD = 0.1


def estimate_noise_variance_gpu(crops):
    """
    Estimate noise variance in the data, combining Poisson and Gaussian components.
    Args:
        data: 3D numpy array (noisy data - not normalized).
    Returns:
        Estimated noise variance.
    """

    sigma = 0.8
    data = cp.asarray(crops, dtype=cp.float32)
    dims = data.ndim

    # Poisson noise: mean signal level
    poisson_var = cp.mean(data, axis=(-2, -1))

    # # # Gaussian noise: residual variance after smoothing
    if dims == 3:
        border_mask = cp.ones_like(data[0, ...], dtype=bool)
    else:
        border_mask = cp.ones_like(data, dtype=bool)
    #
    border_mask[1:-1, 1:-1] = False
    smoothedX = cupyx.scipy.ndimage.gaussian_filter1d(
        data, sigma=sigma, axis=-2
    )  # Adjust sigma if needed
    smoothed = cupyx.scipy.ndimage.gaussian_filter1d(
        smoothedX, sigma=sigma, axis=-1
    )  # Adjust sigma if needed

    if dims == 3:
        border_mean = cp.mean(data[:, border_mask], axis=1)
        data_mean = cp.mean(data, axis=(-2, -1))
        bad_ind = cp.where(border_mean > data_mean)[0]
        residuals = data[:, border_mask] - smoothed[:, border_mask]
    else:
        residuals = data[border_mask] - smoothed[border_mask]

    # residuals=data-smoothed
    gaussian_var = cp.var(residuals, axis=(-2, -1))
    snr = cp.max(smoothed, axis=(-2, -1)) / (poisson_var + gaussian_var)
    snr[bad_ind] = 0.0

    # Total variance is the sum of both components
    return (poisson_var + gaussian_var**2).get(), snr.get()


def get_high_snr_crops(all_crops, peaks):
    noise, snr = estimate_noise_variance_gpu(all_crops)
    flattened_peaks = np.array(flatten(peaks))
    high_snr_ind = np.where(snr > SNR_THRESHOLD)[0]
    peaks_with_high_intensity = flattened_peaks[high_snr_ind]
    crops_with_high_intensity = all_crops[high_snr_ind]
    return peaks_with_high_intensity, crops_with_high_intensity
