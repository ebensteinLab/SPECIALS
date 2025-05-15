import time

import cupy as cp
import cupyx
import numpy as np
from cupyx.scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.ndimage import zoom
from cupyx.scipy.ndimage import convolve
from cupyx.scipy.signal import fftconvolve


def sparse_deconvolution_ista_centered(
    ism_image, psf, lambda_reg=0.1, num_iterations=50
):
    """
    Perform sparse deconvolution using ISTA with centered FFT.

    Args:
        ism_image (numpy.ndarray): Observed ISM image of shape (height, width, num_channels).
        psf (numpy.ndarray): Point spread function (PSF), shape (9, 19).
        lambda_reg (float): Regularization parameter controlling sparsity.
        num_iterations (int): Number of iterations for ISTA.

    Returns:
        numpy.ndarray: Reconstructed sparse image, shape (height, width, num_channels).
    """

    ism_image = cp.asarray(ism_image)
    psf = cp.asarray(psf)

    # Dimensions
    height, width, num_channels = ism_image.shape

    # Pad the PSF to match the ISM image dimensions
    psf_padded = cp.zeros((height, width), dtype=psf.dtype)
    psf_padded[
        height // 2 - psf.shape[0] // 2 : height // 2 + psf.shape[0] // 2 + 1,
        width // 2 - psf.shape[1] // 2 : width // 2 + psf.shape[1] // 2 + 1,
    ] = psf

    # Center the PSF in the spatial domain before FFT
    psf_padded_centered = fftshift(psf_padded)

    # Fourier transform of the centered PSF
    psf_ft = fft2(psf_padded_centered)
    psf_ft_conj = cp.conj(psf_ft)

    # Compute the step size for ISTA
    lipschitz_constant = cp.max(cp.abs(psf_ft) ** 2)
    step_size = 1.0 / lipschitz_constant

    # Initialize the reconstructed image
    reconstructed_image = cp.zeros_like(ism_image)

    # ISTA iterations
    for iteration in range(num_iterations):
        for channel in range(num_channels):
            # Fourier transform of the current estimate
            current_estimate_ft = fft2(fftshift(reconstructed_image[:, :, channel]))

            # Gradient of the data fidelity term
            fidelity_gradient_ft = psf_ft_conj * (
                current_estimate_ft * psf_ft - fft2(fftshift(ism_image[:, :, channel]))
            )
            fidelity_gradient = ifftshift(ifft2(fidelity_gradient_ft).real)

            # Update the reconstructed image with gradient descent
            updated_image = (
                reconstructed_image[:, :, channel] - step_size * fidelity_gradient
            )

            # Apply soft thresholding for sparsity
            reconstructed_image[:, :, channel] = cp.sign(updated_image) * cp.maximum(
                cp.abs(updated_image) - lambda_reg * step_size, 0
            )

        # Optionally monitor convergence (loss, PSNR, etc.)

    return cp.asnumpy(reconstructed_image)


def decompose_to_channels_batched_only_gpu(
    crops_with_high_intensity,
    psfs,
    rounded_peaks,
    number_of_rl_iters=30,
    rl_batch_size=200000,
    scatter_add_batch_size=100000,
    should_compute_final_image=True,
    number_of_channels=3,
):
    """
    Builds the image from the crops.
    Uses GPU (cupy), and has batching (both for the richardson_lucy, and for adding all the convolved crops.
    Leaving the random prints of time, so that you can check if there are any bottlenecks on your hardware.
    The first run does some initialization, so will be a bit slower.
    """
    # Prepare PSF
    t = time.time()
    psf_2d = psfs[:, :, 0]
    pad_size = ((psf_2d.shape[0], psf_2d.shape[0]), (psf_2d.shape[1], psf_2d.shape[1]))
    padded = np.pad(psf_2d, pad_size, mode="constant")
    new_psf = zoom(padded, 0.5)
    new_psf = new_psf / np.sum(new_psf)
    new_psf_gpu = cp.asarray(new_psf)[None, :]

    # Prepare output array
    decomposed_image = cp.zeros((512, 512, number_of_channels))
    all_ch_crops = cp.zeros(
        (
            crops_with_high_intensity.shape[0],
            crops_with_high_intensity.shape[1],
            crops_with_high_intensity.shape[2],
            3,
        )
    )
    # Prepare indexing arrays
    rows = rounded_peaks[:, 1, None, None] + np.arange(-4, 5)[None, :, None]
    cols = rounded_peaks[:, 0, None, None] + np.arange(-12, 6)[None, None, :]
    rows, cols = rows.astype(int), cols.astype(int)
    for ch in range(number_of_channels):
        print(f"Channel {ch + 1}/{number_of_channels} started")
        deconvolved_images = richardson_lucy_batched_gpu(
            crops_with_high_intensity,
            psfs[:, :, ch],
            num_iter=number_of_rl_iters,
            clip=False,
            batch_size=rl_batch_size,
        )

        start_col, end_col = (
            (deconvolved_images.shape[2] - 3) // 2,
            (deconvolved_images.shape[2] + 3) // 2,
        )  # array operations
        center_columns = deconvolved_images.copy()
        center_columns[:, :, :start_col] = 0
        center_columns[:, :, end_col:] = 0

        print(time.time() - t)
        chCrops = convolve(center_columns, new_psf_gpu)  # done on GPU, all pics at once
        print(time.time() - t)
        all_ch_crops[:, :, :, ch] = chCrops
        print(time.time() - t)
        if should_compute_final_image:
            for i in range(0, len(chCrops), scatter_add_batch_size):
                batch_idx = slice(i, min(i + scatter_add_batch_size, len(chCrops)))
                cupyx.scatter_add(
                    decomposed_image[:, :, ch],
                    (rows[batch_idx], cols[batch_idx]),
                    chCrops[batch_idx],
                )  # adding many pics at once, on gpu - way faster than looping.
            print(time.time() - t)

        # Clear GPU memory
        del center_columns, chCrops
        print(time.time() - t)
        print(f"Channel {ch + 1}/3 completed")

    print(f"Total processing time: {time.time() - t:.2f} seconds")
    return decomposed_image, all_ch_crops


def richardson_lucy_batched_gpu(
    images, psf, num_iter=50, clip=False, filter_epsilon=None, batch_size=100
):
    """
    Our implementation for richardson_lucy. Based on skimage, but runs on gpu using cupy and does many images at once.
    Also implemented batching, to solve out of memory problems (runs fine on my graphics card - 2060).
    Please make sure the code review for this function is strict, so we are sure it is correct (:
    """
    float_type = cp.float32
    psf_gpu = cp.asarray(psf, dtype=float_type)

    # Ensure psf is 3D (1, width, height)
    if psf_gpu.ndim == 2:
        psf_gpu = psf_gpu[cp.newaxis, :, :]

    psf_mirror = cp.flip(psf_gpu, axis=(1, 2))
    eps = cp.finfo(float_type).eps

    # Process images in batches
    number_of_images_images = len(images)
    im_deconv_list = []
    for start_idx in range(0, number_of_images_images, batch_size):
        end_idx = min(start_idx + batch_size, number_of_images_images)
        print(end_idx)
        batch_images = cp.asarray(images[start_idx:end_idx], dtype=float_type)

        im_deconv = cp.full(batch_images.shape, 0.5, dtype=float_type)

        for _ in range(num_iter):
            conv = fftconvolve(im_deconv, psf_gpu, mode="same", axes=(1, 2)) + eps

            if filter_epsilon:
                relative_blur = cp.where(conv < filter_epsilon, 0, batch_images / conv)
            else:
                relative_blur = batch_images / conv

            im_deconv *= fftconvolve(
                relative_blur, psf_mirror, mode="same", axes=(1, 2)
            )

        if clip:
            cp.clip(im_deconv, -1, 1, out=im_deconv)

        im_deconv_list.append(im_deconv)

        # Clear GPU memory
        del batch_images, conv, relative_blur, im_deconv
    # Concatenate all batches
    return cp.concatenate(im_deconv_list, axis=0)
